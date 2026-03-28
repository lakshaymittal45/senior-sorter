import concurrent.futures
import io
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np
import pillow_heif
try:
    import rawpy
except Exception:
    rawpy = None
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image
from tqdm import tqdm

pillow_heif.register_heif_opener()

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

FOLDER_LINK_PATTERNS = [
    re.compile(r"https?://drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)"),
    re.compile(r"https?://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)"),
]
FILE_LINK_PATTERNS = [
    re.compile(r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)"),
    re.compile(r"https?://drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)"),
]


@dataclass
class ProcessingConfig:
    max_image_side: int
    face_distance_threshold: float
    relaxed_face_distance_threshold: float
    min_relaxed_hits: int
    min_strict_hits: int
    blur_tolerance: int
    process_extensions: List[str]
    workers: int
    min_partial_face_score: float
    enable_partial_face_mode: bool
    sample_encoding_jitters: int
    candidate_encoding_jitters: int
    face_detector_model: str
    face_upsample_times: int
    enable_detector_fallback: bool
    mean_topk_margin: float
    center_distance_margin: float
    strong_match_distance: float
    min_candidate_face_area_ratio: float
    sample_outlier_percentile: float
    min_sample_encodings: int
    # --- new robustness fields ---
    sample_detector_model: str = "cnn"          # CNN for better sample accuracy
    enable_tiling: bool = True                   # tile large images to find small faces
    tile_size: int = 800                         # px per tile side
    tile_overlap: float = 0.25                   # fractional overlap between tiles
    rotation_angles: str = "-15,15,-30,30"       # comma-sep angles for augment


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_drive_service(credentials_json: str, token_json: str):
    creds = None
    if os.path.exists(token_json):
        creds = Credentials.from_authorized_user_file(token_json, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_json, SCOPES)
            creds = flow.run_local_server(port=53682, authorization_prompt_message="Login to Google", success_message="Login complete. You may close this window.")
        with open(token_json, "w", encoding="utf-8") as token:
            token.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def read_drive_links(file_path: Path) -> List[str]:
    links = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                links.append(line)
    return links


def extract_id_from_link(link: str) -> Tuple[Optional[str], Optional[str]]:
    for pattern in FOLDER_LINK_PATTERNS:
        match = pattern.search(link)
        if match:
            return match.group(1), "folder"

    for pattern in FILE_LINK_PATTERNS:
        match = pattern.search(link)
        if match:
            return match.group(1), "file"

    if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", link):
        return link, "unknown"

    return None, None


def is_supported_image(filename: str, extensions: List[str]) -> bool:
    suffix = Path(filename).suffix.lower().replace(".", "")
    return suffix in extensions


def get_file_metadata(service, file_id: str) -> Optional[Dict]:
    try:
        return (
            service.files()
            .get(
                fileId=file_id,
                fields="id,name,mimeType,parents,shortcutDetails,webViewLink",
                supportsAllDrives=True,
            )
            .execute()
        )
    except HttpError:
        return None


def resolve_shortcut_target(service, file_meta: Dict) -> Dict:
    if file_meta.get("mimeType") == "application/vnd.google-apps.shortcut":
        target_id = file_meta.get("shortcutDetails", {}).get("targetId")
        if target_id:
            target_meta = get_file_metadata(service, target_id)
            if target_meta:
                return target_meta
    return file_meta


def list_folder_images(service, folder_id: str, extensions: List[str]) -> List[Dict]:
    queue = [folder_id]
    visited_folders = set()
    image_files = []

    while queue:
        current = queue.pop()
        if current in visited_folders:
            continue
        visited_folders.add(current)

        page_token = None
        while True:
            response = (
                service.files()
                .list(
                    q=f"'{current}' in parents and trashed=false",
                    fields="nextPageToken, files(id,name,mimeType,shortcutDetails,webViewLink)",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    corpora="allDrives",
                    pageToken=page_token,
                    pageSize=1000,
                )
                .execute()
            )

            for item in response.get("files", []):
                item = resolve_shortcut_target(service, item)
                mime = item.get("mimeType", "")
                name = item.get("name", "")

                if mime == "application/vnd.google-apps.folder":
                    queue.append(item["id"])
                    continue

                if mime.startswith("image/") or is_supported_image(name, extensions):
                    image_files.append(item)

            page_token = response.get("nextPageToken")
            if not page_token:
                break

    return image_files


def gather_candidate_images(service, links: List[str], extensions: List[str]) -> List[Dict]:
    all_candidates: List[Dict] = []
    seen = set()

    for link in links:
        target_id, kind = extract_id_from_link(link)
        if not target_id:
            continue

        meta = get_file_metadata(service, target_id)
        if not meta:
            continue

        meta = resolve_shortcut_target(service, meta)
        mime = meta.get("mimeType", "")

        if kind == "file" or (kind == "unknown" and mime != "application/vnd.google-apps.folder"):
            if mime.startswith("image/") or is_supported_image(meta.get("name", ""), extensions):
                if meta["id"] not in seen:
                    seen.add(meta["id"])
                    tagged = dict(meta)
                    tagged["source_link"] = link
                    all_candidates.append(tagged)
            continue

        if mime == "application/vnd.google-apps.folder":
            folder_files = list_folder_images(service, meta["id"], extensions)
            for f in folder_files:
                if f["id"] in seen:
                    continue
                seen.add(f["id"])
                tagged = dict(f)
                tagged["source_link"] = link
                all_candidates.append(tagged)

    return all_candidates


def _pil_decode_worker(raw_bytes, result_path):
    """Runs in a separate process so it can be force-killed if PIL/HEIC hangs."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
            rgb = np.array(pil_img.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            np.save(result_path, bgr)
    except Exception:
        pass


def _pil_decode_with_timeout(raw: bytes, timeout: int = 6) -> Optional[np.ndarray]:
    """Decode via PIL in a child process with a hard kill timeout."""
    import multiprocessing, tempfile
    result_path = Path(tempfile.mktemp(suffix=".npy"))
    proc = multiprocessing.Process(target=_pil_decode_worker, args=(raw, str(result_path)))
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=2)
    if result_path.exists():
        try:
            arr = np.load(str(result_path))
            return arr
        except Exception:
            return None
        finally:
            try:
                result_path.unlink()
            except Exception:
                pass
    return None


def decode_image_bytes(raw: bytes) -> Optional[np.ndarray]:
    data = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is not None:
        return img_bgr

    # Fallback path for formats OpenCV may not decode (for example HEIC/HEIF).
    # Runs in isolated subprocess so a hung C++ HEIC decoder can be force-killed.
    result = _pil_decode_with_timeout(raw, timeout=6)
    if result is not None:
        return result

    # RAW fallback path (for example .NEF) if rawpy is available.
    if rawpy is not None:
        try:
            with rawpy.imread(io.BytesIO(raw)) as raw_img:
                rgb = raw_img.postprocess(use_camera_wb=True, no_auto_bright=True)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    return None


def download_drive_file_raw(service, file_id: str) -> Optional[bytes]:
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    try:
        while not done:
            _, done = downloader.next_chunk()
    except HttpError:
        return None
    return buffer.getvalue()


def download_drive_image(service, file_id: str) -> Optional[np.ndarray]:
    raw = download_drive_file_raw(service, file_id)
    if raw:
        return decode_image_bytes(raw)
    return None


def load_image_cv(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is not None:
        return img

    # Fallback for local HEIC/HEIF and other formats unsupported by cv2.imread.
    try:
        return decode_image_bytes(path.read_bytes())
    except Exception:
        return None


def resize_if_needed(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img

    scale = max_side / float(longest)
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def variance_of_laplacian(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _parse_rotation_angles(val) -> List[float]:
    """Parse a comma-separated string or list of rotation angles."""
    if not val:
        return []
    if isinstance(val, (list, tuple)):
        return [float(v) for v in val if v != 0]
    return [float(v.strip()) for v in str(val).split(",") if v.strip() and float(v.strip()) != 0]


def preprocess_variants(img_bgr: np.ndarray, blur_tolerance: int, rotation_angles=None) -> List[np.ndarray]:
    variants = [img_bgr]

    blur_score = variance_of_laplacian(img_bgr)
    if blur_score < blur_tolerance:
        # Sharpen + local contrast can recover weak face details in blurry images.
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharp = cv2.filter2D(img_bgr, -1, kernel)
        variants.append(sharp)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    variants.append(enhanced)

    flipped = cv2.flip(img_bgr, 1)
    variants.append(flipped)

    # Rotation augmentation — detects faces at angles (tilted / profile-like).
    angles = _parse_rotation_angles(rotation_angles) if rotation_angles is not None else [-15, 15, -30, 30]
    h, w = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        variants.append(rotated)
        # also flip the rotated variant to double coverage
        variants.append(cv2.flip(rotated, 1))

    return variants


def tile_image(img_bgr: np.ndarray, tile_size: int, overlap: float) -> List[np.ndarray]:
    """Split a large image into overlapping tiles for detecting small faces."""
    h, w = img_bgr.shape[:2]
    # Only worth tiling if the image is substantially larger than one tile.
    if max(h, w) < tile_size * 1.4:
        return []
    stride = max(1, int(tile_size * (1.0 - overlap)))
    tiles = []
    y = 0
    while y < h:
        x = 0
        while x < w:
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            tile = img_bgr[y:y2, x:x2]
            tiles.append(tile)
            if x2 == w:
                break
            x += stride
        if y2 == h:
            break
        y += stride
    return tiles


def extract_face_encodings(
    img_bgr: np.ndarray,
    detector_model: str = "hog",
    num_jitters: int = 1,
    upsample_times: int = 1,
    enable_detector_fallback: bool = True,
    min_face_area_ratio: float = 0.0,
    enable_tiling: bool = False,
    tile_size: int = 800,
    tile_overlap: float = 0.25,
) -> List[np.ndarray]:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detector_order = [detector_model]
    if enable_detector_fallback:
        detector_order.extend([m for m in ("hog", "cnn") if m != detector_model])

    upsample_attempts = [max(0, int(upsample_times))]
    if upsample_attempts[0] < 2:
        upsample_attempts.append(upsample_attempts[0] + 1)

    def _detect_in_rgb(rgb_img):
        h, w = rgb_img.shape[:2]
        locs = []
        for model_name in detector_order:
            for upsample in upsample_attempts:
                try:
                    locs = face_recognition.face_locations(
                        rgb_img,
                        number_of_times_to_upsample=upsample,
                        model=model_name,
                    )
                except Exception:
                    locs = []
                if locs:
                    break
            if locs:
                break
        if not locs:
            return []
        filtered = []
        for top, right, bottom, left in locs:
            area = max(0, bottom - top) * max(0, right - left)
            ratio = area / float(max(1, h * w))
            if ratio >= min_face_area_ratio:
                filtered.append((top, right, bottom, left))
        if not filtered:
            return []
        return face_recognition.face_encodings(rgb_img, known_face_locations=filtered, num_jitters=num_jitters)

    encodings = _detect_in_rgb(rgb)

    # If no faces found on the full image, try overlapping tiles (helps with
    # distant faces in large group photos).
    if not encodings and enable_tiling:
        tiles = tile_image(img_bgr, tile_size, tile_overlap)
        seen_encs: List[np.ndarray] = []
        for tile in tiles:
            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            tile_encs = _detect_in_rgb(tile_rgb)
            for enc in tile_encs:
                # Deduplicate: skip if very similar to an already-collected encoding.
                if seen_encs:
                    dists = face_recognition.face_distance(seen_encs, enc)
                    if float(np.min(dists)) < 0.35:
                        continue
                seen_encs.append(enc)
        encodings = seen_encs

    return list(encodings)


def compute_partial_face_score(known_encodings: List[np.ndarray], candidate_encoding: np.ndarray) -> float:
    distances = face_recognition.face_distance(known_encodings, candidate_encoding)
    if len(distances) == 0:
        return 0.0

    best = float(np.min(distances))
    # Convert distance to a confidence-like score in [0,1], where smaller distance is better.
    score = max(0.0, min(1.0, 1.0 - best))
    return score


def prune_sample_outliers(
    encodings: List[np.ndarray],
    outlier_percentile: float,
    min_sample_encodings: int,
) -> List[np.ndarray]:
    if len(encodings) <= max(min_sample_encodings, 3):
        return encodings

    mat = np.vstack(encodings)
    center = np.mean(mat, axis=0)
    d = np.linalg.norm(mat - center, axis=1)
    cutoff = float(np.percentile(d, outlier_percentile))
    kept = [enc for enc, dist in zip(encodings, d) if float(dist) <= cutoff]

    if len(kept) < min_sample_encodings:
        order = np.argsort(d)
        top_idx = order[:min_sample_encodings]
        kept = [encodings[int(i)] for i in top_idx]

    return kept


def is_match(
    known_encodings: List[np.ndarray],
    img_bgr: np.ndarray,
    threshold: float,
    relaxed_threshold: float,
    min_strict_hits: int,
    min_relaxed_hits: int,
    min_partial_face_score: float,
    enable_partial_face_mode: bool,
    blur_tolerance: int,
    candidate_encoding_jitters: int,
    detector_model: str,
    face_upsample_times: int,
    enable_detector_fallback: bool,
    mean_topk_margin: float,
    center_distance_margin: float,
    strong_match_distance: float,
    min_candidate_face_area_ratio: float,
    rotation_angles=None,
    enable_tiling: bool = True,
    tile_size: int = 800,
    tile_overlap: float = 0.25,
) -> Tuple[bool, float]:
    """Tiered face matching — fast path first, expensive augmentations only if needed."""
    known_center = np.mean(np.vstack(known_encodings), axis=0)
    sample_center_distances = np.linalg.norm(np.vstack(known_encodings) - known_center, axis=1)
    dynamic_center_limit = float(np.percentile(sample_center_distances, 90)) + center_distance_margin
    dynamic_center_limit = max(dynamic_center_limit, threshold + center_distance_margin)

    best_distance = 999.0
    best_partial_score = 0.0
    best_center_distance = 999.0
    strict_hits = 0
    relaxed_hits = 0

    def _extract(variant, use_tiling=False):
        return extract_face_encodings(
            variant,
            detector_model=detector_model,
            num_jitters=candidate_encoding_jitters,
            upsample_times=face_upsample_times,
            enable_detector_fallback=enable_detector_fallback,
            min_face_area_ratio=min_candidate_face_area_ratio,
            enable_tiling=use_tiling,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )

    def _score_variant(variant, use_tiling=False):
        nonlocal best_distance, best_partial_score, best_center_distance
        nonlocal strict_hits, relaxed_hits
        candidate_encodings = _extract(variant, use_tiling)
        variant_strict = False
        variant_relaxed = False
        for candidate in candidate_encodings:
            distances = face_recognition.face_distance(known_encodings, candidate)
            if len(distances) == 0:
                continue
            d = float(np.min(distances))
            k = min(3, len(distances))
            mean_topk = float(np.mean(np.partition(distances, k - 1)[:k]))
            center_d = float(np.linalg.norm(candidate - known_center))
            if d < best_distance:
                best_distance = d
            if center_d < best_center_distance:
                best_center_distance = center_d
            # STRONG MATCH — return immediately, skip everything else.
            if d <= strong_match_distance and center_d <= (dynamic_center_limit + center_distance_margin):
                return True, d
            if d <= threshold and mean_topk <= (threshold + mean_topk_margin) and center_d <= dynamic_center_limit:
                variant_strict = True
            if d <= relaxed_threshold and mean_topk <= (relaxed_threshold + mean_topk_margin) and center_d <= (dynamic_center_limit + center_distance_margin):
                variant_relaxed = True
            if enable_partial_face_mode:
                ps = compute_partial_face_score(known_encodings, candidate)
                best_partial_score = max(best_partial_score, ps)

        if variant_strict:
            strict_hits += 1
            if strict_hits >= min_strict_hits:
                return True, min(best_distance, best_center_distance)
        if variant_relaxed:
            relaxed_hits += 1
            if relaxed_hits >= min_relaxed_hits and best_distance <= relaxed_threshold:
                return True, min(best_distance, best_center_distance)
        return None, None  # not decided yet

    # ── TIER 1: original image only (fastest — catches ~80% of clear matches) ──
    result, score = _score_variant(img_bgr)
    if result is not None:
        return result, score

    # ── TIER 2: CLAHE enhanced + horizontal flip (catches bad lighting / mirrors) ──
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l_ch)
    enhanced = cv2.cvtColor(cv2.merge((l2, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
    for variant in [enhanced, cv2.flip(img_bgr, 1)]:
        result, score = _score_variant(variant)
        if result is not None:
            return result, score

    # ── TIER 3: rotations + sharpening (catches angled / blurry faces) ──
    # Only run if tiers 1-2 found SOME faces but no match,
    # or found no faces and we should try harder.
    tier3_variants = []
    blur_score = variance_of_laplacian(img_bgr)
    if blur_score < blur_tolerance:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        tier3_variants.append(cv2.filter2D(img_bgr, -1, kernel))

    angles = _parse_rotation_angles(rotation_angles)
    if angles:
        h, w = img_bgr.shape[:2]
        center_pt = (w // 2, h // 2)
        for angle in angles:
            M = cv2.getRotationMatrix2D(center_pt, angle, 1.0)
            rotated = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            tier3_variants.append(rotated)

    for variant in tier3_variants:
        result, score = _score_variant(variant)
        if result is not None:
            return result, score

    # ── TIER 4: tiling (catches tiny faces in large group photos) ──
    if enable_tiling and best_distance > relaxed_threshold:
        result, score = _score_variant(img_bgr, use_tiling=True)
        if result is not None:
            return result, score

    # ── Final partial-face check ──
    if (
        enable_partial_face_mode
        and best_partial_score >= min_partial_face_score
        and best_distance <= relaxed_threshold
        and best_center_distance <= (dynamic_center_limit + center_distance_margin)
        and relaxed_hits >= 1
    ):
        return True, 1.0 - best_partial_score

    if best_distance == 999.0:
        return False, 1.0
    return False, best_distance


def prepare_known_encodings(
    samples_dir: Path,
    max_side: int,
    blur_tolerance: int,
    sample_encoding_jitters: int,
    detector_model: str,
    face_upsample_times: int,
    enable_detector_fallback: bool,
    sample_outlier_percentile: float,
    min_sample_encodings: int,
    sample_detector_model: str = "hog",
    rotation_angles=None,
) -> List[np.ndarray]:
    encodings = []
    # HOG is fast; CNN fallback catches anything HOG misses.
    # Always enable fallback so CNN kicks in when HOG fails on sample images.
    effective_detector = sample_detector_model if sample_detector_model else detector_model
    for img_path in samples_dir.glob("**/*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower().replace(".", "") not in {
            "jpg",
            "jpeg",
            "png",
            "webp",
            "heic",
            "heif",
            "nef",
        }:
            continue

        img = load_image_cv(img_path)
        if img is None:
            continue

        img = resize_if_needed(img, max_side)

        # For sample images: use a light set of variants (original + CLAHE enhanced
        # + horizontal flip). Skip rotations — reference photos should be
        # clear and frontal; extra rotations just slow things down.
        sample_variants = [img]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l_ch)
        enhanced = cv2.cvtColor(cv2.merge((l2, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        sample_variants.append(enhanced)
        sample_variants.append(cv2.flip(img, 1))

        for variant in sample_variants:
            sample_encodings = extract_face_encodings(
                variant,
                detector_model=effective_detector,
                num_jitters=sample_encoding_jitters,
                upsample_times=face_upsample_times,
                enable_detector_fallback=True,  # always fall back to CNN for samples
                min_face_area_ratio=0.0,
                enable_tiling=False,
            )
            encodings.extend(sample_encodings)

    return prune_sample_outliers(encodings, sample_outlier_percentile, min_sample_encodings)


def sanitize_filename(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]", "_", value)
    return value.strip() or "unnamed"


def save_match_image(img_bgr: np.ndarray, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), img_bgr)


def process_single_candidate(
    service,
    file_meta: Dict,
    known_encodings: List[np.ndarray],
    pconfig: ProcessingConfig,
    matches_dir: Path,
    non_matches_dir: Path,
    save_non_matches: bool,
) -> Tuple[bool, str, float]:
    file_id = file_meta["id"]
    file_name = file_meta.get("name", file_id)

    try:
        img = download_drive_image(service, file_id)
        if img is None:
            return False, file_name, 1.0

        img = resize_if_needed(img, pconfig.max_image_side)

        rotation_angles = _parse_rotation_angles(pconfig.rotation_angles)

        matched, score = is_match(
            known_encodings,
            img,
            pconfig.face_distance_threshold,
            pconfig.relaxed_face_distance_threshold,
            pconfig.min_strict_hits,
            pconfig.min_relaxed_hits,
            pconfig.min_partial_face_score,
            pconfig.enable_partial_face_mode,
            pconfig.blur_tolerance,
            pconfig.candidate_encoding_jitters,
            pconfig.face_detector_model,
            pconfig.face_upsample_times,
            pconfig.enable_detector_fallback,
            pconfig.mean_topk_margin,
            pconfig.center_distance_margin,
            pconfig.strong_match_distance,
            pconfig.min_candidate_face_area_ratio,
            rotation_angles=rotation_angles,
            enable_tiling=pconfig.enable_tiling,
            tile_size=pconfig.tile_size,
            tile_overlap=pconfig.tile_overlap,
        )

        ext = Path(file_name).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".jpg"

        safe_name = sanitize_filename(Path(file_name).stem)
        out_name = f"{safe_name}__{file_id}{ext}"

        if matched:
            save_match_image(img, matches_dir / out_name)
        elif save_non_matches:
            save_match_image(img, non_matches_dir / out_name)

    except Exception:
        # Catch absolutely any error (MemoryError, segfault-adjacent, etc.)
        # so a single bad image never kills the entire run.
        return False, file_name, 1.0
    finally:
        # Explicitly free image memory — critical on long runs to prevent OOM
        try:
            del img
        except NameError:
            pass
        import gc; gc.collect()

    return matched, file_name, score


def main() -> None:
    root = Path(__file__).parent
    config_path = root / "config.json"

    if not config_path.exists():
        shutil.copy(root / "config.example.json", config_path)
        print("Created config.json from config.example.json. Please edit it, then rerun.")
        return

    config = load_config(config_path)

    credentials_json = str(root / config["google"]["credentials_json"])
    token_json = str(root / config["google"]["token_json"])

    links_file = root / config["input"]["drive_links_file"]
    sample_images_dir = root / config["input"]["sample_images_dir"]

    matches_dir = root / config["output"]["matches_dir"]
    non_matches_dir = root / config["output"]["non_matches_dir"]
    save_non_matches = bool(config["output"]["save_non_matches"])

    pconfig = ProcessingConfig(
        max_image_side=int(config["processing"]["max_image_side"]),
        face_distance_threshold=float(config["processing"]["face_distance_threshold"]),
        relaxed_face_distance_threshold=float(config["processing"].get("relaxed_face_distance_threshold", 0.58)),
        min_strict_hits=int(config["processing"].get("min_strict_hits", 2)),
        min_relaxed_hits=int(config["processing"].get("min_relaxed_hits", 2)),
        blur_tolerance=int(config["processing"]["blur_tolerance"]),
        process_extensions=list(config["processing"]["process_extensions"]),
        workers=int(config["processing"]["workers"]),
        min_partial_face_score=float(config["processing"]["min_partial_face_score"]),
        enable_partial_face_mode=bool(config["processing"]["enable_partial_face_mode"]),
        sample_encoding_jitters=int(config["processing"].get("sample_encoding_jitters", 2)),
        candidate_encoding_jitters=int(config["processing"].get("candidate_encoding_jitters", 1)),
        face_detector_model=str(config["processing"].get("face_detector_model", "hog")),
        face_upsample_times=int(config["processing"].get("face_upsample_times", 1)),
        enable_detector_fallback=bool(config["processing"].get("enable_detector_fallback", True)),
        mean_topk_margin=float(config["processing"].get("mean_topk_margin", 0.015)),
        center_distance_margin=float(config["processing"].get("center_distance_margin", 0.02)),
        strong_match_distance=float(config["processing"].get("strong_match_distance", 0.42)),
        min_candidate_face_area_ratio=float(config["processing"].get("min_candidate_face_area_ratio", 0.01)),
        sample_outlier_percentile=float(config["processing"].get("sample_outlier_percentile", 85.0)),
        min_sample_encodings=int(config["processing"].get("min_sample_encodings", 4)),
    )

    if not links_file.exists():
        print(f"Missing drive links file: {links_file}")
        return

    if not sample_images_dir.exists():
        print(f"Missing sample images directory: {sample_images_dir}")
        return

    ensure_dir(matches_dir)
    if save_non_matches:
        ensure_dir(non_matches_dir)

    print("Loading sample images and preparing face encodings...")
    known_encodings = prepare_known_encodings(
        sample_images_dir,
        pconfig.max_image_side,
        pconfig.blur_tolerance,
        pconfig.sample_encoding_jitters,
        pconfig.face_detector_model,
        pconfig.face_upsample_times,
        pconfig.enable_detector_fallback,
        pconfig.sample_outlier_percentile,
        pconfig.min_sample_encodings,
    )
    if not known_encodings:
        print("No faces found in samples. Add clearer sample images and rerun.")
        return

    print("Authenticating Google Drive...")
    service = get_drive_service(credentials_json, token_json)

    print("Reading links and discovering image files...")
    links = read_drive_links(links_file)
    candidates = gather_candidate_images(service, links, pconfig.process_extensions)

    print(f"Found {len(candidates)} candidate images.")
    if not candidates:
        return

    match_count = 0
    processed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=pconfig.workers) as executor:
        futures = [
            executor.submit(
                process_single_candidate,
                service,
                meta,
                known_encodings,
                pconfig,
                matches_dir,
                non_matches_dir,
                save_non_matches,
            )
            for meta in candidates
        ]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            processed += 1
            try:
                matched, _, _ = future.result()
                if matched:
                    match_count += 1
            except Exception:
                continue

    print(f"Done. Processed: {processed}, Matches: {match_count}")
    print(f"Matched photos saved in: {matches_dir}")


if __name__ == "__main__":
    main()
