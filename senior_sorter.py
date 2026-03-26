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
    blur_tolerance: int
    process_extensions: List[str]
    workers: int
    min_partial_face_score: float
    enable_partial_face_mode: bool
    sample_encoding_jitters: int
    candidate_encoding_jitters: int
    face_detector_model: str


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
            creds = flow.run_local_server(port=0)
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
                fields="id,name,mimeType,parents,shortcutDetails",
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
                    fields="nextPageToken, files(id,name,mimeType,shortcutDetails)",
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


def decode_image_bytes(raw: bytes) -> Optional[np.ndarray]:
    data = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is not None:
        return img_bgr

    # Fallback path for formats OpenCV may not decode (for example HEIC/HEIF).
    try:
        with Image.open(io.BytesIO(raw)) as pil_img:
            rgb = np.array(pil_img.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        pass

    # RAW fallback path (for example .NEF) if rawpy is available.
    if rawpy is not None:
        try:
            with rawpy.imread(io.BytesIO(raw)) as raw_img:
                rgb = raw_img.postprocess(use_camera_wb=True, no_auto_bright=True)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    return None


def download_drive_image(service, file_id: str) -> Optional[np.ndarray]:
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)

    done = False
    try:
        while not done:
            _, done = downloader.next_chunk()
    except HttpError:
        return None

    return decode_image_bytes(buffer.getvalue())


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


def preprocess_variants(img_bgr: np.ndarray, blur_tolerance: int) -> List[np.ndarray]:
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

    return variants


def extract_face_encodings(
    img_bgr: np.ndarray,
    detector_model: str = "hog",
    num_jitters: int = 1,
) -> List[np.ndarray]:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model=detector_model)
    if not locations:
        return []
    return face_recognition.face_encodings(rgb, known_face_locations=locations, num_jitters=num_jitters)


def compute_partial_face_score(known_encodings: List[np.ndarray], candidate_encoding: np.ndarray) -> float:
    distances = face_recognition.face_distance(known_encodings, candidate_encoding)
    if len(distances) == 0:
        return 0.0

    best = float(np.min(distances))
    # Convert distance to a confidence-like score in [0,1], where smaller distance is better.
    score = max(0.0, min(1.0, 1.0 - best))
    return score


def is_match(
    known_encodings: List[np.ndarray],
    img_bgr: np.ndarray,
    threshold: float,
    relaxed_threshold: float,
    min_relaxed_hits: int,
    min_partial_face_score: float,
    enable_partial_face_mode: bool,
    blur_tolerance: int,
    candidate_encoding_jitters: int,
    detector_model: str,
) -> Tuple[bool, float]:
    best_distance = 999.0
    best_partial_score = 0.0
    strict_hits = 0
    relaxed_hits = 0
    known_center = np.mean(np.vstack(known_encodings), axis=0)
    center_margin = 0.01

    for variant in preprocess_variants(img_bgr, blur_tolerance):
        candidate_encodings = extract_face_encodings(
            variant,
            detector_model=detector_model,
            num_jitters=candidate_encoding_jitters,
        )
        for candidate in candidate_encodings:
            distances = face_recognition.face_distance(known_encodings, candidate)
            if len(distances) == 0:
                continue

            d = float(np.min(distances))
            center_d = float(np.linalg.norm(candidate - known_center))
            if d < best_distance:
                best_distance = d

            if d <= threshold and center_d <= (threshold + center_margin):
                strict_hits += 1
                if strict_hits >= 1:
                    return True, min(d, center_d)

            if d <= relaxed_threshold and center_d <= (relaxed_threshold + center_margin):
                relaxed_hits += 1
                if relaxed_hits >= min_relaxed_hits and best_distance <= relaxed_threshold:
                    return True, min(d, center_d)

            if enable_partial_face_mode:
                partial_score = compute_partial_face_score(known_encodings, candidate)
                best_partial_score = max(best_partial_score, partial_score)

    if (
        enable_partial_face_mode
        and best_partial_score >= min_partial_face_score
        and best_distance <= relaxed_threshold
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
) -> List[np.ndarray]:
    encodings = []
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
        for variant in preprocess_variants(img, blur_tolerance):
            sample_encodings = extract_face_encodings(
                variant,
                detector_model=detector_model,
                num_jitters=sample_encoding_jitters,
            )
            encodings.extend(sample_encodings)

    return encodings


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

    img = download_drive_image(service, file_id)
    if img is None:
        return False, file_name, 1.0

    img = resize_if_needed(img, pconfig.max_image_side)

    matched, score = is_match(
        known_encodings,
        img,
        pconfig.face_distance_threshold,
        pconfig.relaxed_face_distance_threshold,
        pconfig.min_relaxed_hits,
        pconfig.min_partial_face_score,
        pconfig.enable_partial_face_mode,
        pconfig.blur_tolerance,
        pconfig.candidate_encoding_jitters,
        pconfig.face_detector_model,
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
        min_relaxed_hits=int(config["processing"].get("min_relaxed_hits", 2)),
        blur_tolerance=int(config["processing"]["blur_tolerance"]),
        process_extensions=list(config["processing"]["process_extensions"]),
        workers=int(config["processing"]["workers"]),
        min_partial_face_score=float(config["processing"]["min_partial_face_score"]),
        enable_partial_face_mode=bool(config["processing"]["enable_partial_face_mode"]),
        sample_encoding_jitters=int(config["processing"].get("sample_encoding_jitters", 2)),
        candidate_encoding_jitters=int(config["processing"].get("candidate_encoding_jitters", 1)),
        face_detector_model=str(config["processing"].get("face_detector_model", "hog")),
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
