import os
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # Allow HTTP on localhost for OAuth

import concurrent.futures
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from senior_sorter import (
    ProcessingConfig,
    ensure_dir,
    gather_candidate_images,
    prepare_known_encodings,
    process_single_candidate,
)

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]


def load_web_config(root: Path) -> Dict:
    config_path = root / "web_config.json"
    if not config_path.exists():
        sample = root / "web_config.example.json"
        if sample.exists():
            config_path.write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            config_path.write_text("{}", encoding="utf-8")
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_links(raw: str) -> List[str]:
    lines = []
    for line in raw.splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            lines.append(item)
    return lines


def get_token_cache_path(root: Path, config: Dict) -> Path:
    oauth_cfg = config.get("oauth", {})
    cache_name = oauth_cfg.get("token_cache_json", "token_web.json")
    return root / cache_name


def load_cached_session_if_any(root: Path, config: Dict) -> None:
    if st.session_state.get("creds_json"):
        return
    cache_path = get_token_cache_path(root, config)
    if cache_path.exists():
        try:
            raw = cache_path.read_text(encoding="utf-8")
            info = json.loads(raw)
            # If the cached token has no refresh_token it cannot be renewed;
            # check whether it has already expired and purge it so the user
            # is sent back to the login flow instead of hitting a later error.
            if not info.get("refresh_token"):
                from datetime import datetime, timezone
                expiry_str = info.get("expiry", "")
                if expiry_str:
                    try:
                        expiry_dt = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
                        if datetime.now(timezone.utc) >= expiry_dt:
                            cache_path.unlink(missing_ok=True)
                            return
                    except Exception:
                        pass
            st.session_state["creds_json"] = raw
        except Exception:
            pass


def persist_session(root: Path, config: Dict, creds_json: str) -> None:
    cache_path = get_token_cache_path(root, config)
    cache_path.write_text(creds_json, encoding="utf-8")


def is_allowed_college_email(email: str, domains: List[str]) -> bool:
    email = (email or "").strip().lower()
    if "@" not in email:
        return False
    if not domains:
        return True
    return any(email.endswith("@" + d.lower()) for d in domains)


def build_processing_config(config: Dict) -> ProcessingConfig:
    p = config.get("processing", {})
    return ProcessingConfig(
        max_image_side=int(p.get("max_image_side", 1600)),
        face_distance_threshold=float(p.get("face_distance_threshold", 0.50)),
        relaxed_face_distance_threshold=float(p.get("relaxed_face_distance_threshold", 0.58)),
        min_strict_hits=int(p.get("min_strict_hits", 1)),
        min_relaxed_hits=int(p.get("min_relaxed_hits", 2)),
        blur_tolerance=int(p.get("blur_tolerance", 50)),
        process_extensions=list(p.get("process_extensions", ["jpg", "jpeg", "png", "webp", "heic", "heif", "nef"])),
        workers=int(p.get("workers", 2)),
        min_partial_face_score=float(p.get("min_partial_face_score", 0.40)),
        enable_partial_face_mode=bool(p.get("enable_partial_face_mode", True)),
        sample_encoding_jitters=int(p.get("sample_encoding_jitters", 3)),
        candidate_encoding_jitters=int(p.get("candidate_encoding_jitters", 1)),
        face_detector_model=str(p.get("face_detector_model", "hog")),
        face_upsample_times=int(p.get("face_upsample_times", 2)),
        enable_detector_fallback=bool(p.get("enable_detector_fallback", True)),
        mean_topk_margin=float(p.get("mean_topk_margin", 0.025)),
        center_distance_margin=float(p.get("center_distance_margin", 0.03)),
        strong_match_distance=float(p.get("strong_match_distance", 0.45)),
        min_candidate_face_area_ratio=float(p.get("min_candidate_face_area_ratio", 0.005)),
        sample_outlier_percentile=float(p.get("sample_outlier_percentile", 90.0)),
        min_sample_encodings=int(p.get("min_sample_encodings", 2)),
        sample_detector_model=str(p.get("sample_detector_model", "cnn")),
        enable_tiling=bool(p.get("enable_tiling", True)),
        tile_size=int(p.get("tile_size", 800)),
        tile_overlap=float(p.get("tile_overlap", 0.25)),
        rotation_angles=str(p.get("rotation_angles", "-15,15,-30,30")),
    )


def get_user_info(creds: Credentials) -> Dict:
    oauth2 = build("oauth2", "v2", credentials=creds)
    return oauth2.userinfo().get().execute()


def has_oauth_client_config(root: Path, config: Dict) -> bool:
    credentials_json = root / config.get("oauth", {}).get("credentials_json", "client_secret.json")
    if credentials_json.exists():
        return True
    return "GOOGLE_CLIENT_SECRET" in os.environ


def get_oauth_client_config(root: Path, config: Dict) -> Dict:
    credentials_json = root / config.get("oauth", {}).get("credentials_json", "client_secret.json")
    if credentials_json.exists():
        return json.loads(credentials_json.read_text(encoding="utf-8"))
    if "GOOGLE_CLIENT_SECRET" in os.environ:
        return json.loads(os.environ["GOOGLE_CLIENT_SECRET"])
    raise FileNotFoundError("OAuth client JSON not found in file or GOOGLE_CLIENT_SECRET env var.")


def get_oauth_redirect_uri(config: Dict) -> str:
    if "SPACE_HOST" in os.environ:
        return f"https://{os.environ['SPACE_HOST']}/"
    oauth_cfg = config.get("oauth", {})
    return oauth_cfg.get("redirect_uri") or oauth_cfg.get("local_redirect_uri", "http://localhost:8501/")


def _qp_get(name: str) -> str:
    value = st.query_params.get(name)
    if value is None:
        return ""
    if isinstance(value, list):
        return value[0] if value else ""
    return str(value)


def login_with_google(root: Path, config: Dict) -> None:
    try:
        client_config = get_oauth_client_config(root, config)
    except FileNotFoundError:
        st.error("Login not configured. Please contact admin.")
        return
    except json.JSONDecodeError:
        st.error("Login configuration is invalid. Please contact admin.")
        return

    credentials_json = root / config.get("oauth", {}).get("credentials_json", "client_secret.json")
    redirect_uri = get_oauth_redirect_uri(config)

    client_section = client_config.get("web") or client_config.get("installed")
    if not client_section:
        st.error("OAuth JSON must contain either 'web' or 'installed' configuration.")
        return

    redirect_uris = client_section.get("redirect_uris", [])
    if redirect_uris and redirect_uri not in redirect_uris:
        st.error(
            "Google OAuth redirect URI mismatch. Add this exact URI in Google Cloud Console: "
            + redirect_uri
        )
        return

    domains = [d.lower() for d in config.get("college_domains", [])]
    primary_domain = domains[0] if domains else None

    try:
        client_cfg = get_oauth_client_config(root, config)
        flow = Flow.from_client_config(
            client_cfg,
            scopes=SCOPES,
            redirect_uri=redirect_uri,
        )
        auth_url, state = flow.authorization_url(
            access_type="offline",
            prompt="select_account",
        )
        st.session_state["oauth_state"] = state
        st.session_state["oauth_redirect_uri"] = redirect_uri
        st.session_state["oauth_login_started"] = True
        st.link_button("Continue Google Sign-In", auth_url)
        st.info("Complete Google sign-in in the opened page. You will be redirected back to this app.")
    except Exception as ex:
        msg = str(ex)
        msg_l = msg.lower()
        if "redirect_uri_mismatch" in msg_l:
            st.error(
                "Google redirect URI mismatch. In Google Cloud Console, add this exact Authorized redirect URI: "
                + redirect_uri
            )
            with st.expander("OAuth error details"):
                st.code(msg)
            return
        st.error("Google login failed. Check OAuth consent screen setup and redirect URI configuration.")
        with st.expander("OAuth error details"):
            st.code(msg)


def complete_google_login_from_callback(root: Path, config: Dict) -> None:
    code = _qp_get("code")
    state = _qp_get("state")
    oauth_error = _qp_get("error")

    if not code and not oauth_error:
        return

    if oauth_error:
        st.error(f"Google login failed: {oauth_error}")
        st.query_params.clear()
        return

    expected_state = st.session_state.get("oauth_state", "")
    if expected_state and state and state != expected_state:
        st.error("OAuth state validation failed. Please try login again.")
        st.query_params.clear()
        return

    credentials_json = root / config.get("oauth", {}).get("credentials_json", "client_secret.json")
    redirect_uri = st.session_state.get("oauth_redirect_uri") or get_oauth_redirect_uri(config)

    try:
        client_cfg = get_oauth_client_config(root, config)
        flow = Flow.from_client_config(
            client_cfg,
            scopes=SCOPES,
            redirect_uri=redirect_uri,
        )
        if state:
            flow.oauth2session.state = state

        authorization_response = f"{redirect_uri}?code={code}"
        if state:
            authorization_response += f"&state={state}"

        flow.fetch_token(authorization_response=authorization_response)
        creds = flow.credentials
    except Exception as ex:
        st.error("Failed to complete Google login from callback.")
        with st.expander("OAuth error details"):
            st.code(str(ex))
        st.query_params.clear()
        return

    domains = [d.lower() for d in config.get("college_domains", [])]
    user = get_user_info(creds)
    email = user.get("email", "").lower()
    if not is_allowed_college_email(email, domains):
        st.error("Signed-in account is not in allowed college domains.")
        st.query_params.clear()
        return

    st.session_state["creds_json"] = creds.to_json()
    st.session_state["user_email"] = email
    st.session_state.pop("oauth_state", None)
    st.session_state.pop("oauth_redirect_uri", None)
    st.session_state.pop("oauth_login_started", None)
    persist_session(root, config, st.session_state["creds_json"])
    st.query_params.clear()
    st.rerun()


def get_drive_service_from_session():
    creds_json = st.session_state.get("creds_json")
    if not creds_json:
        return None

    info = json.loads(creds_json)
    if info.get("refresh_token"):
        creds = Credentials.from_authorized_user_info(info, SCOPES)
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                st.session_state["creds_json"] = creds.to_json()
            except Exception:
                # Refresh failed – clear session so user can re-login cleanly.
                st.session_state.pop("creds_json", None)
                st.session_state.pop("user_email", None)
                st.error("Session expired. Please login again.")
                return None
    else:
        from datetime import datetime, timezone
        expiry_str = info.get("expiry", "")
        token_expired = False
        if expiry_str:
            try:
                expiry_dt = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
                token_expired = datetime.now(timezone.utc) >= expiry_dt
            except Exception:
                pass
        if not info.get("token") or token_expired:
            st.session_state.pop("creds_json", None)
            st.session_state.pop("user_email", None)
            st.error("Session expired. Please login again with your college Google account.")
            return None
        creds = Credentials(
            token=info.get("token"),
            token_uri=info.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=info.get("client_id"),
            client_secret=info.get("client_secret"),
            scopes=SCOPES,
        )

    return build("drive", "v3", credentials=creds)


def save_uploaded_samples(files, target_dir: Path) -> int:
    ensure_dir(target_dir)
    count = 0
    for uploaded in files:
        name = Path(uploaded.name).name
        out = target_dir / name
        out.write_bytes(uploaded.getbuffer())
        count += 1
    return count


def zip_matches(matches_dir: Path, doubtful_dir: Path, zip_path: Path) -> Tuple[int, int]:
    matched_files = [p for p in matches_dir.glob("**/*") if p.is_file()]
    doubtful_files = [p for p in doubtful_dir.glob("**/*") if p.is_file()]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in matched_files:
            zf.write(p, arcname=f"extracted/{p.name}")
        for p in doubtful_files:
            zf.write(p, arcname=f"doubtful/{p.name}")
    return len(matched_files), len(doubtful_files)


def run_extraction(root: Path, config: Dict, links_text: str, uploaded_samples) -> Dict:
    service = get_drive_service_from_session()
    if service is None:
        return {"ok": False, "error": "Please login again with college Google account. Previous session token expired."}

    links = parse_links(links_text)
    if not links:
        return {"ok": False, "error": "Add at least one Google Drive link."}

    if not uploaded_samples:
        return {"ok": False, "error": "Upload at least one sample image of the senior."}

    job_root = root / "output" / "web_jobs"
    ensure_dir(job_root)

    job_id = st.session_state.get("user_email", "user").replace("@", "_").replace(".", "_")
    work_dir = job_root / job_id
    samples_dir = work_dir / "samples"
    matches_dir = work_dir / "matches"
    doubtful_dir = work_dir / "doubtful"
    non_matches_dir = work_dir / "non_matches"

    ensure_dir(samples_dir)
    ensure_dir(matches_dir)
    ensure_dir(doubtful_dir)
    ensure_dir(non_matches_dir)

    def _safe_unlink(p: Path):
        try:
            p.unlink()
        except (PermissionError, OSError):
            pass  # file still locked by previous thread on Windows

    for old in samples_dir.glob("*"):
        if old.is_file():
            _safe_unlink(old)
    for old in matches_dir.glob("*"):
        if old.is_file():
            _safe_unlink(old)
    for old in doubtful_dir.glob("*"):
        if old.is_file():
            _safe_unlink(old)

    save_uploaded_samples(uploaded_samples, samples_dir)

    pconfig = build_processing_config(config)
    save_non_matches = bool(config.get("processing", {}).get("save_non_matches", False))

    with st.spinner("Preparing face embeddings from samples..."):
        known_encodings = prepare_known_encodings(
            samples_dir,
            pconfig.max_image_side,
            pconfig.blur_tolerance,
            pconfig.sample_encoding_jitters,
            pconfig.face_detector_model,
            pconfig.face_upsample_times,
            pconfig.enable_detector_fallback,
            pconfig.sample_outlier_percentile,
            pconfig.min_sample_encodings,
            sample_detector_model=pconfig.sample_detector_model,
            rotation_angles=pconfig.rotation_angles,
        )

    if not known_encodings:
        return {"ok": False, "error": "No detectable faces found in sample images."}

    with st.spinner("Discovering candidate images from Drive links..."):
        candidates = gather_candidate_images(service, links, pconfig.process_extensions)

    link_stats: Dict[str, Dict[str, int]] = {link: {"candidates": 0, "processed": 0, "matched": 0} for link in links}
    for meta in candidates:
        src = meta.get("source_link", "unknown")
        if src not in link_stats:
            link_stats[src] = {"candidates": 0, "processed": 0, "matched": 0}
        link_stats[src]["candidates"] += 1

    total = len(candidates)
    if total == 0:
        return {"ok": False, "error": "No candidate images found from given links."}

    progress = st.progress(0.0)
    status = st.empty()
    detail = st.empty()
    matched = 0
    doubtful_count = 0
    processed_count = 0
    workers = max(1, pconfig.workers)
    per_image_timeout = 10  # seconds — save to doubtful if takes longer

    import time as _time
    from senior_sorter import download_drive_image, resize_if_needed, save_match_image, sanitize_filename

    def _process_one(meta):
        try:
            return process_single_candidate(
                service, meta, known_encodings, pconfig,
                matches_dir, non_matches_dir, save_non_matches,
            )
        except Exception:
            return False, meta.get("name", "?"), 1.0

    def _save_doubtful(meta):
        """Download image and save to doubtful folder."""
        try:
            file_id = meta["id"]
            file_name = meta.get("name", file_id)
            img = download_drive_image(service, file_id)
            if img is not None:
                img = resize_if_needed(img, pconfig.max_image_side)
                ext = Path(file_name).suffix.lower()
                if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                    ext = ".jpg"
                safe_name = sanitize_filename(Path(file_name).stem)
                out_name = f"{safe_name}__{file_id}{ext}"
                save_match_image(img, doubtful_dir / out_name)
        except Exception:
            pass

    if workers <= 1:
        for meta in candidates:
            processed_count += 1
            file_name = meta.get("name", meta.get("id", "unknown"))
            src_link = meta.get("webViewLink", meta.get("source_link", "unknown"))
            status.markdown(f"**Processing {processed_count}/{total}:** `{file_name}`  \n🔗 **Source File:** [{src_link}]({src_link})")
            t0 = _time.time()

            ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = ex.submit(_process_one, meta)
            try:
                ok, _, _ = future.result(timeout=per_image_timeout)
                ex.shutdown(wait=True)
            except concurrent.futures.TimeoutError:
                ok = False
                doubtful_count += 1
                detail.warning(f"⏭️ Doubtful (slow): {file_name} → saved for review")
                _save_doubtful(meta)
                try:
                    ex.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    ex.shutdown(wait=False)  # Fallback for Python < 3.9
            except Exception:
                ok = False
                try:
                    ex.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    ex.shutdown(wait=False)

            elapsed = _time.time() - t0
            src = meta.get("source_link", "unknown")
            link_stats[src]["processed"] += 1
            if ok:
                matched += 1
                link_stats[src]["matched"] += 1
            progress.progress(processed_count / total)
            status.markdown(
                f"**Processed {processed_count}/{total}** ({elapsed:.1f}s) | "
                f"**Matched {matched}**" + (f" | **Doubtful {doubtful_count}**" if doubtful_count else "")
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_meta = {
                executor.submit(_process_one, meta): meta
                for meta in candidates
            }
            for future in concurrent.futures.as_completed(future_to_meta):
                meta = future_to_meta[future]
                processed_count += 1
                file_name = meta.get("name", meta.get("id", "unknown"))
                try:
                    ok, _, _ = future.result(timeout=per_image_timeout)
                except concurrent.futures.TimeoutError:
                    ok = False
                    doubtful_count += 1
                    _save_doubtful(meta)
                except Exception:
                    ok = False
                src = meta.get("source_link", "unknown")
                link_stats[src]["processed"] += 1
                if ok:
                    matched += 1
                    link_stats[src]["matched"] += 1
                progress.progress(processed_count / total)
                status.text(
                    f"Processed {processed_count}/{total}: {file_name} | "
                    f"Matched {matched}" + (f" | Doubtful {doubtful_count}" if doubtful_count else "")
                )

    zip_path = work_dir / "senior_matches.zip"
    count_in_zip, doubtful_in_zip = zip_matches(matches_dir, doubtful_dir, zip_path)

    return {
        "ok": True,
        "processed": total,
        "matched": matched,
        "doubtful": doubtful_count,
        "zip_path": str(zip_path),
        "zip_count": count_in_zip,
        "zip_doubtful": doubtful_in_zip,
        "per_link_stats": [
            {
                "link": link,
                "candidates": stats["candidates"],
                "processed": stats["processed"],
                "matched": stats["matched"],
            }
            for link, stats in link_stats.items()
        ],
    }


def render_header() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@500&display=swap');
        html, body, [class*="css"]  { font-family: 'Space Grotesk', sans-serif; }
        .hero {
            background: linear-gradient(140deg, #12213d 0%, #0f6c5a 48%, #f2a154 100%);
            padding: 1.2rem 1.4rem;
            border-radius: 14px;
            color: #f8f9fa;
            border: 1px solid rgba(255,255,255,0.28);
            margin-bottom: 0.8rem;
        }
        .hero h1 { margin: 0; font-size: 1.6rem; letter-spacing: 0.2px; }
        .hero p { margin: 0.45rem 0 0; opacity: 0.95; }
        .mono { font-family: 'IBM Plex Mono', monospace; }
        </style>
        <div class='hero'>
            <h1>Senior Photo Sorter</h1>
            <p>Login with college Google ID, add Drive links, upload reference images, and export matched photos as ZIP.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    root = Path(__file__).parent
    config = load_web_config(root)

    # MUST be the very first Streamlit command in the script.
    st.set_page_config(page_title="Senior Photo Sorter", page_icon="📸", layout="wide")

    load_cached_session_if_any(root, config)
    complete_google_login_from_callback(root, config)

    if "is_processing" not in st.session_state:
        st.session_state["is_processing"] = False

    render_header()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Login")
        allowed = config.get("college_domains", [])
        if allowed:
            st.caption("Allowed domains: " + ", ".join(allowed))
        st.caption("You must login with your PEC Google ID before extraction.")

        creds_ready = has_oauth_client_config(root, config)
        if not creds_ready:
            st.warning("Login temporarily unavailable. App admin must configure Google OAuth first.")

        _redirect_uri_display = get_oauth_redirect_uri(config)
        st.caption(f"🔗 Redirect URI in use: `{_redirect_uri_display}`")
        if st.button("Login with College Google", disabled=not creds_ready):
            login_with_google(root, config)

        email = st.session_state.get("user_email")
        if email:
            st.success(f"Logged in as: {email}")
            if st.button("Logout"):
                st.session_state.pop("creds_json", None)
                st.session_state.pop("user_email", None)
                cache_path = get_token_cache_path(root, config)
                if cache_path.exists():
                    cache_path.unlink()
                st.rerun()

    with col2:
        st.subheader("Input")
        logged_in = bool(st.session_state.get("user_email"))
        if not logged_in:
            st.warning("Login required: sign in with PEC Google ID to enable inputs and extraction.")

        links_text = st.text_area(
            "Google Drive links (one per line)",
            height=180,
            placeholder="https://drive.google.com/drive/folders/...\nhttps://drive.google.com/file/d/.../view",
            disabled=not logged_in,
            key="links_text",
        )
        uploaded_samples = st.file_uploader(
            "Reference photos of the senior",
            type=["jpg", "jpeg", "png", "webp", "heic", "heif", "nef"],
            accept_multiple_files=True,
            disabled=not logged_in,
            key="uploaded_samples",
        )

        run = st.button(
            "Extract Senior Photos",
            type="primary",
            disabled=not logged_in or st.session_state.get("is_processing", False),
        )
        if run:
            st.session_state["is_processing"] = True
            try:
                result = run_extraction(root, config, links_text, uploaded_samples)
            finally:
                st.session_state["is_processing"] = False

            if not result.get("ok"):
                st.error(result.get("error", "Extraction failed."))
            else:
                msg = f"Done: Processed {result['processed']} images, matched {result['matched']}."
                if result.get("doubtful"):
                    msg += f" ({result['doubtful']} doubtful/slow)."
                msg += f" ZIP contains {result['zip_count']} matched and {result.get('zip_doubtful', 0)} doubtful files."
                st.success(msg)

                if result.get("per_link_stats"):
                    st.subheader("Per-Link Stats")
                    st.dataframe(result["per_link_stats"], use_container_width=True)
                
                zip_path = Path(result["zip_path"])
                if zip_path.exists():
                    st.download_button(
                        "Download ZIP (Final)",
                        data=zip_path.read_bytes(),
                        file_name="senior_matches.zip",
                        mime="application/zip",
                    )

        st.divider()
        st.subheader("Available Downloads")
        st.markdown("If the process was stopped early or got stuck, you can download whatever has been matched so far here:")
        
        job_id = st.session_state.get("user_email", "user").replace("@", "_").replace(".", "_")
        work_dir = root / "output" / "web_jobs" / job_id
        matches_dir = work_dir / "matches"
        doubtful_dir = work_dir / "doubtful"
        partial_zip_path = work_dir / "senior_matches_partial.zip"

        if work_dir.exists():
            mc = len(list(matches_dir.glob("**/*"))) if matches_dir.exists() else 0
            dc = len(list(doubtful_dir.glob("**/*"))) if doubtful_dir.exists() else 0
            
            if mc > 0 or dc > 0:
                st.info(f"Currently found on disk: **{mc} matched**, **{dc} doubtful**.")
                # We do the zipping outside of a button so the download_button can get the data without a rerun issue
                save_mc, save_dc = zip_matches(matches_dir, doubtful_dir, partial_zip_path)
                if partial_zip_path.exists():
                    st.download_button(
                        "💾 Click to Download Partial ZIP of everything found so far",
                        data=partial_zip_path.read_bytes(),
                        file_name="senior_matches_partial.zip",
                        mime="application/zip",
                        key="partial_download_button",
                    )
            else:
                st.write("No matched files found on disk yet.")

            # --- INTERACTIVE REVIEW UI ---
            if dc > 0:
                st.divider()
                st.subheader("👀 Review Doubtful Images")
                st.markdown("These images timed out cleanly. Manually decide what to do with them below:")
                
                doubtful_files = list(doubtful_dir.glob("*.*"))
                if not doubtful_files:
                    st.success("All doubtful images resolved!")
                else:
                    cols = st.columns(3)
                    for idx, file_path in enumerate(doubtful_files):
                        with cols[idx % 3]:
                            st.image(str(file_path), use_container_width=True)
                            st.caption(f"**{file_path.name}**")
                            if st.button("✅ Match", key=f"match_{idx}_{file_path.name}", use_container_width=True):
                                dest = matches_dir / file_path.name
                                if not dest.exists():
                                    file_path.rename(dest)
                                else:
                                    file_path.unlink()  # duplicate
                                st.rerun()
                            if st.button("❌ Trash", key=f"trash_{idx}_{file_path.name}", use_container_width=True):
                                file_path.unlink()
                                st.rerun()

if __name__ == "__main__":
    main()
