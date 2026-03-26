import json
import zipfile
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import streamlit as st
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
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
            st.session_state["creds_json"] = cache_path.read_text(encoding="utf-8")
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
        face_distance_threshold=float(p.get("face_distance_threshold", 0.47)),
        relaxed_face_distance_threshold=float(p.get("relaxed_face_distance_threshold", 0.52)),
        min_relaxed_hits=int(p.get("min_relaxed_hits", 3)),
        blur_tolerance=int(p.get("blur_tolerance", 60)),
        process_extensions=list(p.get("process_extensions", ["jpg", "jpeg", "png", "webp", "heic", "heif", "nef"])),
        workers=int(p.get("workers", 1)),
        min_partial_face_score=float(p.get("min_partial_face_score", 0.45)),
        enable_partial_face_mode=bool(p.get("enable_partial_face_mode", False)),
        sample_encoding_jitters=int(p.get("sample_encoding_jitters", 2)),
        candidate_encoding_jitters=int(p.get("candidate_encoding_jitters", 1)),
        face_detector_model=str(p.get("face_detector_model", "hog")),
    )


def get_user_info(creds: Credentials) -> Dict:
    oauth2 = build("oauth2", "v2", credentials=creds)
    return oauth2.userinfo().get().execute()


def has_oauth_client_config(root: Path, config: Dict) -> bool:
    credentials_json = root / config.get("oauth", {}).get("credentials_json", "client_secret.json")
    return credentials_json.exists()


def get_oauth_client_config(root: Path, config: Dict) -> Dict:
    credentials_json = root / config.get("oauth", {}).get("credentials_json", "client_secret.json")
    if credentials_json.exists():
        return json.loads(credentials_json.read_text(encoding="utf-8"))

    raise FileNotFoundError("OAuth client JSON not found.")


def login_with_google(root: Path, config: Dict) -> None:
    try:
        client_config = get_oauth_client_config(root, config)
    except FileNotFoundError:
        st.error("Login not configured. Please contact admin.")
        return
    except json.JSONDecodeError:
        st.error("Login configuration is invalid. Please contact admin.")
        return

    oauth_cfg = config.get("oauth", {})
    local_redirect_uri = oauth_cfg.get("local_redirect_uri", "http://localhost:53682/")

    client_section = client_config.get("web") or client_config.get("installed")
    if not client_section:
        st.error("OAuth JSON must contain either 'web' or 'installed' configuration.")
        return

    redirect_uris = client_section.get("redirect_uris", [])
    if redirect_uris and local_redirect_uri not in redirect_uris:
        st.error(
            "Google OAuth redirect URI mismatch. Add this exact URI in Google Cloud Console: "
            + local_redirect_uri
        )
        return

    parsed = urlparse(local_redirect_uri)
    port = parsed.port or 53682

    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    flow.redirect_uri = local_redirect_uri
    domains = [d.lower() for d in config.get("college_domains", [])]
    primary_domain = domains[0] if domains else None
    try:
        creds = flow.run_local_server(
            port=port,
            authorization_prompt_message="Browser login started. Complete Google sign-in and return to Streamlit.",
            success_message="Login complete. Return to the app window.",
            prompt="select_account",
            hd=primary_domain,
        )
    except Exception as ex:
        msg = str(ex)
        msg_l = msg.lower()
        if "winerror 10048" in msg_l or "only one usage of each socket address" in msg_l:
            st.error(
                f"Login callback port is busy ({port}). Close any previous login attempt and retry."
            )
            st.info(
                "If it keeps happening, change oauth.local_redirect_uri in web_config.json to another localhost port "
                "and add that exact URI to Google OAuth Authorized redirect URIs."
            )
            with st.expander("Port troubleshooting"):
                st.code(
                    "netstat -ano | findstr :" + str(port) + "\n"
                    "taskkill /PID <PID_FROM_NETSTAT> /F"
                )
            with st.expander("OAuth error details"):
                st.code(msg)
            return
        if "redirect_uri_mismatch" in msg_l:
            st.error(
                "Google redirect URI mismatch. In Google Cloud Console, add this exact Authorized redirect URI: "
                + local_redirect_uri
            )
            with st.expander("OAuth error details"):
                st.code(msg)
            return
        if "access_denied" in msg_l or "verification process" in msg_l or "error 403" in msg_l:
            st.error(
                "Google blocked access because the OAuth app is in testing mode. "
                "Add this PEC account as a Test User in Google Cloud Console OAuth consent screen."
            )
            with st.expander("OAuth error details"):
                st.code(msg)
            return
        if "unauthorized_client" in msg_l:
            st.error(
                "OAuth client is not authorized for this flow. Check OAuth client type and consent screen configuration."
            )
            with st.expander("OAuth error details"):
                st.code(msg)
            return
        st.error("Google login failed. Check OAuth consent screen setup and redirect URI configuration.")
        with st.expander("OAuth error details"):
            st.code(msg)
        return

    user = get_user_info(creds)
    email = user.get("email", "").lower()

    if not is_allowed_college_email(email, domains):
        st.error("Signed-in account is not in allowed college domains.")
        return

    st.session_state["creds_json"] = creds.to_json()
    st.session_state["user_email"] = email
    persist_session(root, config, st.session_state["creds_json"])
    st.rerun()


def get_drive_service_from_session():
    creds_json = st.session_state.get("creds_json")
    if not creds_json:
        return None

    info = json.loads(creds_json)
    # Some OAuth responses (especially in testing flows) may not return refresh_token.
    # In that case, build a non-refreshable credential from access token and ask user to relogin after expiry.
    if info.get("refresh_token"):
        creds = Credentials.from_authorized_user_info(info, SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            st.session_state["creds_json"] = creds.to_json()
    else:
        creds = Credentials(
            token=info.get("token"),
            token_uri=info.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=info.get("client_id"),
            client_secret=info.get("client_secret"),
            scopes=SCOPES,
        )
        if not creds.token:
            st.error("Session token is invalid or expired. Please login again.")
            return None

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


def zip_matches(matches_dir: Path, zip_path: Path) -> int:
    files = [p for p in matches_dir.glob("**/*") if p.is_file()]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, arcname=p.name)
    return len(files)


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
    non_matches_dir = work_dir / "non_matches"

    ensure_dir(samples_dir)
    ensure_dir(matches_dir)
    ensure_dir(non_matches_dir)

    for old in samples_dir.glob("*"):
        if old.is_file():
            old.unlink()
    for old in matches_dir.glob("*"):
        if old.is_file():
            old.unlink()

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
    matched = 0

    for idx, meta in enumerate(candidates, start=1):
        ok, _, _ = process_single_candidate(
            service,
            meta,
            known_encodings,
            pconfig,
            matches_dir,
            non_matches_dir,
            save_non_matches,
        )
        src = meta.get("source_link", "unknown")
        link_stats[src]["processed"] += 1
        if ok:
            matched += 1
            link_stats[src]["matched"] += 1
        progress.progress(idx / total)
        status.text(f"Processed {idx}/{total} | Matched {matched}")

    zip_path = work_dir / "senior_matches.zip"
    count_in_zip = zip_matches(matches_dir, zip_path)

    return {
        "ok": True,
        "processed": total,
        "matched": matched,
        "zip_path": str(zip_path),
        "zip_count": count_in_zip,
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
    load_cached_session_if_any(root, config)

    if "is_processing" not in st.session_state:
        st.session_state["is_processing"] = False

    st.set_page_config(page_title="Senior Photo Sorter", page_icon="📸", layout="wide")
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
                st.success(
                    f"Done: Processed {result['processed']} images, matched {result['matched']} images, zip contains {result['zip_count']} files."
                )
                if result.get("per_link_stats"):
                    st.subheader("Per-Link Stats")
                    st.dataframe(result["per_link_stats"], use_container_width=True)
                zip_path = Path(result["zip_path"])
                if zip_path.exists():
                    st.download_button(
                        "Download ZIP",
                        data=zip_path.read_bytes(),
                        file_name="senior_matches.zip",
                        mime="application/zip",
                    )


if __name__ == "__main__":
    main()
