# Senior Photo Extractor from Google Drive Links

This project has two modes:

- Streamlit web UI for login, links, uploads, and ZIP download.
- CLI script for direct local processing.

## What it does

- Accepts direct Drive file links and folder links.
- Recursively scans nested folders.
- Downloads images and runs face matching using your sample photos.
- Handles blurry images better via preprocessing.
- Supports partial-face mode for difficult cases (limited by face visibility).
- Lets users login with college Google ID and enforces allowed email domains.
- Exports matched results as a ZIP file from the web app.

## 1) Prerequisites (Windows)

- Python 3.10+ recommended.
- A Google Cloud OAuth client for Drive API (`client_secret.json`).
- Access permissions to all listed Drive links.

## 2) Setup

```powershell
cd d:\Sorter
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If `face-recognition` fails to install on Windows, install Visual C++ Build Tools and CMake, then retry.

## 3) Google Drive API setup

1. Go to Google Cloud Console.
2. Create/select a project.
3. Enable **Google Drive API**.
4. Create OAuth client credentials (Desktop app).
5. Download JSON and place it as `client_secret.json` in `d:\Sorter`.

If you create a **Web application** OAuth client instead, add this redirect URI in Google Cloud Console:

- `http://localhost:53682/`

And keep `local_redirect_uri` in `web_config.json` aligned with that exact value.

If you see `[WinError 10048]` during login, another process is using the callback port.

1. Find process on the port:
  - `netstat -ano | findstr :53682`
2. Kill that process:
  - `taskkill /PID <PID_FROM_NETSTAT> /F`
3. Retry login.

Alternative:

1. Change `oauth.local_redirect_uri` in `web_config.json` to a different localhost port.
2. Add that exact URI in Google OAuth Authorized redirect URIs.

## 4) Prepare input

1. Copy `config.example.json` to `config.json` if you want CLI mode.
2. Copy `web_config.example.json` to `web_config.json` for web mode.
3. Update `college_domains` in `web_config.json` (example: `pec.edu`, `pec.edu.in`).
4. Put your Google OAuth desktop client file as `client_secret.json` in `d:\Sorter` (one-time admin setup).
5. In web mode, links and samples are entered directly from the UI.
6. In CLI mode, use `samples/` and `drive_links.txt`.

Supported link styles include:

- `https://drive.google.com/drive/folders/<ID>`
- `https://drive.google.com/file/d/<ID>/view`

## 5) Run Web UI (Recommended)

```powershell
cd d:\Sorter
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Then:

1. Click login and sign in with your college Google ID.
2. Paste all Drive links (one per line).
3. Upload 1-5 sample images of one senior.
4. Click Extract and download the generated ZIP.

## 6) Run CLI (Optional)

```powershell
cd d:\Sorter
.\.venv\Scripts\Activate.ps1
python senior_sorter.py
```

On first run, browser-based OAuth will open and create `token.json`.

## 7) Output

- Matched photos: `output/senior_matches`
- Optional non-matches (if enabled): `output/non_matches`
- Web job output: `output/web_jobs/<user>/senior_matches.zip`


## Accuracy notes

- Very occluded faces (for example only 20% visible) are inherently hard for face embeddings.
- Appearance shifts like beard/bald/trimmed beard and 3-4 year age difference are supported using adaptive matching settings.
- Improve recall by:
  - adding more varied sample images (angles, lighting, expressions),
  - increasing `face_distance_threshold` slightly (for example from `0.5` to `0.55`),
  - keeping `enable_partial_face_mode` true.
- Higher recall may increase false positives, so do a quick manual review of final matches.

Recommended tuning for appearance changes:

- `face_distance_threshold`: `0.5` (strict pass)
- `relaxed_face_distance_threshold`: `0.58` (fallback pass)
- `min_relaxed_hits`: `2` (requires repeated relaxed evidence)
- `sample_encoding_jitters`: `2` for stronger sample embeddings
- `candidate_encoding_jitters`: `1` to keep processing speed reasonable
