#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/senior-sorter"
cd "$APP_DIR"

docker build -t senior-sorter:latest .

docker rm -f senior-sorter >/dev/null 2>&1 || true

docker run -d \
  --name senior-sorter \
  --restart unless-stopped \
  -p 8501:8501 \
  -v "$APP_DIR/client_secret.json:/app/client_secret.json:ro" \
  -v "$APP_DIR/web_config.json:/app/web_config.json:ro" \
  -v "$APP_DIR/output:/app/output" \
  senior-sorter:latest

docker ps --filter "name=senior-sorter"
