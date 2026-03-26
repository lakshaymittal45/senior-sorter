#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/senior-sorter"

sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin unzip
sudo systemctl enable docker
sudo systemctl start docker

sudo mkdir -p "$APP_DIR"
sudo chown -R "$USER":"$USER" "$APP_DIR"

echo "Place project files into $APP_DIR and run deploy/aws/run_container.sh"
