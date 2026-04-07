#!/usr/bin/env bash
set -euo pipefail

APP_NAME="RVDStudio"
INSTALL_DIR="/opt/${APP_NAME}"
BIN_LINK="/usr/local/bin/rvdstudio"
DESKTOP_FILE="/usr/share/applications/rvdstudio.desktop"

echo "Removing ${APP_NAME}..."
sudo rm -rf "${INSTALL_DIR}"
sudo rm -f "${BIN_LINK}"
sudo rm -f "${DESKTOP_FILE}"
echo "${APP_NAME} removed."
