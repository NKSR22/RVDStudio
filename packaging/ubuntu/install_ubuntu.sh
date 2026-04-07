#!/usr/bin/env bash
set -euo pipefail

APP_NAME="RVDStudio"
INSTALL_DIR="/opt/${APP_NAME}"
BIN_LINK="/usr/local/bin/rvdstudio"
DESKTOP_DIR="/usr/share/applications"
DESKTOP_FILE="${DESKTOP_DIR}/rvdstudio.desktop"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${SCRIPT_DIR}/app"

if [[ ! -d "${PACKAGE_DIR}" ]]; then
  echo "App bundle not found: ${PACKAGE_DIR}" >&2
  exit 1
fi

echo "Installing ${APP_NAME} into ${INSTALL_DIR}"
sudo rm -rf "${INSTALL_DIR}"
sudo mkdir -p "${INSTALL_DIR}"
sudo cp -r "${PACKAGE_DIR}/." "${INSTALL_DIR}/"
sudo install -m 0755 "${SCRIPT_DIR}/rvdstudio" "${INSTALL_DIR}/rvdstudio"
sudo ln -sf "${INSTALL_DIR}/rvdstudio" "${BIN_LINK}"
sudo install -m 0644 "${SCRIPT_DIR}/rvdstudio.desktop" "${DESKTOP_FILE}"

echo
echo "${APP_NAME} installed successfully."
echo "Launch from Applications menu or run: rvdstudio"
