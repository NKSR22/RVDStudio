from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"
RELEASE_ROOT = ROOT / "release" / "ubuntu"
PACKAGING_DIR = ROOT / "packaging" / "ubuntu"
APP_NAME = "RVDStudio"
VERSION = "1.0"
RELEASE_NAME = f"{APP_NAME}-{VERSION}-ubuntu-x86_64"
STAGE_DIR = RELEASE_ROOT / RELEASE_NAME
PYINSTALLER_NAME = APP_NAME


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_binary() -> Path:
    pyinstaller_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--onedir",
        "--name",
        PYINSTALLER_NAME,
        "--collect-data",
        "ultralytics",
        "--collect-submodules",
        "ultralytics",
        "--copy-metadata",
        "ultralytics",
        "--copy-metadata",
        "torch",
        "--hidden-import",
        "cv2",
        "--hidden-import",
        "numpy",
        "--hidden-import",
        "torch",
        "--hidden-import",
        "PySide6.QtCore",
        "--hidden-import",
        "PySide6.QtGui",
        "--hidden-import",
        "PySide6.QtWidgets",
        "--hidden-import",
        "PySide6.QtDBus",
        "scripts/launch_gui.py",
    ]
    run(pyinstaller_cmd)
    return DIST_DIR / PYINSTALLER_NAME


def create_installer_assets() -> None:
    shutil.copy2(PACKAGING_DIR / "install_ubuntu.sh", STAGE_DIR / "install_ubuntu.sh")
    shutil.copy2(PACKAGING_DIR / "uninstall_ubuntu.sh", STAGE_DIR / "uninstall_ubuntu.sh")
    shutil.copy2(PACKAGING_DIR / "rvdstudio", STAGE_DIR / "rvdstudio")
    shutil.copy2(PACKAGING_DIR / "rvdstudio.desktop", STAGE_DIR / "rvdstudio.desktop")
    for path in [
        STAGE_DIR / "install_ubuntu.sh",
        STAGE_DIR / "uninstall_ubuntu.sh",
        STAGE_DIR / "rvdstudio",
    ]:
        path.chmod(0o755)


def create_release_docs() -> None:
    install_guide = f"""# Ubuntu Installation Guide

Package: `{RELEASE_NAME}`

## Contents

- compiled app bundle in `app/`
- installer script: `install_ubuntu.sh`
- uninstaller script: `uninstall_ubuntu.sh`
- desktop entry: `rvdstudio.desktop`

## System Requirements

- Ubuntu 22.04+ recommended
- x86_64 CPU
- graphical desktop session
- webcam access if you want to use live capture

## Install Steps

1. Open Terminal in this folder.
2. Run:

```bash
chmod +x install_ubuntu.sh
./install_ubuntu.sh
```

3. Start the app from the Applications menu or run:

```bash
rvdstudio
```

## Remove the App

```bash
chmod +x uninstall_ubuntu.sh
./uninstall_ubuntu.sh
```

## Notes

- The installer copies the app to `/opt/RVDStudio`
- A launcher link is created at `/usr/local/bin/rvdstudio`
- A desktop entry is installed at `/usr/share/applications/rvdstudio.desktop`
"""
    user_guide = """# Ubuntu User Guide

## Main Areas

- `Live Capture`: open a camera, load a model, and capture frames
- `Review / Label`: inspect saved samples and correct bounding boxes
- `Train / Evaluate`: export datasets, start training, and watch logs
- `Help > About`: view version and developer details

## Typical Workflow

1. Open `Live Capture`
2. Start the camera and optionally load a YOLO model
3. Save important frames with `Capture Frame`
4. Open `Review / Label` to correct labels
5. Open `Train / Evaluate`
6. Export the dataset
7. Start training and monitor the log panel

## Data Locations

- raw captures: `data/raw/`
- datasets: `datasets/`
- models: `models/`
- training runs: `runs/`

## Troubleshooting

- If the GUI does not open, run `python scripts/diagnose_gui.py` in the source project
- If the camera is busy, close other apps using the webcam
- If a training task fails, review the log panel in `Train / Evaluate`
"""
    write_text(STAGE_DIR / "docs" / "UBUNTU_INSTALL_GUIDE.md", install_guide)
    write_text(STAGE_DIR / "docs" / "UBUNTU_USER_GUIDE.md", user_guide)


def assemble_release(app_bundle_dir: Path) -> None:
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    (STAGE_DIR / "app").mkdir(parents=True, exist_ok=True)
    shutil.copytree(app_bundle_dir, STAGE_DIR / "app", dirs_exist_ok=True)
    create_installer_assets()
    create_release_docs()
    shutil.copy2(ROOT / "README.md", STAGE_DIR / "README.md")
    shutil.copy2(ROOT / "LICENSE", STAGE_DIR / "LICENSE")


def make_archive() -> Path:
    archive_path = RELEASE_ROOT / f"{RELEASE_NAME}.tar.gz"
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(STAGE_DIR, arcname=STAGE_DIR.name)
    return archive_path


def main() -> int:
    RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
    app_bundle_dir = build_binary()
    assemble_release(app_bundle_dir)
    archive = make_archive()
    print(f"Ubuntu release folder: {STAGE_DIR}")
    print(f"Ubuntu release archive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
