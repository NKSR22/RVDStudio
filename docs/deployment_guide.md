# Deployment Guide

This guide explains how to run the desktop tool on `Windows`, `macOS`, and `Ubuntu`, and how to move the trained model toward Raspberry Pi use.

## Scope

Use the desktop app for:

- live webcam validation
- field data collection
- dataset export
- model training and evaluation

Use Raspberry Pi later for:

- final inference deployment
- camera + LiDAR integration
- robot decision logic

## Shared Requirements

- Python `3.10+`
- a working webcam recognized by the OS
- internet access for the first dependency install
- enough disk space for `.venv`, datasets, and trained models

## Windows

### Setup

```powershell
cd C:\path\to\OpenCV
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

or:

```powershell
python scripts/setup_venv.py
```

### Run GUI

```powershell
.venv\Scripts\Activate.ps1
python -m app.gui
```

### Notes

- if PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- camera index may differ from other operating systems
- if the webcam is busy in another app, close that app first

## macOS

### Setup

```bash
cd /path/to/OpenCV
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

or:

```bash
python scripts/setup_venv.py
```

### Run GUI

```bash
source .venv/bin/activate
python -m app.gui
```

### Notes

- allow camera access for Terminal or your IDE in `System Settings > Privacy & Security > Camera`
- Apple Silicon works well for desktop validation with MPS-backed PyTorch when available
- camera probing may expose different indices than Windows or Ubuntu

## Ubuntu

### Setup

```bash
cd /path/to/OpenCV
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

or:

```bash
python3 scripts/setup_venv.py
```

### Run GUI

```bash
source .venv/bin/activate
python -m app.gui
```

### Notes

- if PySide6 or OpenCV camera access fails, install common packages such as `python3-venv`, `libgl1`, and `v4l-utils`
- check webcam detection with `ls /dev/video*`
- make sure your user has permission to access the camera device

## Training Workflow

```bash
python scripts/export_yolo_dataset.py --raw data/raw --out datasets/robot_obstacle --clear-out
python scripts/train.py --data configs/dataset.yaml --model models/yolo11n.pt --epochs 50 --imgsz 640
python scripts/evaluate.py --model runs/train/robot_obstacle/weights/best.pt --data configs/dataset.yaml
python scripts/test_model.py --model runs/train/robot_obstacle/weights/best.pt --source 0
```

## Toward Raspberry Pi

- train on desktop or laptop first
- validate the best checkpoint with real webcam scenes
- export or convert the final model only after accuracy is acceptable
- keep the robot camera position consistent between data collection and deployment

## Recommended Project Hygiene

- keep `.venv` local to each machine
- keep raw captures under `data/raw/`
- keep reusable weights under `models/`
- keep trained outputs under `runs/`

