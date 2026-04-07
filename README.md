# Robot Vision Data Studio

Desktop toolkit for collecting robot-view webcam data, reviewing labels, training YOLO models, and validating obstacle-detection workflows before Raspberry Pi deployment.

Version: `1.0`

**Developer**

- `Nakarin Sripanya`
- `Electrical Engineering Program`
- `Faculty of Industry and Technology`
- `Rajamangala University of Technology Isan`
- GitHub: `https://github.com/NKSR22/RVDStudio`

## Features

- desktop GUI for live webcam testing and field data capture
- built-in `Review / Label` tab for correcting detections before export
- built-in `Train / Evaluate` tab for dataset export, training kickoff, and live run logs
- YOLO dataset export, training, evaluation, and test scripts
- cross-platform usage on `Windows`, `macOS`, and `Ubuntu`

## Included Tools

- collecting real-world webcam data from the robot viewpoint
- testing a pretrained detection model
- saving metadata for later labeling and retraining
- training improved YOLO models
- evaluating trained models before deployment to Raspberry Pi

## Project Goals

This project is built around a simple first decision policy:

- `person` -> stop and wait
- `object` -> mark as bypass candidate

The GUI helps collect field data in the real robot camera position, while the scripts support training and testing with the collected dataset.

## Repository Status

- suitable for local development and GitHub publishing
- includes `.gitignore`, GitHub Actions smoke workflow, deployment guides, and contribution guidance
- intended as a baseline for desktop collection/training before Raspberry Pi integration

## Quick Start

### 1. Create and activate a virtual environment

Cross-platform setup helper:

```bash
python scripts/setup_venv.py
```

Manual setup:

`macOS / Ubuntu`

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`Windows PowerShell`

```bash
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Start the GUI

`macOS / Ubuntu`

```bash
source .venv/bin/activate
python -m app.gui
```

`Windows PowerShell`

```bash
.venv\Scripts\Activate.ps1
python -m app.gui
```

Put your YOLO model files (`.pt` / `.onnx`) into `models/` to show up in the GUI dropdown.

From the GUI you can now:

- capture and review samples
- export a YOLO dataset from the `Train / Evaluate` tab
- start model training from the same tab
- monitor export and training logs inside the app
- open `Help > About` to view version and developer information

## Platform Notes

- `PySide6 + OpenCV + Ultralytics` work across `Windows`, `macOS`, and `Ubuntu`
- camera index numbers may differ by OS, so use the GUI `Camera` selector or press `Refresh`
- the current desktop app is intended for data collection and model validation on a laptop/PC before deployment to Raspberry Pi
- on Ubuntu, if the GUI does not appear (Qt `xcb` plugin issues), run `python scripts/diagnose_gui.py` and install the suggested OS packages

### 3. Train a model

Prepare a YOLO dataset and dataset config, then run:

```bash
source .venv/bin/activate
python scripts/train.py --data configs/dataset.yaml --model models/yolo11n.pt --epochs 50 --imgsz 640
```

### 4. Evaluate a trained model

```bash
source .venv/bin/activate
python scripts/evaluate.py --model runs/train/robot_obstacle/weights/best.pt --data configs/dataset.yaml
```

### 5. Test a model on webcam, image folder, or video

```bash
source .venv/bin/activate
python scripts/test_model.py --model runs/train/robot_obstacle/weights/best.pt --source 0
```

## Project Layout

- `app/` desktop GUI and shared logic
- `configs/` YAML configs
- `data/raw/` captured field sessions
- `datasets/` YOLO datasets
- `docs/` operating guides
- `runs/` training and evaluation outputs
- `scripts/` setup, training, export, and testing commands

## Main Workflow

1. Mount the real webcam in the robot position.
2. Use the GUI to record field sessions and capture important frames.
3. Use the GUI `Review / Label` tab to correct labels and metadata.
4. Open `Train / Evaluate` and export the final dataset in YOLO format.
5. Start training from the same tab and monitor the run log.
6. Evaluate the trained model.
7. Test it again with live webcam or recorded video before moving to Raspberry Pi.

## Documentation

- [Collection Guide](docs/data_collection_guide.md)
- [Workflow Guide](docs/workflow.md)
- [Deployment Guide](docs/deployment_guide.md)
- [Release Checklist](docs/release_checklist.md)
- [คู่มือเก็บข้อมูล (TH)](docs/guide_th_collection.md)
- [คู่มือเทรนและทดสอบ (TH)](docs/guide_th_train_test.md)
- [คู่มือการติดตั้งและใช้งานข้ามระบบ (TH)](docs/guide_th_deployment.md)
- [Contributing](CONTRIBUTING.md)

## License

This project is released under the `MIT` license.
