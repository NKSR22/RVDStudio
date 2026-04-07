# Ubuntu User Guide

## Main Interface

- `Live Capture` for camera preview and frame capture
- `Review / Label` for checking and correcting detections
- `Train / Evaluate` for dataset export, training, and log viewing
- `Help > About` for version and developer details

## Recommended Workflow

1. Start the app with `rvdstudio`
2. Open `Live Capture` and test the camera
3. Save frames into a session
4. Correct labels in `Review / Label`
5. Export a YOLO dataset in `Train / Evaluate`
6. Start training and monitor the log panel

## Files Created by the App

- raw captures in `data/raw/`
- datasets in `datasets/`
- training runs in `runs/`
- reusable models in `models/`

## Troubleshooting

- If the GUI fails to open in the source checkout, run `python scripts/diagnose_gui.py`
- If camera access fails, close other apps using the webcam
- If training fails, review the run log in the `Train / Evaluate` tab
