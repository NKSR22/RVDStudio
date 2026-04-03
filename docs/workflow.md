# Workflow Guide

## End-to-End Flow

1. Collect real webcam data with the GUI.
2. Review captures in the GUI `Review / Label` tab and correct labels.
3. Clean bad samples and refine YOLO labels.
4. Organize the dataset into train, val, and test splits.
5. Update `configs/dataset.yaml` if the dataset path changes.
6. Run training with `scripts/train.py`.
7. Run validation with `scripts/evaluate.py`.
8. Run live or offline inference with `scripts/test_model.py`.
9. Export and optimize the best model for Raspberry Pi in a later step.

## Suggested Directory Convention

Example YOLO dataset layout:

```text
datasets/robot_obstacle/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

## Training Example

```bash
python scripts/train.py \
  --data configs/dataset.yaml \
  --model models/yolo11n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16
```

## Evaluation Example

```bash
python scripts/evaluate.py \
  --model runs/train/robot_obstacle/weights/best.pt \
  --data configs/dataset.yaml
```

## Live Test Example

```bash
python scripts/test_model.py \
  --model runs/train/robot_obstacle/weights/best.pt \
  --source 0
```

## Baseline Recommendation

- start with `yolo11n.pt` or another small YOLO model
- keep class count small at first
- validate on real camera scenes, not only curated images
- treat person detection quality as the highest priority
