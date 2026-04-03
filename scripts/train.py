from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO model on the collected obstacle dataset.")
    parser.add_argument("--data", required=True, help="Path to dataset YAML.")
    parser.add_argument("--model", default="models/yolo11n.pt", help="Base YOLO model or checkpoint.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="robot_obstacle")
    parser.add_argument("--batch", type=int, default=16)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def main() -> int:
    args = parse_args()
    data_path = resolve_path(args.data)
    model_path = resolve_path(args.model)
    project_path = resolve_path(args.project)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    project_path.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(project_path),
        name=args.name,
        batch=args.batch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
