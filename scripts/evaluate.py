from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained YOLO model.")
    parser.add_argument("--model", required=True, help="Path to trained model weights.")
    parser.add_argument("--data", required=True, help="Path to dataset YAML.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="runs/eval")
    parser.add_argument("--name", default="robot_obstacle_eval")
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
    metrics = model.val(data=str(data_path), imgsz=args.imgsz, project=str(project_path), name=args.name)
    summary = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    summary_path = project_path / args.name / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
