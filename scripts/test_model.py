from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained model.")
    parser.add_argument("--model", required=True, help="Path to model weights.")
    parser.add_argument("--source", default="0", help="Camera index, image path, folder, or video path.")
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def _normalize_source(source: str):
    return int(source) if source.isdigit() else source


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def main() -> int:
    args = parse_args()
    model_path = resolve_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = YOLO(str(model_path))
    source = _normalize_source(args.source)

    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera source {source}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = model.predict(frame, conf=args.conf, verbose=False)
            plotted = results[0].plot()
            cv2.imshow("Model Test", plotted)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        return 0

    source_path = resolve_path(str(source))
    input_source = str(source_path) if source_path.exists() else str(source)
    results = model.predict(input_source, conf=args.conf, save=True, project=str(ROOT / "runs" / "predict"), name="robot_obstacle_test")
    print(f"Saved predictions for {len(results)} item(s) to runs/predict/robot_obstacle_test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
