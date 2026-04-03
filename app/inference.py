from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional import during setup
    YOLO = None


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    decision_hint: str


class Detector:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.model_name = "none"

    def load(self, model_path: str) -> None:
        if not model_path:
            self.model = None
            self.model_name = "none"
            return
        if YOLO is None:
            raise RuntimeError("Ultralytics is not installed. Please install requirements first.")
        self.model = YOLO(model_path)
        self.model_name = Path(model_path).name

    def predict(self, frame, conf: float = 0.25) -> list[Detection]:
        if self.model is None:
            return []

        results = self.model.predict(frame, conf=conf, verbose=False)
        if not results:
            return []

        result = results[0]
        detections: list[Detection] = []
        names = result.names
        boxes = result.boxes
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            if isinstance(names, dict):
                label = str(names.get(cls_id, str(cls_id)))
            elif isinstance(names, list) and 0 <= cls_id < len(names):
                label = str(names[cls_id])
            else:
                label = str(cls_id)
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            decision_hint = "stop_wait" if label == "person" else "bypass_candidate"
            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    decision_hint=decision_hint,
                )
            )
        return detections


def draw_detections(frame, detections: list[Detection]):
    output = frame.copy()
    h, w = output.shape[:2]

    # Visual guide: split preview into left/center/right zones.
    zone_color = (0, 255, 0)  # BGR
    x_left = w // 3
    x_right = (2 * w) // 3
    cv2.line(output, (x_left, 0), (x_left, h - 1), zone_color, 2)
    cv2.line(output, (x_right, 0), (x_right, h - 1), zone_color, 2)

    for det in detections:
        color = (0, 0, 255) if det.label == "person" else (0, 180, 255)
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        text = f"{det.label} {det.confidence:.2f} {det.decision_hint}"
        cv2.putText(output, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return output
