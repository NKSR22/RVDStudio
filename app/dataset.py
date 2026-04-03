from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from .config import sanitize_session_name


@dataclass(slots=True)
class CaptureRecord:
    image_path: str
    timestamp: str
    source_name: str
    session_name: str
    scene_tag: str
    operator_note: str
    model_name: str
    confidence_threshold: float
    frame_width: int
    frame_height: int
    detections: list[dict[str, Any]] = field(default_factory=list)
    corrected_detections: list[dict[str, Any]] = field(default_factory=list)


def utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def session_dir(base_dir: Path, session_name: str) -> Path:
    safe_name = sanitize_session_name(session_name)
    folder = base_dir / safe_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "images").mkdir(exist_ok=True)
    (folder / "meta").mkdir(exist_ok=True)
    return folder


def save_capture(
    base_dir: Path,
    session_name: str,
    frame,
    source_name: str,
    scene_tag: str,
    operator_note: str,
    model_name: str,
    confidence_threshold: float,
    detections: list[dict[str, Any]],
) -> CaptureRecord:
    safe_session_name = sanitize_session_name(session_name)
    folder = session_dir(base_dir, safe_session_name)
    stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_rel = Path(safe_session_name) / "images" / f"{stem}.jpg"
    meta_rel = Path(safe_session_name) / "meta" / f"{stem}.json"
    image_path = base_dir / image_rel
    meta_path = base_dir / meta_rel

    frame_height, frame_width = frame.shape[:2]
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"Failed to write image to {image_path}")

    record = CaptureRecord(
        image_path=str(image_rel),
        timestamp=utc_stamp(),
        source_name=source_name,
        session_name=safe_session_name,
        scene_tag=scene_tag,
        operator_note=operator_note,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        frame_width=frame_width,
        frame_height=frame_height,
        detections=detections,
    )
    meta_path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    return record


def list_sessions(base_dir: Path) -> list[str]:
    if not base_dir.exists():
        return []
    return sorted([item.name for item in base_dir.iterdir() if item.is_dir()])


def list_capture_meta(base_dir: Path, session_name: str) -> list[Path]:
    safe_name = sanitize_session_name(session_name)
    meta_dir = base_dir / safe_name / "meta"
    if not meta_dir.exists():
        return []
    return sorted(meta_dir.glob("*.json"))


def load_capture_meta(meta_path: Path) -> dict[str, Any]:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def save_capture_meta(meta_path: Path, payload: dict[str, Any]) -> None:
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
