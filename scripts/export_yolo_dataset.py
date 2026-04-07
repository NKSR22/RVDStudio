from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExportItem:
    image_abs: Path
    meta_abs: Path
    stem: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export captured GUI data into a YOLO detection dataset.")
    p.add_argument("--raw", default="data/raw", help="Raw capture root (default: data/raw).")
    p.add_argument("--out", default="datasets/robot_obstacle", help="Output YOLO dataset dir.")
    p.add_argument("--train", type=float, default=0.8, help="Train split ratio.")
    p.add_argument("--val", type=float, default=0.1, help="Val split ratio.")
    p.add_argument("--test", type=float, default=0.1, help="Test split ratio.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    p.add_argument("--min-conf", type=float, default=0.0, help="Minimum confidence to keep a detection.")
    p.add_argument("--clear-out", action="store_true", help="Delete the previous exported dataset before writing.")
    p.add_argument(
        "--label-mode",
        choices=["prelabel", "empty"],
        default="prelabel",
        help="prelabel: use detections in meta JSON; empty: create empty label files for manual labeling.",
    )
    return p.parse_args()


def _ensure_dirs(out_dir: Path) -> None:
    for split in ["train", "val", "test"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def _clean_out_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)


def _collect_items(raw_dir: Path) -> list[ExportItem]:
    items: list[ExportItem] = []
    for meta_path in raw_dir.rglob("meta/*.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        image_rel = meta.get("image_path")
        if not image_rel:
            continue
        image_abs = (raw_dir / image_rel).resolve()
        if not image_abs.exists():
            continue
        items.append(ExportItem(image_abs=image_abs, meta_abs=meta_path.resolve(), stem=image_abs.stem))
    return items


def _xyxy_to_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(float(width - 1), x1))
    x2 = max(0.0, min(float(width - 1), x2))
    y1 = max(0.0, min(float(height - 1), y1))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    w = max(0.0, (x2 - x1) / width)
    h = max(0.0, (y2 - y1) / height)
    return (cx, cy, w, h)


def _map_class(label: str) -> int | None:
    # Dataset classes are fixed:
    # 0: person
    # 1: obstacle
    if label == "person":
        return 0
    # treat everything else as obstacle (when using prelabels)
    return 1


def _export_detections(meta: dict) -> list[dict]:
    corrected = meta.get("corrected_detections")
    if isinstance(corrected, list):
        # Fresh captures historically stored an empty corrected_detections list before any review happened.
        # Only prefer corrected labels when a review timestamp exists or when the list actually contains boxes.
        if meta.get("corrected_at") or corrected:
            return corrected
    detections = meta.get("detections")
    if isinstance(detections, list):
        return detections
    return []


def _write_labels_from_meta(meta: dict, label_path: Path, min_conf: float) -> None:
    width = int(meta.get("frame_width") or 0)
    height = int(meta.get("frame_height") or 0)
    detections = _export_detections(meta)
    lines: list[str] = []

    if width <= 0 or height <= 0:
        label_path.write_text("", encoding="utf-8")
        return

    for det in detections:
        try:
            label = str(det.get("label", ""))
            conf = float(det.get("confidence", 0.0))
            bbox = det.get("bbox", [])
            if conf < min_conf:
                continue
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            cls = _map_class(label)
            if cls is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            cx, cy, w, h = _xyxy_to_yolo(x1, y1, x2, y2, width, height)
            # Drop degenerate boxes.
            if w <= 0.0 or h <= 0.0:
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        except Exception:
            continue

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _split(items: list[ExportItem], train: float, val: float, test: float, seed: int) -> dict[str, list[ExportItem]]:
    if train < 0 or val < 0 or test < 0:
        raise ValueError("Split ratios must be >= 0.")
    total = train + val + test
    if total <= 0:
        raise ValueError("At least one split ratio must be > 0.")

    train /= total
    val /= total
    test /= total

    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train)
    n_val = int(n * val)
    # remainder goes to test
    train_items = shuffled[:n_train]
    val_items = shuffled[n_train : n_train + n_val]
    test_items = shuffled[n_train + n_val :]
    return {"train": train_items, "val": val_items, "test": test_items}


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw).resolve()
    out_dir = Path(args.out).resolve()
    if args.clear_out:
        _clean_out_dir(out_dir)
    _ensure_dirs(out_dir)

    items = _collect_items(raw_dir)
    if not items:
        raise SystemExit(f"No capture items found under: {raw_dir}")

    splits = _split(items, train=args.train, val=args.val, test=args.test, seed=args.seed)
    exported = 0

    for split_name, split_items in splits.items():
        for item in split_items:
            out_img = out_dir / "images" / split_name / f"{item.stem}.jpg"
            out_lbl = out_dir / "labels" / split_name / f"{item.stem}.txt"

            shutil.copy2(item.image_abs, out_img)

            if args.label_mode == "empty":
                out_lbl.write_text("", encoding="utf-8")
            else:
                meta = json.loads(item.meta_abs.read_text(encoding="utf-8"))
                _write_labels_from_meta(meta, out_lbl, min_conf=args.min_conf)

            exported += 1

    print(f"Exported {exported} images to {out_dir}")
    print("Class map: 0=person, 1=obstacle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
