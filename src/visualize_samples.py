"""
Visualise ground-truth OBB annotations from the training set.

Filters out images with missing, empty, or invalid label files so only
properly annotated bottle/can images appear in the portfolio output.

Usage:
    python src/visualize_samples.py
    python src/visualize_samples.py --n 6 --min-boxes 1 --seed 42
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "dataset_samples"

CLASS_NAMES = {0: "bottle", 1: "can"}
COLORS = {0: (0, 200, 0), 1: (0, 100, 255)}   # BGR: bottle=green, can=orange
LABEL_BG = {0: (0, 160, 0), 1: (0, 70, 200)}


def parse_label(label_path: Path, img_w: int, img_h: int) -> list[tuple]:
    """Return list of (cls, pts_int32) from a YOLO-OBB label file."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:9]))
            pts = np.array(
                [[coords[i] * img_w, coords[i + 1] * img_h] for i in range(0, 8, 2)],
                dtype=np.int32,
            )
            boxes.append((cls, pts))
    return boxes


def draw_obb(img: np.ndarray, boxes: list[tuple]) -> np.ndarray:
    out = img.copy()
    for cls, pts in boxes:
        color = COLORS.get(cls, (200, 200, 200))
        bg_color = LABEL_BG.get(cls, (100, 100, 100))
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=3)
        label = CLASS_NAMES.get(cls, str(cls))
        x0, y0 = pts[0]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        pad = 4
        cv2.rectangle(out, (x0 - pad, y0 - th - pad * 2), (x0 + tw + pad, y0), bg_color, -1)
        cv2.putText(out, label, (x0, y0 - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise ground-truth OBB annotations")
    p.add_argument("--split", default="train", help="Dataset split to sample from (default: train)")
    p.add_argument("--n", type=int, default=8, help="Number of samples to generate")
    p.add_argument("--min-boxes", type=int, default=1,
                   help="Minimum number of annotation boxes required (filters out unannotated images)")
    p.add_argument("--min-size", type=int, default=300,
                   help="Minimum image dimension in pixels (filters out tiny images)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    img_dir = PROC / args.split / "images"
    lbl_dir = PROC / args.split / "labels"

    if not img_dir.exists():
        print(f"[ERROR] Image directory not found: {img_dir}")
        print("  → Download the dataset first (see COLAB_QUICKSTART.md)")
        return

    all_images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    # Filter: only keep images that are large enough and have valid annotations
    valid: list[tuple[Path, list]] = []
    for img_path in all_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if min(h, w) < args.min_size:
            continue  # skip tiny images

        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        boxes = parse_label(lbl_path, w, h)
        if len(boxes) < args.min_boxes:
            continue  # skip unannotated or under-annotated images

        valid.append((img_path, boxes))

    print(f"[INFO] {len(valid)} valid images out of {len(all_images)} total")
    if not valid:
        print("[WARN] No valid images found — check --min-size and --min-boxes settings.")
        return

    random.seed(args.seed)
    sample = random.sample(valid, min(args.n, len(valid)))

    saved = 0
    for i, (img_path, boxes) in enumerate(sample, start=1):
        img = cv2.imread(str(img_path))
        annotated = draw_obb(img, boxes)
        out_path = OUT / f"sample_{i:02d}.jpg"
        cv2.imwrite(str(out_path), annotated)
        class_summary = [CLASS_NAMES.get(c, c) for c, _ in boxes]
        print(f"  sample_{i:02d}.jpg  boxes={len(boxes)}  classes={class_summary[:5]}")
        saved += 1

    print(f"\n[OK] {saved} annotation samples saved to {OUT}")


if __name__ == "__main__":
    main()
