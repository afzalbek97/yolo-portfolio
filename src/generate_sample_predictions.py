"""
Generate portfolio-quality sample prediction images.

Draws OBB polygons with custom colors (green=bottle, orange=can) so the
rotated bounding boxes are clearly visible — unlike the default ultralytics
blue-rectangle output.

Usage:
    python src/generate_sample_predictions.py
    python src/generate_sample_predictions.py --source data/processed/test/images
    python src/generate_sample_predictions.py --conf 0.25 --top 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OBB_WEIGHTS = ROOT / "models" / "best.pt"

CLASS_NAMES = {0: "bottle", 1: "can"}
COLORS = {0: (0, 200, 0), 1: (0, 100, 255)}  # BGR: bottle=green, can=orange
LABEL_BG = {0: (0, 160, 0), 1: (0, 70, 200)}


def draw_obb_predictions(img_bgr: np.ndarray, result, conf_threshold: float) -> np.ndarray:
    """Draw OBB detections with class-specific colors and filled label background."""
    out = img_bgr.copy()
    if result.obb is None or len(result.obb.cls) == 0:
        return out

    polys = result.obb.xyxyxyxy.cpu().numpy().astype(np.int32)
    confs = result.obb.conf.cpu().numpy()
    clses = result.obb.cls.cpu().numpy().astype(int)

    for poly, conf, cls in zip(polys, confs, clses):
        if conf < conf_threshold:
            continue
        color = COLORS.get(cls, (200, 200, 200))
        bg_color = LABEL_BG.get(cls, (100, 100, 100))

        cv2.polylines(out, [poly], isClosed=True, color=color, thickness=3)

        label = f"{CLASS_NAMES.get(cls, cls)} {conf:.2f}"
        x0, y0 = poly[0]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        pad = 4
        cv2.rectangle(out, (x0 - pad, y0 - th - pad * 2), (x0 + tw + pad, y0), bg_color, -1)
        cv2.putText(out, label, (x0, y0 - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def score_image(result) -> float:
    """Score a detection result for portfolio quality.
    Prefers: high confidence, presence of 'can', visible rotation.
    """
    if result.obb is None or len(result.obb.cls) == 0:
        return -1.0

    confs = result.obb.conf.cpu().numpy()
    clses = result.obb.cls.cpu().numpy().astype(int)
    polys = result.obb.xyxyxyxy.cpu().numpy()

    score = float(confs.mean())
    if 1 in clses:
        score += 0.4  # bonus for having a can (rare class)

    # Bonus for visible rotation
    max_tilt = 0.0
    for poly in polys:
        xs, ys = poly[:, 0].astype(float), poly[:, 1].astype(float)
        angle = abs(np.degrees(np.arctan2(ys[1] - ys[0], xs[1] - xs[0])))
        tilt = min(angle, 90 - angle) if angle <= 90 else min(180 - angle, angle - 90)
        max_tilt = max(max_tilt, tilt)
    score += min(max_tilt / 30.0, 0.3)  # up to +0.3 for rotation

    return score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate OBB sample prediction images")
    p.add_argument("--source", default=str(ROOT / "results" / "sample_predictions"),
                   help="Directory of images to run inference on")
    p.add_argument("--out", default=str(ROOT / "results" / "sample_predictions"),
                   help="Output directory (default: same as source)")
    p.add_argument("--conf", type=float, default=0.20,
                   help="Confidence threshold (default 0.20)")
    p.add_argument("--top", type=int, default=3,
                   help="Number of top-scoring images to save as sample_pred_N.jpg")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not OBB_WEIGHTS.exists():
        print(f"[ERROR] Model weights not found: {OBB_WEIGHTS}")
        sys.exit(1)

    source_dir = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(source_dir.glob("*.jpg")) + sorted(source_dir.glob("*.png"))
    # Exclude already-named portfolio samples to avoid processing them twice
    images = [p for p in images if not p.stem.startswith("sample_pred")]

    if not images:
        print(f"[ERROR] No images found in {source_dir}")
        sys.exit(1)

    print(f"[INFO] Loading model: {OBB_WEIGHTS}")
    model = YOLO(str(OBB_WEIGHTS))
    print(f"[INFO] Running inference on {len(images)} images (conf≥{args.conf})...")

    scored: list[tuple[float, Path, object]] = []
    for img_path in images:
        result = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
        s = score_image(result)
        if s > 0:
            scored.append((s, img_path, result))

    scored.sort(key=lambda x: -x[0])
    print(f"[INFO] Found {len(scored)} images with detections")

    if not scored:
        print("[WARN] No detections found. Try lowering --conf.")
        return

    top_n = scored[: args.top]
    for rank, (score, img_path, result) in enumerate(top_n, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        annotated = draw_obb_predictions(img, result, args.conf)

        out_name = f"sample_pred_{rank}.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), annotated)

        clses = [int(c) for c in result.obb.cls.cpu().numpy()]
        confs = [round(float(c), 2) for c in result.obb.conf.cpu().numpy()]
        class_summary = [(CLASS_NAMES.get(c, c), cf) for c, cf in zip(clses, confs)]
        print(f"  [{rank}] {out_name}  score={score:.3f}  detections={class_summary}")

    print(f"\n[OK] {args.top} sample prediction images saved to {out_dir}")
    print("     Update README.md if needed to reference these files.")


if __name__ == "__main__":
    main()
