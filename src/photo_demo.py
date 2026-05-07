"""
Photo capture demo: press SPACE to capture a frame, model runs inference on it.

This approach works better than live video for models trained on studio images —
you control the framing and lighting, reducing domain shift.

Usage:
    python src/photo_demo.py
    python src/photo_demo.py --conf 0.20

Keys:
    SPACE — capture frame and run detection
    s     — save last annotated result to results/webcam_snapshots/
    q     — quit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OBB_WEIGHTS = ROOT / "models" / "best.pt"

CLASS_NAMES = {0: "bottle", 1: "can"}
COLORS      = {0: (0, 200, 0), 1: (0, 100, 255)}
LABEL_BG    = {0: (0, 160, 0), 1: (0, 70, 200)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0")
    p.add_argument("--conf", type=float, default=0.20)
    return p.parse_args()


def run_inference(model, frame, conf):
    results = model.predict(frame, conf=conf, verbose=False)
    annotated = frame.copy()
    n_bottle = n_can = 0

    if results and results[0].obb is not None:
        obb   = results[0].obb
        polys = obb.xyxyxyxy.cpu().numpy().astype(np.int32)
        confs = obb.conf.cpu().numpy()
        clses = obb.cls.cpu().numpy().astype(int)

        for poly, cf, cls in zip(polys, confs, clses):
            color = COLORS.get(cls, (200, 200, 200))
            bg    = LABEL_BG.get(cls, (80, 80, 80))
            label = f"{CLASS_NAMES.get(cls, cls)}  {cf:.2f}"
            cv2.polylines(annotated, [poly], True, color, 3)
            x0, y0 = poly[0]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(annotated, (x0-4, y0-th-8), (x0+tw+4, y0), bg, -1)
            cv2.putText(annotated, label, (x0, y0-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            if cls == 0: n_bottle += 1
            else:        n_can += 1

    return annotated, n_bottle, n_can


def main() -> None:
    args = parse_args()

    if not OBB_WEIGHTS.exists():
        print(f"[ERROR] Weights not found: {OBB_WEIGHTS}")
        sys.exit(1)

    print(f"[INFO] Loading model: {OBB_WEIGHTS}")
    model = YOLO(str(OBB_WEIGHTS))

    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    snapshot_dir = ROOT / "results" / "webcam_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Ready — SPACE to detect, 's' to save, 'q' to quit")

    last_annotated = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Show live preview with instructions
        preview = frame.copy()
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(preview, "Hold bottle or can close to camera — press SPACE to detect",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        display = last_annotated if last_annotated is not None else preview
        cv2.imshow("Bottle vs Can — Photo Mode", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord(" "):
            print("[INFO] Detecting...")
            annotated, n_bottle, n_can = run_inference(model, frame, args.conf)
            hud = (f"bottle: {n_bottle}   can: {n_can}   "
                   f"conf≥{args.conf}   [SPACE=new  s=save  q=quit]")
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 36), (0, 0, 0), -1)
            cv2.putText(annotated, hud, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            last_annotated = annotated
            print(f"[OK] bottle={n_bottle}  can={n_can}")

        if key == ord("s") and last_annotated is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out = snapshot_dir / f"snapshot_{ts}.jpg"
            cv2.imwrite(str(out), last_annotated)
            print(f"[OK] Saved: {out}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
