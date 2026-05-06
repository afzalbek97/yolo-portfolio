"""
Real-time webcam demo: bottle and can detection with YOLOv8-OBB.

Usage:
    cd yolo-portfolio
    source .venv/bin/activate
    python src/webcam_demo.py

    # Lower threshold if cans are missed (class imbalance causes lower can confidence)
    python src/webcam_demo.py --conf 0.20

    # Use a video file instead of webcam
    python src/webcam_demo.py --source path/to/video.mp4

Keys: 'q' quit, 's' save snapshot to results/webcam_snapshots/
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
COLORS      = {0: (0, 200, 0), 1: (0, 100, 255)}   # BGR: bottle=green, can=orange
LABEL_BG    = {0: (0, 160, 0), 1: (0, 70, 200)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0",
                   help="Camera index (0, 1, …) or path to a video file")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Detection confidence threshold (default 0.25; lower = more detections)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not OBB_WEIGHTS.exists():
        print(f"[ERROR] Weights not found: {OBB_WEIGHTS}")
        print("  → Train the model in the Colab notebook and place models/best.pt here.")
        sys.exit(1)

    print(f"[INFO] Loading model: {OBB_WEIGHTS}")
    model = YOLO(str(OBB_WEIGHTS))
    print(f"[INFO] conf={args.conf}  classes={CLASS_NAMES}")

    source: int | str = args.source
    try:
        source = int(args.source)
    except ValueError:
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        print("  → macOS: System Settings → Privacy & Security → Camera → allow Terminal/Python")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    snapshot_dir = ROOT / "results" / "webcam_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    print("[INFO] Ready — press 's' to save, 'q' to quit.")

    fps_t0  = time.time()
    n_frames = 0
    fps      = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=args.conf, verbose=False)
        annotated = frame.copy()
        n_bottle = 0
        n_can    = 0

        if results and results[0].obb is not None:
            obb   = results[0].obb
            polys = obb.xyxyxyxy.cpu().numpy().astype(np.int32)
            confs = obb.conf.cpu().numpy()
            clses = obb.cls.cpu().numpy().astype(int)

            for poly, cf, cls in zip(polys, confs, clses):
                color  = COLORS.get(cls, (200, 200, 200))
                bg     = LABEL_BG.get(cls, (80, 80, 80))
                label  = f"{CLASS_NAMES.get(cls, cls)}  {cf:.2f}"

                cv2.polylines(annotated, [poly], True, color, 3)

                x0, y0 = poly[0]
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(annotated, (x0-4, y0-th-8), (x0+tw+4, y0), bg, -1)
                cv2.putText(annotated, label, (x0, y0-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

                if cls == 0: n_bottle += 1
                else:        n_can    += 1

        # FPS (rolling 10-frame window)
        n_frames += 1
        if n_frames >= 10:
            fps     = n_frames / (time.time() - fps_t0)
            fps_t0  = time.time()
            n_frames = 0

        hud = (f"bottle: {n_bottle}   can: {n_can}   "
               f"conf≥{args.conf}   FPS: {fps:.0f}   [s=save  q=quit]")
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(annotated, hud, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Bottle vs Can — YOLOv8-OBB", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            ts       = time.strftime("%Y%m%d_%H%M%S")
            out_path = snapshot_dir / f"snapshot_{ts}.jpg"
            cv2.imwrite(str(out_path), annotated)
            print(f"[OK] Saved: {out_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
