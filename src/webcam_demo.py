"""
Real-time webcam demo: bottle and can detection (TWO-STAGE PIPELINE)

Stage 1: COCO-pretrained YOLOv8n (general detector, 80 classes)
         — filters out non-container objects (clothes, furniture, etc.)
Stage 2: Custom YOLOv8n-OBB (oriented bottle/can classifier)
         — keeps only OBB detections that overlap a Stage 1 container box

This cascade architecture dramatically reduces false positives caused by
domain shift from the limited Roboflow training set.

Usage:
    cd yolo-portfolio
    source .venv/bin/activate
    python src/webcam_demo.py

Keys: 'q' quit, 's' save screenshot
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from detection_utils import iou_aabb, polygon_to_aabb

ROOT = Path(__file__).resolve().parent.parent
OBB_WEIGHTS = ROOT / "models" / "best.pt"
COCO_WEIGHTS = "yolov8n.pt"  # auto-downloaded by ultralytics on first run

CLASS_NAMES = {0: "bottle", 1: "can"}
COLORS = {0: (0, 200, 0), 1: (0, 100, 255)}  # BGR: bottle=green, can=orange

# COCO class IDs that correspond to container-like objects
# bottle=39, wine glass=40, cup=41
COCO_CONTAINER_CLASSES = {39, 40, 41}

OBB_CONF = 0.35   # lower threshold is fine — Stage 1 already pre-filters
COCO_CONF = 0.25
IOU_GATE = 0.20   # minimum AABB overlap to keep an OBB detection


def main() -> None:
    if not OBB_WEIGHTS.exists():
        print(f"[ERROR] Model weights not found: {OBB_WEIGHTS}")
        print("  → Train the model in the Colab notebook and copy models/best.pt here.")
        sys.exit(1)

    print(f"[INFO] Loading Stage 1 (COCO): {COCO_WEIGHTS}")
    coco_model = YOLO(COCO_WEIGHTS)
    print(f"[INFO] Loading Stage 2 (OBB): {OBB_WEIGHTS}")
    obb_model = YOLO(str(OBB_WEIGHTS))
    print("[INFO] Two-stage pipeline ready")

    print("[INFO] Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera could not be opened.")
        print("  → macOS: System Settings → Privacy & Security → Camera → allow Terminal/Python")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    snapshot_dir = ROOT / "results" / "webcam_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    print("[INFO] Ready! Press 'q' to quit, 's' to save a snapshot.")

    fps_t0 = time.time()
    n_frames = 0
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # === STAGE 1: COCO general detector ===
        coco_results = coco_model.predict(frame, conf=COCO_CONF, verbose=False)
        coco_aabbs: list[tuple] = []
        if coco_results and coco_results[0].boxes is not None:
            boxes = coco_results[0].boxes
            for cls_id, xyxy in zip(boxes.cls.cpu().numpy(), boxes.xyxy.cpu().numpy()):
                if int(cls_id) in COCO_CONTAINER_CLASSES:
                    coco_aabbs.append(tuple(map(float, xyxy)))

        # === STAGE 2: custom OBB, then gate against Stage 1 results ===
        obb_results = obb_model.predict(frame, conf=OBB_CONF, verbose=False)

        annotated = frame.copy()
        kept = 0
        rejected = 0

        if obb_results and obb_results[0].obb is not None:
            obb = obb_results[0].obb
            polys = obb.xyxyxyxy.cpu().numpy()
            confs = obb.conf.cpu().numpy()
            clses = obb.cls.cpu().numpy()
            for poly, cf, cls in zip(polys, confs, clses):
                aabb = polygon_to_aabb(poly)
                best_iou = max((iou_aabb(aabb, c) for c in coco_aabbs), default=0.0)
                if best_iou >= IOU_GATE:
                    cls_int = int(cls)
                    color = COLORS.get(cls_int, (255, 255, 255))
                    pts = poly.astype(np.int32)
                    cv2.polylines(annotated, [pts], True, color, 3)
                    label = f"{CLASS_NAMES.get(cls_int, cls_int)} {cf:.2f}"
                    cv2.putText(annotated, label, tuple(pts[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    kept += 1
                else:
                    rejected += 1

        # Rolling FPS counter (updated every 10 frames)
        n_frames += 1
        if n_frames >= 10:
            fps = n_frames / (time.time() - fps_t0)
            fps_t0 = time.time()
            n_frames = 0

        hud1 = f"Detections: {kept}  (rejected by COCO gate: {rejected})"
        hud2 = f"FPS: {fps:.1f}  |  Stage1 containers: {len(coco_aabbs)}  |  q: quit, s: save"
        cv2.putText(annotated, hud1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, hud2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (180, 180, 180), 2, cv2.LINE_AA)

        cv2.imshow("Bottle vs Can — Two-Stage Live Demo", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = snapshot_dir / f"snapshot_{ts}.jpg"
            cv2.imwrite(str(out_path), annotated)
            print(f"[OK] Saved: {out_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done")


if __name__ == "__main__":
    main()
