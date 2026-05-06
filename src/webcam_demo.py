"""
Real-time webcam demo: butilka va banka detection (TWO-STAGE PIPELINE)

Stage 1: COCO-pretrained YOLOv8n (general detector, 80 classes)
         — filters out non-container objects (clothes, furniture, etc.)
Stage 2: Custom YOLOv8n-OBB (oriented bottle/can classifier)
         — runs only on regions Stage 1 confirms are containers

This cascade architecture dramatically reduces false positives caused by
domain shift from the limited Roboflow training set.

Ishlatish:
    cd ~/Documents/yolo-portfolio
    source .venv/bin/activate
    python src/webcam_demo.py

Tugmalar: 'q' chiqish, 's' screenshot saqlash
"""
from __future__ import annotations

# --- PyTorch 2.6+ compatibility patch (must run BEFORE ultralytics import) ---
import torch as _torch
_orig = _torch.load
_torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
# --- end patch ---

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OBB_WEIGHTS = ROOT / "models" / "best.pt"
# COCO pretrained — auto-downloads on first run (~6MB)
COCO_WEIGHTS = "yolov8n.pt"

CLASS_NAMES = {0: "bottle", 1: "can"}
COLORS = {0: (0, 200, 0), 1: (0, 100, 255)}  # BGR: bottle=green, can=orange

# COCO classes that look like our targets
# bottle=39, wine glass=40, cup=41
COCO_CONTAINER_CLASSES = {39, 40, 41}

# Hyper-parameters
OBB_CONF = 0.35           # OBB confidence (lower because Stage 1 already filtered)
COCO_CONF = 0.25          # Stage 1 confidence
IOU_GATE = 0.20           # Stage 2 detection must overlap a Stage 1 box by >= this


def polygon_to_aabb(poly: np.ndarray) -> tuple[float, float, float, float]:
    """4-point polygon (4,2) → axis-aligned bbox (x1,y1,x2,y2)."""
    xs = poly[:, 0]
    ys = poly[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def iou_aabb(a: tuple, b: tuple) -> float:
    """Axis-aligned IoU."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    print(f"[INFO] Stage 1 (COCO) yuklanmoqda: {COCO_WEIGHTS}")
    coco_model = YOLO(COCO_WEIGHTS)
    print(f"[INFO] Stage 2 (OBB) yuklanmoqda: {OBB_WEIGHTS}")
    obb_model = YOLO(str(OBB_WEIGHTS))
    print("[INFO] Ikki bosqichli pipeline tayyor")

    print("[INFO] Kamera ochilmoqda...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera ochilmadi. System Settings → Privacy & Security → Camera "
              "ostida Terminal/Python ga ruxsat berilganini tekshiring.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Tayyor! 'q' — chiqish, 's' — screenshot")
    snapshot_dir = ROOT / "results" / "webcam_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

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
            for cls_id, xyxy in zip(boxes.cls.cpu().numpy(),
                                     boxes.xyxy.cpu().numpy()):
                if int(cls_id) in COCO_CONTAINER_CLASSES:
                    coco_aabbs.append(tuple(map(float, xyxy)))

        # === STAGE 2: OBB on full frame, then gate by COCO ===
        obb_results = obb_model.predict(frame, conf=OBB_CONF, verbose=False)

        # Start from a clean frame and draw only confirmed detections
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
                # Gate: OBB box must overlap a COCO container box
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

        # FPS counter (rolling 10 frames)
        n_frames += 1
        if n_frames >= 10:
            dt = time.time() - fps_t0
            fps = n_frames / dt
            fps_t0 = time.time()
            n_frames = 0

        # HUD
        hud1 = f"Detections: {kept}  (rejected by COCO gate: {rejected})"
        hud2 = f"FPS: {fps:.1f}  |  Stage1 containers: {len(coco_aabbs)}  |  q: quit, s: save"
        cv2.putText(annotated, hud1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, hud2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (180, 180, 180), 2, cv2.LINE_AA)

        cv2.imshow("Bottle vs Can - Two-Stage Live Demo", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = snapshot_dir / f"snapshot_{ts}.jpg"
            cv2.imwrite(str(out_path), annotated)
            print(f"[OK] Saqlandi: {out_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Tugadi")


if __name__ == "__main__":
    main()
