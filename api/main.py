"""
FastAPI server for Bottle vs Can YOLOv8-OBB detection.

Architecture: TWO-STAGE CASCADE PIPELINE
    Stage 1: COCO-pretrained YOLOv8n (general detector, knows 80 classes)
             — filters out non-container objects (clothes, walls, hangers, etc.)
    Stage 2: Custom YOLOv8n-OBB (oriented bounding-box, bottle vs can)
             — final detections only kept if Stage 1 also flagged a container
               in the same region

This dramatically reduces false positives caused by domain shift from the
small (~292-image) Roboflow training set.

Endpoints:
    GET  /health             — liveness check
    POST /predict            — multipart image, returns JSON detections (cascade-filtered)
    POST /predict/visualize  — multipart image, returns annotated PNG
    POST /predict/raw        — multipart image, BYPASSES Stage 1 (raw OBB output)
                               useful for debugging / comparing pipelines

Run:
    cd yolo-portfolio
    source .venv/bin/activate
    uvicorn api.main:app --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

# Make src/ importable when running as `uvicorn api.main:app` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

from detection_utils import iou_aabb as _iou_aabb, polygon_to_aabb as _polygon_to_aabb

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OBB_WEIGHTS = ROOT / "models" / "best.pt"
OBB_WEIGHTS_PATH = Path(os.getenv("YOLO_WEIGHTS", DEFAULT_OBB_WEIGHTS))
COCO_WEIGHTS = os.getenv("COCO_WEIGHTS", "yolov8n.pt")  # auto-downloads on first run

CLASS_NAMES = {0: "bottle", 1: "can"}

# COCO classes that count as "container-like": bottle=39, wine glass=40, cup=41
COCO_CONTAINER_CLASSES = {39, 40, 41}

# Cascade hyper-parameters
DEFAULT_OBB_CONF = 0.35
DEFAULT_COCO_CONF = 0.25
DEFAULT_IOU_GATE = 0.20

_obb_model: YOLO | None = None
_coco_model: YOLO | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _obb_model, _coco_model
    if not OBB_WEIGHTS_PATH.exists():
        print(f"[WARN] OBB weights not found: {OBB_WEIGHTS_PATH}. /predict will return 503.")
    else:
        print(f"[INFO] Loading Stage 2 OBB model: {OBB_WEIGHTS_PATH}")
        _obb_model = YOLO(str(OBB_WEIGHTS_PATH))
        print(f"[INFO] Loading Stage 1 COCO model: {COCO_WEIGHTS}")
        _coco_model = YOLO(COCO_WEIGHTS)
        print("[INFO] Two-stage pipeline ready")
    yield
    _obb_model = None
    _coco_model = None


app = FastAPI(
    title="Bottle vs Can Detection API",
    description=(
        "Two-stage YOLOv8-OBB cascade: a COCO general detector filters non-container "
        "regions, then a custom OBB model produces oriented bottle/can boxes. "
        "Returns 4-corner polygons (8 coordinates) per detection."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# --- Schemas ---
class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    polygon: List[List[float]]  # 4 corner points of the OBB
    coco_iou: float = 0.0       # IoU with the nearest Stage 1 container box (debug)


class PredictionResponse(BaseModel):
    image_size: List[int]      # [width, height]
    num_detections: int        # cascade-filtered count
    num_rejected: int          # OBB detections rejected by COCO gate
    coco_containers_found: int # how many container boxes Stage 1 produced
    detections: List[Detection]


# --- Helpers ---
def _read_image(file_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {e}") from e
    return np.array(img)


def _ensure_models() -> tuple[YOLO, YOLO]:
    if _obb_model is None or _coco_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Models not ready. Check that weights exist at "
                f"'{OBB_WEIGHTS_PATH}' and restart the server."
            ),
        )
    return _obb_model, _coco_model


def _run_cascade(
    img_rgb: np.ndarray,
    obb_conf: float,
    coco_conf: float,
    iou_gate: float,
):
    """Returns (kept_detections, rejected_count, coco_aabbs)."""
    obb_model, coco_model = _ensure_models()

    # Stage 1
    coco_results = coco_model.predict(img_rgb, conf=coco_conf, verbose=False)
    coco_aabbs: list[tuple] = []
    if coco_results and coco_results[0].boxes is not None:
        boxes = coco_results[0].boxes
        for cls_id, xyxy in zip(boxes.cls.cpu().numpy(), boxes.xyxy.cpu().numpy()):
            if int(cls_id) in COCO_CONTAINER_CLASSES:
                coco_aabbs.append(tuple(map(float, xyxy)))

    # Stage 2
    obb_results = obb_model.predict(img_rgb, conf=obb_conf, verbose=False)

    kept: list[Detection] = []
    rejected = 0
    if obb_results and obb_results[0].obb is not None:
        obb = obb_results[0].obb
        polys = obb.xyxyxyxy.cpu().numpy() if hasattr(obb.xyxyxyxy, "cpu") else obb.xyxyxyxy
        confs = obb.conf.cpu().numpy() if hasattr(obb.conf, "cpu") else obb.conf
        clses = obb.cls.cpu().numpy() if hasattr(obb.cls, "cpu") else obb.cls
        for poly, cf, cls in zip(polys, confs, clses):
            aabb = _polygon_to_aabb(poly)
            best_iou = max((_iou_aabb(aabb, c) for c in coco_aabbs), default=0.0)
            if best_iou >= iou_gate:
                cls_int = int(cls)
                kept.append(Detection(
                    class_id=cls_int,
                    class_name=CLASS_NAMES.get(cls_int, str(cls_int)),
                    confidence=float(cf),
                    polygon=[[float(x), float(y)] for x, y in poly],
                    coco_iou=float(best_iou),
                ))
            else:
                rejected += 1

    return kept, rejected, coco_aabbs


# --- Routes ---
@app.get("/health")
def health():
    return {
        "status": "ok",
        "obb_loaded": _obb_model is not None,
        "coco_loaded": _coco_model is not None,
        "obb_weights": str(OBB_WEIGHTS_PATH),
        "coco_weights": COCO_WEIGHTS,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    obb_conf: float = DEFAULT_OBB_CONF,
    coco_conf: float = DEFAULT_COCO_CONF,
    iou_gate: float = DEFAULT_IOU_GATE,
):
    """Cascade detection. Returns OBB polygons confirmed by COCO container detector."""
    img = _read_image(await file.read())
    h, w = img.shape[:2]
    kept, rejected, coco_aabbs = _run_cascade(img, obb_conf, coco_conf, iou_gate)
    return PredictionResponse(
        image_size=[w, h],
        num_detections=len(kept),
        num_rejected=rejected,
        coco_containers_found=len(coco_aabbs),
        detections=kept,
    )


@app.post("/predict/visualize")
async def predict_visualize(
    file: UploadFile = File(...),
    obb_conf: float = DEFAULT_OBB_CONF,
    coco_conf: float = DEFAULT_COCO_CONF,
    iou_gate: float = DEFAULT_IOU_GATE,
):
    """Cascade detection rendered onto the input image as a PNG."""
    img = _read_image(await file.read())
    kept, rejected, _ = _run_cascade(img, obb_conf, coco_conf, iou_gate)

    bgr = img[:, :, ::-1].copy()
    colors = {0: (0, 200, 0), 1: (0, 100, 255)}  # BGR
    for d in kept:
        pts = np.array(d.polygon, dtype=np.int32)
        cv2.polylines(bgr, [pts], True, colors.get(d.class_id, (255, 255, 255)), 3)
        label = f"{d.class_name} {d.confidence:.2f}"
        cv2.putText(bgr, label, tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    colors.get(d.class_id, (255, 255, 255)), 2)

    hud = f"Cascade: {len(kept)} kept, {rejected} rejected"
    cv2.putText(bgr, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)

    pil = Image.fromarray(bgr[:, :, ::-1])
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/predict/raw")
async def predict_raw(file: UploadFile = File(...), conf: float = 0.25):
    """RAW OBB output without Stage 1 cascade. Useful for debugging."""
    obb_model, _ = _ensure_models()
    img = _read_image(await file.read())
    h, w = img.shape[:2]
    results = obb_model.predict(img, conf=conf, verbose=False)
    detections: list[Detection] = []
    if results and results[0].obb is not None:
        obb = results[0].obb
        polys = obb.xyxyxyxy.cpu().numpy() if hasattr(obb.xyxyxyxy, "cpu") else obb.xyxyxyxy
        confs = obb.conf.cpu().numpy() if hasattr(obb.conf, "cpu") else obb.conf
        clses = obb.cls.cpu().numpy() if hasattr(obb.cls, "cpu") else obb.cls
        for poly, cf, cls in zip(polys, confs, clses):
            cls_int = int(cls)
            detections.append(Detection(
                class_id=cls_int,
                class_name=CLASS_NAMES.get(cls_int, str(cls_int)),
                confidence=float(cf),
                polygon=[[float(x), float(y)] for x, y in poly],
                coco_iou=0.0,
            ))
    return PredictionResponse(
        image_size=[w, h],
        num_detections=len(detections),
        num_rejected=0,
        coco_containers_found=0,
        detections=detections,
    )


@app.get("/")
def root():
    return {
        "name": "Bottle vs Can Detection API",
        "version": "2.0.0",
        "architecture": "two-stage cascade (COCO filter + custom OBB)",
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/predict/visualize", "/predict/raw"],
    }
