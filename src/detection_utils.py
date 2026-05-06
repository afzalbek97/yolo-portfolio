"""Reusable geometry helpers for OBB detection pipeline."""
from __future__ import annotations

import numpy as np


def polygon_to_aabb(poly: np.ndarray) -> tuple[float, float, float, float]:
    """4-point polygon (shape 4×2) → axis-aligned bbox (x1, y1, x2, y2)."""
    xs = poly[:, 0]
    ys = poly[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def iou_aabb(a: tuple, b: tuple) -> float:
    """Intersection-over-union for two axis-aligned bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
