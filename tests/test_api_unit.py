"""
Unit tests for geometry helper functions (no model loading, no GPU required).
Run: pytest tests/ -v
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detection_utils import polygon_to_aabb, iou_aabb


class TestPolygonToAabb:
    def test_axis_aligned_rectangle(self):
        poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        assert polygon_to_aabb(poly) == (0.0, 0.0, 10.0, 10.0)

    def test_rotated_diamond(self):
        # Diamond: top (5,0), right (10,5), bottom (5,10), left (0,5)
        poly = np.array([[5, 0], [10, 5], [5, 10], [0, 5]], dtype=float)
        x1, y1, x2, y2 = polygon_to_aabb(poly)
        assert x1 == 0.0 and y1 == 0.0 and x2 == 10.0 and y2 == 10.0

    def test_negative_coords(self):
        poly = np.array([[-5, -3], [5, -3], [5, 3], [-5, 3]], dtype=float)
        assert polygon_to_aabb(poly) == (-5.0, -3.0, 5.0, 3.0)

    def test_returns_floats(self):
        poly = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
        result = polygon_to_aabb(poly)
        assert all(isinstance(v, float) for v in result)


class TestIouAabb:
    def test_perfect_overlap(self):
        box = (0.0, 0.0, 10.0, 10.0)
        assert iou_aabb(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = (0.0, 0.0, 5.0, 5.0)
        b = (6.0, 6.0, 10.0, 10.0)
        assert iou_aabb(a, b) == 0.0

    def test_half_overlap(self):
        a = (0.0, 0.0, 10.0, 10.0)  # area=100
        b = (5.0, 0.0, 15.0, 10.0)  # area=100, inter=50
        assert iou_aabb(a, b) == pytest.approx(50 / 150)

    def test_contained_box(self):
        outer = (0.0, 0.0, 10.0, 10.0)  # area=100
        inner = (2.0, 2.0, 8.0, 8.0)    # area=36, inter=36
        assert iou_aabb(outer, inner) == pytest.approx(36 / 100)

    def test_touching_edge_is_zero(self):
        a = (0.0, 0.0, 5.0, 5.0)
        b = (5.0, 0.0, 10.0, 5.0)  # shares only an edge
        assert iou_aabb(a, b) == 0.0

    def test_symmetry(self):
        a = (1.0, 2.0, 7.0, 8.0)
        b = (4.0, 5.0, 10.0, 11.0)
        assert iou_aabb(a, b) == pytest.approx(iou_aabb(b, a))

    def test_iou_gate_threshold(self):
        # Simulate the cascade gate: OBB box with 30% overlap passes, 10% is rejected
        obb = (0.0, 0.0, 10.0, 10.0)
        coco_70pct = (3.0, 0.0, 13.0, 10.0)  # ~23% IoU
        coco_small = (9.0, 0.0, 19.0, 10.0)  # ~5% IoU

        assert iou_aabb(obb, coco_70pct) >= 0.20  # should pass the gate
        assert iou_aabb(obb, coco_small) < 0.20   # should be rejected
