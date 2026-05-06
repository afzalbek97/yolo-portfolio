"""
Smoke test for the FastAPI server.

Boots a request against /health, /predict, /predict/visualize, and /predict/raw
on a local server, prints a compact summary, and saves the visualised PNG.

Run:
    1. Boot the server in another terminal:
       cd ~/Documents/yolo-portfolio
       source .venv/bin/activate
       uvicorn api.main:app --host 127.0.0.1 --port 8765

    2. Run this script:
       python src/test_api.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8765"

TEST_IMAGES_DIR = ROOT / "data" / "processed" / "test" / "images"
OUT_DIR = ROOT / "results" / "api_smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    # --- /health ---
    print("=== /health ===")
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"FAIL: server unreachable at {BASE}: {e}")
        print("Did you start uvicorn? See module docstring.")
        return 1
    health = r.json()
    print(json.dumps(health, indent=2))
    if not (health.get("obb_loaded") and health.get("coco_loaded")):
        print("FAIL: models not loaded")
        return 1
    print("OK")

    # Pick a test image
    candidates = sorted(TEST_IMAGES_DIR.glob("*.jpg"))[:3]
    if not candidates:
        print(f"FAIL: no test images in {TEST_IMAGES_DIR}")
        return 1

    for img_path in candidates:
        print(f"\n=== {img_path.name} ===")
        with open(img_path, "rb") as f:
            files = {"file": (img_path.name, f, "image/jpeg")}

            # /predict (cascade)
            r = requests.post(f"{BASE}/predict", files=files, timeout=30)
            r.raise_for_status()
            data = r.json()
            print(
                f"  /predict           : {data['num_detections']} kept, "
                f"{data['num_rejected']} rejected, "
                f"{data['coco_containers_found']} COCO containers"
            )
            for d in data["detections"]:
                print(f"    - {d['class_name']:6s} conf={d['confidence']:.2f} "
                      f"coco_iou={d['coco_iou']:.2f}")

        # /predict/raw (no cascade) — re-open file because requests consumed it
        with open(img_path, "rb") as f:
            files = {"file": (img_path.name, f, "image/jpeg")}
            r = requests.post(f"{BASE}/predict/raw", files=files, timeout=30)
            r.raise_for_status()
            raw = r.json()
            print(f"  /predict/raw       : {raw['num_detections']} raw OBB detections")

        # /predict/visualize → save PNG
        with open(img_path, "rb") as f:
            files = {"file": (img_path.name, f, "image/jpeg")}
            r = requests.post(f"{BASE}/predict/visualize", files=files, timeout=30)
            r.raise_for_status()
            out = OUT_DIR / f"{img_path.stem}_cascade.png"
            out.write_bytes(r.content)
            print(f"  /predict/visualize : saved -> {out.relative_to(ROOT)}")

    print(f"\nAll smoke tests passed. Visualisations in {OUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
