"""
Sample annotated rasmlarni vizualizatsiya qilish.
README va Q&A uchun obrazli misollar tayyorlaydi.

Ishlatish:
    python src/visualize_samples.py
"""
import os
import glob
import random
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "dataset_samples"
OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "bottle", 1: "can"}
COLORS = {0: (0, 200, 0), 1: (0, 100, 255)}  # BGR: bottle=yashil, can=apelsin


def draw_obb(img, label_path):
    """OBB labelni rasmga chizish. Format: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized)."""
    h, w = img.shape[:2]
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:9]))
            pts = np.array([
                [coords[0] * w, coords[1] * h],
                [coords[2] * w, coords[3] * h],
                [coords[4] * w, coords[5] * h],
                [coords[6] * w, coords[7] * h],
            ], dtype=np.int32)
            color = COLORS.get(cls, (255, 255, 255))
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=3)
            x, y = pts[0]
            cv2.putText(img, CLASS_NAMES.get(cls, str(cls)),
                        (x, max(y - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img


def main():
    random.seed(7)
    images = sorted(glob.glob(str(PROC / "train" / "images" / "*.jpg")))
    sample = random.sample(images, min(8, len(images)))
    for i, img_path in enumerate(sample, 1):
        lbl_path = (PROC / "train" / "labels" /
                    Path(img_path).with_suffix(".txt").name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = draw_obb(img, str(lbl_path))
        out_path = OUT / f"sample_{i:02d}.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"Saqlandi: {out_path.name}")
    print(f"\n{len(sample)} ta sample rasm tayyor: {OUT}")


if __name__ == "__main__":
    main()
