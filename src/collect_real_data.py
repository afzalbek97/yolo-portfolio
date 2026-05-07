"""
Collects real webcam frames for fine-tuning.

Usage:
    python src/collect_real_data.py --class bottle   # hold a bottle in front of camera
    python src/collect_real_data.py --class can      # hold a can in front of camera
    python src/collect_real_data.py --class background  # no objects (negative samples)

Keys:
    SPACE — save current frame
    q     — quit
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--class", dest="cls", required=True,
                   choices=["bottle", "can", "background"])
    p.add_argument("--source", default="0")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = ROOT / "data" / "real_webcam" / args.cls
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.jpg")))
    print(f"[INFO] Saving to: {out_dir}")
    print(f"[INFO] Already collected: {existing} frames")
    print(f"[INFO] Hold a {args.cls} in front of camera, press SPACE to save, Q to quit")

    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    n = existing
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        preview = frame.copy()
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(preview, f"Class: {args.cls}  Saved: {n}  |  SPACE=save  Q=quit",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(f"Collecting: {args.cls}", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            path = out_dir / f"{args.cls}_{n:04d}.jpg"
            cv2.imwrite(str(path), frame)
            n += 1
            print(f"[OK] Saved {path.name}  (total: {n})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Collected {n - existing} new frames, total: {n}")


if __name__ == "__main__":
    main()
