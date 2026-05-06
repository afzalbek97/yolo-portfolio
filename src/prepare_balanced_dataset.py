"""
Creates a properly class-balanced train/val/test split from the downloaded
Roboflow v7 dataset and writes it to data/v8_balanced/.

Run once before uploading to Colab:
    python src/prepare_balanced_dataset.py

Output layout:
    data/v8_balanced/
        train/images/  train/labels/
        valid/images/  valid/labels/
        test/images/   test/labels/
        data.yaml
"""
from __future__ import annotations

import random
import shutil
from collections import defaultdict
from pathlib import Path

random.seed(42)

ROOT    = Path(__file__).resolve().parent.parent
SRC     = ROOT / "data" / "v7"
DST     = ROOT / "data" / "v8_balanced"
SPLITS  = ["train", "valid", "test"]


def dominant_class(label_path: Path) -> int | None:
    counts: dict[int, int] = defaultdict(int)
    for line in label_path.read_text().strip().splitlines():
        if line:
            counts[int(line.split()[0])] += 1
    if not counts:
        return None
    return max(counts, key=counts.__getitem__)


def collect_all() -> dict[int, list[Path]]:
    by_class: dict[int, list[Path]] = defaultdict(list)
    for split in SPLITS:
        for img in (SRC / split / "images").glob("*.jpg"):
            lbl = SRC / split / "labels" / (img.stem + ".txt")
            if not lbl.exists():
                continue
            cls = dominant_class(lbl)
            if cls is not None:
                by_class[cls].append(img)
    return by_class


def copy_pair(img: Path, src_split: str, dst_split: str) -> None:
    lbl = SRC / src_split / "labels" / (img.stem + ".txt")
    # find actual source split
    for s in SPLITS:
        candidate = SRC / s / "images" / img.name
        if candidate.exists():
            img = candidate
            lbl = SRC / s / "labels" / (img.stem + ".txt")
            break
    (DST / dst_split / "images").mkdir(parents=True, exist_ok=True)
    (DST / dst_split / "labels").mkdir(parents=True, exist_ok=True)
    shutil.copy(img, DST / dst_split / "images" / img.name)
    if lbl.exists():
        shutil.copy(lbl, DST / dst_split / "labels" / lbl.name)


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)

    by_class = collect_all()
    print("Total images by class:")
    for cls, imgs in by_class.items():
        print(f"  class {cls}: {len(imgs)}")

    # Balance: use the smaller class count as cap, shuffle both
    n = min(len(v) for v in by_class.values())
    print(f"\nCapping each class to {n} images for balance")

    balanced: list[tuple[Path, int]] = []
    for cls, imgs in by_class.items():
        random.shuffle(imgs)
        for img in imgs[:n]:
            balanced.append((img, cls))

    random.shuffle(balanced)

    # 80 / 10 / 10 split
    total = len(balanced)
    n_test = max(50, int(total * 0.10))
    n_val  = max(50, int(total * 0.10))
    n_train = total - n_test - n_val

    train_set = balanced[:n_train]
    val_set   = balanced[n_train:n_train + n_val]
    test_set  = balanced[n_train + n_val:]

    for img, _ in train_set:
        copy_pair(img, "", "train")
    for img, _ in val_set:
        copy_pair(img, "", "valid")
    for img, _ in test_set:
        copy_pair(img, "", "test")

    # Write data.yaml
    yaml_text = (
        "path: /content/dataset\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "names:\n"
        "  0: bottle\n"
        "  1: can\n\n"
        "nc: 2\n"
    )
    (DST / "data.yaml").write_text(yaml_text)

    print(f"\nDone — saved to {DST}")
    print(f"  train: {len(train_set)}  val: {len(val_set)}  test: {len(test_set)}")

    # Quick class check
    from collections import Counter
    for split in ["train", "valid", "test"]:
        c: Counter = Counter()
        for lbl in (DST / split / "labels").glob("*.txt"):
            for line in lbl.read_text().strip().splitlines():
                if line:
                    c[int(line.split()[0])] += 1
        total = sum(c.values())
        print(f"  {split:5s}: bottle={c[0]} ({100*c[0]//max(total,1)}%)  "
              f"can={c[1]} ({100*c[1]//max(total,1)}%)")


if __name__ == "__main__":
    main()
