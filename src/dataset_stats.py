"""
Dataset statistics and class balance chart.

Generates results/class_distribution.png for the README.

Usage:
    python src/dataset_stats.py
"""
from __future__ import annotations

import glob
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "v7"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["bottle", "can"]

VAL_SPLIT_NAME = "valid"  # Roboflow v7 uses "valid" not "val"


def count_split(split: str) -> Counter:
    counts: Counter = Counter()
    label_dir = DATASET / split / "labels"
    for lbl in glob.glob(str(label_dir / "*.txt")):
        with open(lbl) as f:
            for line in f:
                line = line.strip()
                if line:
                    counts[int(line.split()[0])] += 1
    return counts


def main() -> None:
    splits = ["train", VAL_SPLIT_NAME, "test"]
    data = {s: count_split(s) for s in splits}

    bottles = [data[s].get(0, 0) for s in splits]
    cans = [data[s].get(1, 0) for s in splits]
    labels = ["train", "val", "test"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Per-split bar chart
    x = range(len(splits))
    width = 0.35
    axes[0].bar([i - width / 2 for i in x], bottles, width, label="bottle", color="#2ecc71")
    axes[0].bar([i + width / 2 for i in x], cans, width, label="can", color="#e67e22")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels)
    axes[0].set_title("Annotation count per split by class  (Roboflow v7)")
    axes[0].set_ylabel("Number of annotations")
    axes[0].legend()
    for i, v in enumerate(bottles):
        axes[0].text(i - width / 2, v + 5, str(v), ha="center", fontsize=9)
    for i, v in enumerate(cans):
        axes[0].text(i + width / 2, v + 5, str(v), ha="center", fontsize=9)

    # Total pie chart (train only — val is skewed by Roboflow augmentation strategy)
    total_bottle = data["train"].get(0, 0)
    total_can = data["train"].get(1, 0)
    axes[1].pie(
        [total_bottle, total_can],
        labels=[f"bottle ({total_bottle})", f"can ({total_can})"],
        colors=["#2ecc71", "#e67e22"],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[1].set_title("Train split class ratio  (well-balanced)")

    plt.tight_layout()
    out_path = OUT / "class_distribution.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")

    print("\nSummary:")
    for s, label in zip(splits, labels):
        b = data[s].get(0, 0)
        c = data[s].get(1, 0)
        total = b + c
        if total > 0:
            print(f"  {label:5s}: bottle={b}, can={c}  ({100*b/total:.1f}% / {100*c/total:.1f}%)")
    t_total = total_bottle + total_can
    print(f"\n  Train overall: bottle={total_bottle} ({100*total_bottle/t_total:.1f}%), "
          f"can={total_can} ({100*total_can/t_total:.1f}%)")


if __name__ == "__main__":
    main()
