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
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["bottle", "can"]


def count_split(split: str) -> Counter:
    counts: Counter = Counter()
    for lbl in glob.glob(str(PROC / split / "labels" / "*.txt")):
        with open(lbl) as f:
            for line in f:
                line = line.strip()
                if line:
                    counts[int(line.split()[0])] += 1
    return counts


def main() -> None:
    splits = ["train", "val", "test"]
    data = {s: count_split(s) for s in splits}

    bottles = [data[s].get(0, 0) for s in splits]
    cans = [data[s].get(1, 0) for s in splits]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Per-split bar chart
    x = range(len(splits))
    width = 0.35
    axes[0].bar([i - width / 2 for i in x], bottles, width, label="bottle", color="#2ecc71")
    axes[0].bar([i + width / 2 for i in x], cans, width, label="can", color="#e67e22")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(splits)
    axes[0].set_title("Annotation count per split by class")
    axes[0].set_ylabel("Number of annotations")
    axes[0].legend()
    for i, v in enumerate(bottles):
        axes[0].text(i - width / 2, v + 1, str(v), ha="center", fontsize=9)
    for i, v in enumerate(cans):
        axes[0].text(i + width / 2, v + 1, str(v), ha="center", fontsize=9)

    # Total pie chart
    total_bottle = sum(bottles)
    total_can = sum(cans)
    axes[1].pie(
        [total_bottle, total_can],
        labels=[f"bottle ({total_bottle})", f"can ({total_can})"],
        colors=["#2ecc71", "#e67e22"],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[1].set_title("Overall class ratio (imbalance)")

    plt.tight_layout()
    out_path = OUT / "class_distribution.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")

    print("\nSummary:")
    for s in splits:
        b = data[s].get(0, 0)
        c = data[s].get(1, 0)
        total = b + c
        if total > 0:
            print(f"  {s:5s}: bottle={b}, can={c}  ({100*b/total:.1f}% / {100*c/total:.1f}%)")
    total = total_bottle + total_can
    print(f"\n  Overall: bottle={total_bottle} ({100*total_bottle/total:.1f}%), "
          f"can={total_can} ({100*total_can/total:.1f}%)")
    print(f"  Imbalance ratio bottle:can = {total_bottle/max(total_can,1):.2f}:1")


if __name__ == "__main__":
    main()
