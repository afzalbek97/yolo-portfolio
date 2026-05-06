"""
Dataset statistikasi va klass balansi grafigi.
README ga qo'yiladigan grafiklarni tayyorlaydi.
"""
import os
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


def count_split(split):
    counts = Counter()
    for lbl in glob.glob(str(PROC / split / "labels" / "*.txt")):
        with open(lbl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                counts[int(line.split()[0])] += 1
    return counts


def main():
    splits = ["train", "val", "test"]
    data = {s: count_split(s) for s in splits}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Per-split chart
    x = range(len(splits))
    bottles = [data[s].get(0, 0) for s in splits]
    cans = [data[s].get(1, 0) for s in splits]
    width = 0.35
    ax[0].bar([i - width/2 for i in x], bottles, width, label="bottle", color="#2ecc71")
    ax[0].bar([i + width/2 for i in x], cans, width, label="can", color="#e67e22")
    ax[0].set_xticks(list(x))
    ax[0].set_xticklabels(splits)
    ax[0].set_title("Klasslar bo'yicha annotation sonining taqsimoti")
    ax[0].set_ylabel("Annotation soni")
    ax[0].legend()
    for i, v in enumerate(bottles):
        ax[0].text(i - width/2, v + 2, str(v), ha="center", fontsize=9)
    for i, v in enumerate(cans):
        ax[0].text(i + width/2, v + 2, str(v), ha="center", fontsize=9)

    # Total pie
    total_bottle = sum(bottles)
    total_can = sum(cans)
    ax[1].pie([total_bottle, total_can],
              labels=[f"bottle ({total_bottle})", f"can ({total_can})"],
              colors=["#2ecc71", "#e67e22"],
              autopct="%1.1f%%",
              startangle=90)
    ax[1].set_title("Umumiy klass nisbati (imbalance)")

    plt.tight_layout()
    out_path = OUT / "class_distribution.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saqlandi: {out_path}")

    # Konsolda ham ko'rsatish
    print("\nXulosa:")
    for s in splits:
        b = data[s].get(0, 0)
        c = data[s].get(1, 0)
        total = b + c
        if total > 0:
            print(f"  {s:5s}: bottle={b}, can={c} ({100*b/total:.1f}% / {100*c/total:.1f}%)")
    print(f"\nUmumiy imbalance ratio: bottle/can = {total_bottle/max(total_can,1):.2f}")


if __name__ == "__main__":
    main()
