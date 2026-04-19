"""
scripts/plot_convergence.py — Training convergence curves for all 5 attempts.

Generates two plots:
  1. Val IoU vs. epoch (all 5 attempts overlaid)
  2. Train loss vs. epoch (all 5 attempts overlaid)

Usage:
    python scripts/plot_convergence.py --save roof-segmentation/convergence.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_ROOT = Path("roof-segmentation/results")
ATTEMPTS = {
    "01": "U-Net scratch",
    "02": "SMP U-Net + ResNet34",
    "03": "SMP DeepLabV3+",
    "04": "SAM frozen + conv dec",
    "05": "SAM + U-Net dec",
}
COLORS = {
    "01": "#888888",
    "02": "#2196F3",
    "03": "#1565C0",
    "04": "#FF7043",
    "05": "#E53935",
}


def load_history(attempt):
    path = RESULTS_ROOT / attempt / "history.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {
        "epoch": [int(r["epoch"]) for r in rows],
        "train_loss": [float(r["train_loss"]) for r in rows],
        "val_loss": [float(r["val_loss"]) for r in rows],
        "val_iou": [float(r["val_iou"]) for r in rows],
        "val_f1": [float(r["val_f1"]) for r in rows],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for attempt, name in ATTEMPTS.items():
        h = load_history(attempt)
        color = COLORS[attempt]
        ax1.plot(h["epoch"], h["val_iou"], label=name, color=color, linewidth=1.5)
        ax2.plot(h["epoch"], h["train_loss"], label=name, color=color, linewidth=1.5)

    # Mark the unfreeze point for CNN attempts #02 and #03
    ax1.axvline(x=21, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.text(22, 0.62, "encoder\nunfreeze", fontsize=7, color="#666666")
    ax2.axvline(x=21, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.text(22, 0.45, "encoder\nunfreeze", fontsize=7, color="#666666")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val IoU")
    ax1.set_title("Validation IoU")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_ylim(0.55, 0.85)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Train Loss")
    ax2.set_title("Training Loss")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
