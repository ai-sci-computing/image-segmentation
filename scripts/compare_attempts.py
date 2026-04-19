"""
scripts/compare_attempts.py — Visual comparison of all attempts side by side.

Generates a grid image where:
  - Each row is one tile (val or test)
  - Each column is one attempt's overlay prediction
  - First column is the ground truth overlay

Usage:
    python scripts/compare_attempts.py --split val --save comparison_val.png
    python scripts/compare_attempts.py --split test --save comparison_test.png
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


RESULTS_ROOT = Path("roof-segmentation/results")
PATCHES_ROOT = Path("patches")
ATTEMPTS = ["01", "02", "03", "04", "05"]
ATTEMPT_NAMES = {
    "01": "U-Net scratch",
    "02": "SMP U-Net\nResNet34",
    "03": "SMP DeepLab\nResNet34",
    "04": "SAM frozen\nconv decoder",
    "05": "SAM + U-Net\ndecoder",
}


def load_image_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    msk = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (msk > 127).astype(np.uint8)


def overlay(img, msk, color=(255, 0, 0), alpha=0.4):
    """Blend a colored mask overlay onto the RGB image."""
    blended = img.astype(np.float32).copy()
    m = msk.astype(bool)
    for c in range(3):
        blended[m, c] = (1 - alpha) * blended[m, c] + alpha * color[c]
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    split = args.split
    images_dir = PATCHES_ROOT / split / "images"
    masks_dir = PATCHES_ROOT / split / "masks"

    stems = sorted(
        [p.stem for p in images_dir.glob("austin*.jpg")],
        key=lambda s: int(s[len("austin"):]),
    )

    n_tiles = len(stems)
    n_cols = 1 + len(ATTEMPTS)  # ground truth + 5 attempts

    fig, axes = plt.subplots(n_tiles, n_cols, figsize=(n_cols * 3, n_tiles * 3))

    # Column headers
    headers = ["Ground Truth"] + [ATTEMPT_NAMES[a] for a in ATTEMPTS]
    for j, header in enumerate(headers):
        axes[0, j].set_title(header, fontsize=9, fontweight="bold")

    for i, stem in enumerate(stems):
        img = load_image_rgb(images_dir / f"{stem}.jpg")
        gt_mask = load_mask(masks_dir / f"{stem}.png")

        # Ground truth overlay (green)
        gt_overlay = overlay(img, gt_mask, color=(0, 200, 0), alpha=0.4)
        axes[i, 0].imshow(gt_overlay)
        axes[i, 0].set_ylabel(stem, fontsize=8, rotation=0, labelpad=50)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Attempt overlays (loaded from saved predictions)
        for j, attempt in enumerate(ATTEMPTS, start=1):
            overlay_path = RESULTS_ROOT / attempt / "predictions" / split / "overlays" / f"{stem}.png"
            if overlay_path.exists():
                pred_overlay = load_image_rgb(overlay_path)
                axes[i, j].imshow(pred_overlay)
            else:
                axes[i, j].text(0.5, 0.5, "N/A", ha="center", va="center",
                               transform=axes[i, j].transAxes, fontsize=14)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
