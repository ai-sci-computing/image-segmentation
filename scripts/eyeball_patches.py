"""
scripts/eyeball_patches.py

Manual visual check of the 36 patches produced by make_patches.py.

Renders a grid of (image + red mask overlay) pairs for every committed
patch, so that the operator can eyeball all of them in one pass and spot
any obvious preprocessing failures — e.g. a crop that landed on a park,
a river edge, or a badly mis-registered mask.

This is the only step in the pipeline that needs a human in the loop. Once
the patches are approved here, they are the dataset-of-record and the 5000x5000
source TIFFs under data/ can be removed.

Usage:
    python scripts/eyeball_patches.py                     # grid, train->val->test order
    python scripts/eyeball_patches.py --sort-by-coverage  # worst coverage first
"""

import argparse
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


PATCHES_ROOT = Path("patches")


def parse_tile_number(stem: str) -> int:
    """Extract the integer N from a filename stem like 'austin5' -> 5."""
    return int(stem[len("austin"):])


def load_patch(split: str, stem: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (image_rgb, mask_binary) for a patch already on disk.

    The on-disk JPEG is BGR (cv2 convention), so we convert to RGB here for
    matplotlib. Masks on disk are {0, 255} PNGs; we collapse to {0, 1}.
    """
    img_path = PATCHES_ROOT / split / "images" / f"{stem}.jpg"
    msk_path = PATCHES_ROOT / split / "masks" / f"{stem}.png"

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"missing image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    msk = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
    if msk is None:
        raise IOError(f"missing mask: {msk_path}")
    msk = (msk > 127).astype(np.uint8)

    return img, msk


def iter_all_patches() -> Iterator[Tuple[str, str]]:
    """Yield (split, stem) for every committed patch.

    Order is train -> val -> test, and within each split by tile number,
    which keeps the grid output easy to compare against the source list.
    """
    for split in ("train", "val", "test"):
        images_dir = PATCHES_ROOT / split / "images"
        if not images_dir.is_dir():
            continue
        for p in sorted(images_dir.glob("austin*.jpg"),
                        key=lambda path: parse_tile_number(path.stem)):
            yield split, p.stem


def overlay(img: np.ndarray, msk: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a solid red mask overlay onto the RGB image for visualization.

    Args:
        img: H*W*3 uint8 RGB array.
        msk: H*W uint8 binary array in {0, 1}.
        alpha: blending strength on positive pixels (0 = no overlay,
            1 = mask completely replaces image).
    """
    red = np.zeros_like(img)
    red[..., 0] = 255  # red channel only

    m = msk.astype(bool)
    blended = img.astype(np.float32)
    blended[m] = (1 - alpha) * blended[m] + alpha * red[m]
    return np.clip(blended, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--sort-by-coverage", action="store_true",
        help="sort the grid by positive-pixel ratio, worst first",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="save the grid to a file (e.g. --save grid.png) instead of showing it",
    )
    args = parser.parse_args()

    patches = list(iter_all_patches())
    if not patches:
        print(f"error: no patches found under {PATCHES_ROOT}/")
        return 1

    # Load everything up front — 36 * ~400 KB is cheap and lets us sort.
    loaded = []
    for split, stem in patches:
        img, msk = load_patch(split, stem)
        coverage = float(msk.mean())
        loaded.append((split, stem, img, msk, coverage))

    if args.sort_by_coverage:
        loaded.sort(key=lambda t: t[4])

    # Grid layout: 6 columns gives 36 patches in exactly 6 rows.
    n = len(loaded)
    cols = 6
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for ax, (split, stem, img, msk, cov) in zip(axes, loaded):
        ax.imshow(overlay(img, msk))
        ax.set_title(f"{stem} ({split})\ncoverage={cov * 100:.1f}%", fontsize=8)
        ax.axis("off")

    # Hide any unused axes in the final row.
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved grid to {args.save}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
