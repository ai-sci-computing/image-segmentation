"""
scripts/make_patches.py

One-time preprocessing: turn the 36 source 5000x5000 TIFFs into 36 fixed
1024x1024 patches committed to patches/{train,val,test}/{images,masks}/.

See README.md section "Preprocessing" for the design rationale behind the
smart-center crop logic, the 1024x1024 single-resolution choice, and the
three-way train/val/test split.

Smart-center crop (the only non-obvious thing in this file):
  - Slide a 1024x1024 window on a stride-256 grid over the 5000x5000 source.
  - For each candidate (top, left), compute the positive-pixel ratio of the
    mask window.
  - Pick the position that maximizes `min(ratio, SOFT_CAP)`.
    - Windows with ratio below SOFT_CAP (0.30) are compared directly.
    - Windows with ratio above SOFT_CAP are all treated as "equally good" so
      the selection does not systematically prefer ultra-dense downtown over
      more representative mixed-density crops.
  - Tiebreak by Manhattan distance to the geometric center (closer wins)
    so that two tied candidates prefer the more centered one.
  - Fail loudly if no candidate reaches MIN_COVERAGE (0.05) — that means the
    source tile truly has no roof content worth training on, which we want
    to know about immediately rather than silently poison the dataset.

Outputs:
    patches/<split>/images/austin{N}.jpg   (JPEG q90 RGB)
    patches/<split>/masks/austin{N}.png    (PNG binary, values in {0, 255})

Usage:
    python scripts/make_patches.py --src data --dst patches
"""

import argparse
import sys
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np


# ============================================================================
#  CONFIG
# ============================================================================
# All tunable constants live here so they're easy to find later. Changing any
# of these requires regenerating patches/ and re-running all attempts.

# Target patch size. 1024 is SAM's native input size and the single resolution
# used across the whole pipeline — see README section "Preprocessing".
PATCH_SIZE = 1024

# Stride of the sliding-window search. 256 gives (5000 - 1024) / 256 + 1 = 16
# candidate positions per axis -> 256 total candidates per tile. Cheap enough
# to evaluate all of them without a more clever search.
STRIDE = 256

# Minimum acceptable positive-pixel ratio. Tiles whose best candidate window
# is below this floor cause the script to fail. 5% is enough for the
# segmentation model to get meaningful positive supervision from the sample.
MIN_COVERAGE = 0.05

# Soft cap on positive-pixel ratio during window selection. Windows with
# ratio > SOFT_CAP are treated as equivalent to SOFT_CAP so that the selection
# does not systematically prefer ultra-dense downtown over mixed-density
# crops. See README section "Preprocessing".
SOFT_CAP = 0.30

# Train / val / test split by source tile number. Fixed once and never
# revisited — see README section "Split".
SPLIT = {
    "train": {2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20,
              22, 23, 24, 25, 27, 28, 29, 30},
    "val":   {1, 6, 11, 16, 21, 26},
    "test":  {31, 32, 33, 34, 35, 36},
}

# JPEG quality for the RGB images. 90 is visually indistinguishable from
# the source for natural imagery while keeping each patch ~400 KB on disk.
JPEG_QUALITY = 90


# ============================================================================
#  TILE I/O
# ============================================================================
# Thin wrappers around cv2 so callers never have to think about flags or BGR
# conversion rules.

def read_image(path: Path) -> np.ndarray:
    """Load a source RGB TIFF as a uint8 H*W*3 array in BGR order.

    We keep the BGR convention cv2 gives us all the way through this script
    because cv2.imwrite also expects BGR — so we never have to convert, and
    the round-trip to disk is lossless. The training dataloader is where the
    BGR->RGB conversion happens.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"failed to read image: {path}")
    return img


def read_mask(path: Path) -> np.ndarray:
    """Load a binary mask TIFF as a uint8 H*W array with values in {0, 1}.

    Some exporters write masks as {0, 255}, others as {0, 1}. We normalize
    to a strict {0, 1} so that `.mean()` directly gives the positive-pixel
    ratio used by the smart-center selection rule.
    """
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise IOError(f"failed to read mask: {path}")
    # Collapse any multi-channel mask to single channel before thresholding
    # — just in case an odd export saved grayscale as 3-channel identical.
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 127).astype(np.uint8)


def write_image_jpg(path: Path, img_bgr: np.ndarray) -> None:
    """Save a BGR image as JPEG q90. Creates parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise IOError(f"failed to write image: {path}")


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    """Save a binary mask as PNG. Values in {0, 255} for visibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), (mask * 255).astype(np.uint8))
    if not ok:
        raise IOError(f"failed to write mask: {path}")


# ============================================================================
#  SMART-CENTER CROP
# ============================================================================

def candidate_positions(h: int, w: int, size: int, stride: int) -> Iterator[Tuple[int, int]]:
    """Yield (top, left) positions for every valid size*size crop on a grid.

    A candidate is valid iff its full size*size window fits inside h*w.
    """
    for top in range(0, h - size + 1, stride):
        for left in range(0, w - size + 1, stride):
            yield top, left


def smart_center_crop(
    mask: np.ndarray,
    size: int = PATCH_SIZE,
    stride: int = STRIDE,
    soft_cap: float = SOFT_CAP,
) -> Tuple[int, int, float]:
    """Find the best (top, left) crop position for this tile.

    Selection rule:
        score(top, left) = min(coverage(window), soft_cap)
    Highest score wins. Ties broken by Manhattan distance to image center
    (closer wins), so that two equally-good candidates prefer the more
    centered one.

    Args:
        mask: H*W uint8 array in {0, 1}. The selection uses the mask only;
            the image content is not consulted.
        size: edge length of the crop window.
        stride: sliding step on the candidate grid.
        soft_cap: the score ceiling — windows denser than this all score the
            same so that we don't systematically prefer ultra-dense downtown.

    Returns:
        (top, left, coverage) — top-left corner of the chosen window and
        the positive-pixel ratio (unsoftened) of that window.
    """
    h, w = mask.shape[:2]
    assert h >= size and w >= size, f"source too small: {h}x{w} for size={size}"

    cy, cx = h / 2, w / 2   # geometric center, used only as a tiebreaker

    best_score = -1.0
    best_pos = (0, 0)
    best_coverage = 0.0
    best_dist = float("inf")

    for top, left in candidate_positions(h, w, size, stride):
        window = mask[top:top + size, left:left + size]
        # mask is {0, 1}, so mean() == positive-pixel ratio — cheap
        coverage = float(window.mean())
        score = min(coverage, soft_cap)

        wy, wx = top + size / 2, left + size / 2
        dist = abs(wy - cy) + abs(wx - cx)

        # Strictly better score wins; on a tie, prefer closer to center.
        if (score > best_score) or (score == best_score and dist < best_dist):
            best_score = score
            best_pos = (top, left)
            best_coverage = coverage
            best_dist = dist

    return best_pos[0], best_pos[1], best_coverage


def crop(array: np.ndarray, top: int, left: int, size: int = PATCH_SIZE) -> np.ndarray:
    """Return a size*size crop of `array` starting at (top, left)."""
    return array[top:top + size, left:left + size]


# ============================================================================
#  SPLIT ASSIGNMENT
# ============================================================================

def parse_tile_number(stem: str) -> int:
    """Extract the integer N from a filename stem like 'austin5' -> 5."""
    if not stem.startswith("austin"):
        raise ValueError(f"unexpected filename stem: {stem}")
    return int(stem[len("austin"):])


def split_for(n: int) -> str:
    """Return 'train' / 'val' / 'test' for tile number n, or raise."""
    for name, members in SPLIT.items():
        if n in members:
            return name
    raise ValueError(f"tile number {n} is not in any split")


# ============================================================================
#  MAIN
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate 1024x1024 training patches from 5000x5000 source TIFFs.",
    )
    parser.add_argument(
        "--src", type=Path, default=Path("data"),
        help="source directory containing imgs/ and msks/ subdirs (default: data)",
    )
    parser.add_argument(
        "--dst", type=Path, default=Path("patches"),
        help="output directory for patches/{split}/{images,masks}/ (default: patches)",
    )
    args = parser.parse_args()

    src_imgs = args.src / "imgs"
    src_msks = args.src / "msks"
    if not src_imgs.is_dir() or not src_msks.is_dir():
        print(f"error: expected {src_imgs} and {src_msks} to exist", file=sys.stderr)
        return 1

    # Collect source tiles sorted by tile number, not lexicographically
    # (austin10 would come before austin2 otherwise, which is confusing).
    tiles = sorted(
        src_imgs.glob("austin*.tif"),
        key=lambda p: parse_tile_number(p.stem),
    )
    print(f"found {len(tiles)} source images in {src_imgs}")

    report = []        # per-tile stats for the final summary
    failures = []      # tiles we could not process at all

    for img_path in tiles:
        stem = img_path.stem
        msk_path = src_msks / img_path.name
        if not msk_path.exists():
            failures.append((stem, "mask missing"))
            continue

        n = parse_tile_number(stem)
        split = split_for(n)

        image = read_image(img_path)
        mask = read_mask(msk_path)

        # Sanity: image and mask must cover the same spatial extent.
        if image.shape[:2] != mask.shape[:2]:
            failures.append((
                stem,
                f"shape mismatch {image.shape[:2]} vs {mask.shape[:2]}",
            ))
            continue

        top, left, coverage = smart_center_crop(mask)

        img_crop = crop(image, top, left)
        msk_crop = crop(mask, top, left)

        out_img = args.dst / split / "images" / f"{stem}.jpg"
        out_msk = args.dst / split / "masks" / f"{stem}.png"
        write_image_jpg(out_img, img_crop)
        write_mask_png(out_msk, msk_crop)

        report.append({
            "tile": stem,
            "split": split,
            "top": top,
            "left": left,
            "coverage": coverage,
        })

        marker = "OK " if coverage >= MIN_COVERAGE else "LOW"
        print(f"  [{marker}] {stem:12s} split={split:5s}  "
              f"pos=({top:4d},{left:4d})  coverage={coverage*100:5.2f}%")

    # Summary. We fail the whole run if anything is below the coverage floor
    # or could not be processed — better to know now than to train on garbage.
    n_ok = sum(1 for r in report if r["coverage"] >= MIN_COVERAGE)
    n_low = sum(1 for r in report if r["coverage"] < MIN_COVERAGE)
    print()
    print(f"processed: {len(report)}  ok: {n_ok}  below-floor: {n_low}")

    if failures:
        print(f"failed to process {len(failures)} tile(s):")
        for stem, reason in failures:
            print(f"  - {stem}: {reason}")
        return 2

    if n_low > 0:
        print(f"error: {n_low} tile(s) did not reach {MIN_COVERAGE*100:.0f}% coverage")
        return 3

    print(f"{n_ok}/{len(report)} patches passed coverage check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
