"""
src/data.py — Dataset and oversampling sampler for roof segmentation.

This module provides two things:

1. `RoofPatchDataset` — a PyTorch Dataset that reads the committed 1024x1024
   patches from patches/{split}/{images,masks}/ and applies an Albumentations
   transform on each fetch.

2. `OversamplingSampler` — a PyTorch Sampler that draws `samples_per_epoch`
   indices with replacement from the dataset. This is the critical trick for
   tiny datasets: with only 24 training patches, a naive epoch is 6 optimizer
   steps (at batch 4), which is too small for LR schedules and early stopping
   to work sensibly. By oversampling to e.g. 400 draws per epoch, each with
   a fresh augmentation roll, we restore normal training dynamics while still
   drawing from only 24 unique scenes.

   This pattern is ported from an earlier Keras SegmentationSequence
   implementation that used the same with-replacement oversampling approach.

3. `make_dataloaders` — a factory that wires dataset + sampler + DataLoader
   for train, val, and optionally test splits.

All patches are stored as JPEG (images) and PNG (masks) at 1024x1024.
No runtime resizing is performed — the model eats the same bytes that are
on disk. See README § Preprocessing.
"""

from pathlib import Path
from typing import Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


# ============================================================================
#  DATASET
# ============================================================================

class RoofPatchDataset(Dataset):
    """Reads committed 1024x1024 patches and applies augmentation on-the-fly.

    Each __getitem__ call returns a (image, mask) pair where:
      - image is a float32 tensor of shape (3, H, W) in [0, 1], RGB order
      - mask  is a float32 tensor of shape (1, H, W) in {0.0, 1.0}

    Args:
        root: path to patches/<split>/ containing images/ and masks/ subdirs.
        transform: an Albumentations Compose applied to (image, mask) jointly.
            If None, no augmentation is applied (appropriate for val/test).
    """

    def __init__(self, root: Path, transform: Optional[A.Compose] = None):
        self.root = Path(root)
        self.transform = transform

        images_dir = self.root / "images"
        masks_dir = self.root / "masks"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"images dir not found: {images_dir}")
        if not masks_dir.is_dir():
            raise FileNotFoundError(f"masks dir not found: {masks_dir}")

        # Collect (image_path, mask_path) pairs sorted by tile number so
        # iteration order is deterministic and easy to cross-reference.
        self.samples = []
        for img_path in sorted(
            images_dir.glob("austin*.jpg"),
            key=lambda p: int(p.stem[len("austin"):]),
        ):
            stem = img_path.stem
            msk_path = masks_dir / f"{stem}.png"
            if not msk_path.exists():
                raise FileNotFoundError(f"mask missing for {stem}: {msk_path}")
            self.samples.append((img_path, msk_path))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"no patches found in {images_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, msk_path = self.samples[idx]

        # Load image as RGB uint8 (cv2 reads BGR, so convert).
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask as single-channel uint8 in {0, 1}.
        msk = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        msk = (msk > 127).astype(np.uint8)

        # Albumentations expects (H, W, C) numpy arrays for both image and
        # mask. The mask must be 2D (H, W) — Albumentations handles the
        # spatial transforms jointly so that flips/rotations are consistent.
        if self.transform is not None:
            augmented = self.transform(image=img, mask=msk)
            img = augmented["image"]
            msk = augmented["mask"]

        # Convert to float32 tensors in the shapes PyTorch expects:
        #   image: (3, H, W) in [0, 1]
        #   mask:  (1, H, W) in {0.0, 1.0}
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        msk = torch.from_numpy(msk).unsqueeze(0).float()

        return img, msk

    @property
    def stems(self):
        """Return the list of tile stems (e.g. ['austin2', 'austin3', ...])."""
        return [p.stem for p, _ in self.samples]


# ============================================================================
#  OVERSAMPLING SAMPLER
# ============================================================================

class OversamplingSampler(Sampler):
    """Draws `samples_per_epoch` indices with replacement from a dataset.

    Why this exists (critical for tiny datasets):
      With 24 training patches and batch_size=4, a naive epoch is only 6
      optimizer steps. LR schedulers, early stopping, and ReduceLROnPlateau
      all operate in epoch units — 6 steps per "epoch" makes their patience
      counters fire too quickly and their step sizes too coarse.

      By oversampling to e.g. 400 draws per epoch, we get ~100 steps per
      epoch, which restores normal training dynamics. Each draw gets a fresh
      augmentation roll from the dataset's transform, so the model sees 400
      genuinely distinct augmented views per epoch despite having only 24
      source patches.

    This is the PyTorch equivalent of a Keras SegmentationSequence pattern
    that draws random indices with replacement each epoch.

    Args:
        dataset: the dataset to sample from.
        samples_per_epoch: how many indices to yield per iteration (epoch).
        generator: optional torch.Generator for reproducibility.
    """

    def __init__(
        self,
        dataset: Dataset,
        samples_per_epoch: int = 400,
        generator: Optional[torch.Generator] = None,
    ):
        self.n = len(dataset)
        self.samples_per_epoch = samples_per_epoch
        self.generator = generator

    def __iter__(self):
        # Draw indices with replacement — each epoch is a fresh random
        # selection of source patches, and each will be independently
        # augmented by the dataset's transform.
        indices = torch.randint(
            0, self.n, (self.samples_per_epoch,), generator=self.generator,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.samples_per_epoch


# ============================================================================
#  DATALOADER FACTORY
# ============================================================================

def make_dataloaders(
    patches_root: Path,
    train_transform: Optional[A.Compose] = None,
    batch_size: int = 4,
    samples_per_epoch: int = 400,
    num_workers: int = 2,
    seed: int = 42,
) -> dict:
    """Build train, val, and test DataLoaders from the committed patches.

    Args:
        patches_root: path to the patches/ directory containing train/,
            val/, and test/ subdirs.
        train_transform: Albumentations Compose for training augmentation.
            Val and test always use no augmentation.
        batch_size: batch size for all loaders.
        samples_per_epoch: how many oversampled draws per training epoch.
        num_workers: DataLoader worker processes (0 for main-thread loading,
            useful for debugging; 2+ for Colab GPU training).
        seed: random seed for the oversampling sampler.

    Returns:
        dict with keys 'train', 'val', 'test', each mapping to a DataLoader.
        'test' is only included if patches_root/test/ exists.
    """
    patches_root = Path(patches_root)

    # Training: augmented + oversampled
    train_ds = RoofPatchDataset(patches_root / "train", transform=train_transform)
    train_gen = torch.Generator().manual_seed(seed)
    train_sampler = OversamplingSampler(
        train_ds, samples_per_epoch=samples_per_epoch, generator=train_gen,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Validation: no augmentation, no oversampling, sequential
    val_ds = RoofPatchDataset(patches_root / "val", transform=None)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    loaders = {"train": train_loader, "val": val_loader}

    # Test: same as val — no augmentation, sequential. Only included if the
    # test directory exists (it should, but defensive coding for smoke tests
    # on partial data).
    test_dir = patches_root / "test"
    if test_dir.is_dir() and any(test_dir.glob("images/austin*.jpg")):
        test_ds = RoofPatchDataset(test_dir, transform=None)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders["test"] = test_loader

    return loaders
