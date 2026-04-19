"""
tests/test_dataset.py — Unit tests for RoofPatchDataset and OversamplingSampler.

These tests run on the committed patches/ directory (real data, not
synthetic), so they also serve as a smoke test that the preprocessing
output is still well-formed.
"""

import torch
from pathlib import Path

from src.data import RoofPatchDataset, OversamplingSampler, make_dataloaders
from src.augment import get_preset

PATCHES_ROOT = Path("patches")


class TestRoofPatchDataset:
    """Tests for the dataset's __getitem__ contract."""

    def test_train_length(self):
        """Train split should have exactly 24 patches."""
        ds = RoofPatchDataset(PATCHES_ROOT / "train")
        assert len(ds) == 24

    def test_val_length(self):
        """Val split should have exactly 6 patches."""
        ds = RoofPatchDataset(PATCHES_ROOT / "val")
        assert len(ds) == 6

    def test_test_length(self):
        """Test split should have exactly 6 patches."""
        ds = RoofPatchDataset(PATCHES_ROOT / "test")
        assert len(ds) == 6

    def test_getitem_shapes(self):
        """Each sample should be (3, 1024, 1024) image + (1, 1024, 1024) mask."""
        ds = RoofPatchDataset(PATCHES_ROOT / "train")
        img, msk = ds[0]
        assert img.shape == (3, 1024, 1024), f"image shape: {img.shape}"
        assert msk.shape == (1, 1024, 1024), f"mask shape: {msk.shape}"

    def test_getitem_dtypes(self):
        """Image and mask should both be float32."""
        ds = RoofPatchDataset(PATCHES_ROOT / "train")
        img, msk = ds[0]
        assert img.dtype == torch.float32
        assert msk.dtype == torch.float32

    def test_image_range(self):
        """Image values should be in [0, 1] after normalization."""
        ds = RoofPatchDataset(PATCHES_ROOT / "train")
        img, _ = ds[0]
        assert img.min() >= 0.0, f"image min: {img.min()}"
        assert img.max() <= 1.0, f"image max: {img.max()}"

    def test_mask_is_binary(self):
        """Mask should only contain 0.0 and 1.0 values."""
        ds = RoofPatchDataset(PATCHES_ROOT / "train")
        _, msk = ds[0]
        unique = torch.unique(msk)
        assert all(v in (0.0, 1.0) for v in unique.tolist()), \
            f"mask has non-binary values: {unique.tolist()}"

    def test_mask_has_positive_pixels(self):
        """At least the first train sample should have some positive pixels.

        The preprocessing script guarantees >= 5% coverage, so any sample
        should have some positive pixels.
        """
        ds = RoofPatchDataset(PATCHES_ROOT / "train")
        _, msk = ds[0]
        assert msk.sum() > 0, "mask is all-zero despite coverage guarantee"

    def test_no_nans(self):
        """Image and mask should never contain NaN."""
        ds = RoofPatchDataset(PATCHES_ROOT / "train")
        img, msk = ds[0]
        assert not torch.isnan(img).any(), "image contains NaN"
        assert not torch.isnan(msk).any(), "mask contains NaN"

    def test_stems_property(self):
        """The stems property should return tile names in sorted order."""
        ds = RoofPatchDataset(PATCHES_ROOT / "val")
        stems = ds.stems
        assert len(stems) == 6
        assert stems[0] == "austin1"

    def test_with_augmentation_preserves_contract(self):
        """Augmented samples should still satisfy the shape/range contract."""
        transform = get_preset("medium")
        ds = RoofPatchDataset(PATCHES_ROOT / "train", transform=transform)
        img, msk = ds[0]
        assert img.shape == (3, 1024, 1024)
        assert msk.shape == (1, 1024, 1024)
        assert img.min() >= 0.0 and img.max() <= 1.0
        # Mask must remain binary after augmentation — Albumentations
        # should only apply spatial transforms (which preserve binary values)
        # and never interpolate the mask.
        unique = torch.unique(msk)
        assert all(v in (0.0, 1.0) for v in unique.tolist()), \
            f"augmentation broke mask binarity: {unique.tolist()}"


class TestOversamplingSampler:
    """Tests for the oversampling sampler's index-generation contract."""

    def test_length(self):
        """Sampler should report exactly samples_per_epoch as its length."""
        ds = RoofPatchDataset(PATCHES_ROOT / "train")
        sampler = OversamplingSampler(ds, samples_per_epoch=400)
        assert len(sampler) == 400

    def test_yields_correct_count(self):
        """Iterating should yield exactly samples_per_epoch indices."""
        ds = RoofPatchDataset(PATCHES_ROOT / "train")
        sampler = OversamplingSampler(ds, samples_per_epoch=100)
        indices = list(sampler)
        assert len(indices) == 100

    def test_indices_in_range(self):
        """All yielded indices should be valid dataset indices."""
        ds = RoofPatchDataset(PATCHES_ROOT / "train")
        sampler = OversamplingSampler(ds, samples_per_epoch=200)
        indices = list(sampler)
        assert all(0 <= i < len(ds) for i in indices), \
            "sampler produced out-of-range index"

    def test_reproducibility_with_seed(self):
        """Same seed should produce the same index sequence."""
        ds = RoofPatchDataset(PATCHES_ROOT / "train")
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        s1 = list(OversamplingSampler(ds, samples_per_epoch=50, generator=gen1))
        s2 = list(OversamplingSampler(ds, samples_per_epoch=50, generator=gen2))
        assert s1 == s2


class TestMakeDataloaders:
    """Tests for the dataloader factory."""

    def test_returns_train_and_val(self):
        """Factory should always return at least train and val loaders."""
        loaders = make_dataloaders(PATCHES_ROOT, batch_size=2, num_workers=0)
        assert "train" in loaders
        assert "val" in loaders

    def test_returns_test_when_present(self):
        """Factory should include test loader when test/ directory exists."""
        loaders = make_dataloaders(PATCHES_ROOT, batch_size=2, num_workers=0)
        assert "test" in loaders

    def test_train_batch_shape(self):
        """A train batch should have the expected shapes."""
        loaders = make_dataloaders(
            PATCHES_ROOT, batch_size=2, samples_per_epoch=4, num_workers=0,
        )
        imgs, msks = next(iter(loaders["train"]))
        assert imgs.shape == (2, 3, 1024, 1024)
        assert msks.shape == (2, 1, 1024, 1024)
