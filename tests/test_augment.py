"""
tests/test_augment.py — Unit tests for augmentation presets.

Verifies that each preset:
  - Preserves image and mask shapes
  - Keeps masks binary (no interpolation artifacts)
  - Handles edge cases: all-zero mask, all-positive mask
  - Is retrievable by name via get_preset()
"""

import numpy as np
import pytest

from src.augment import get_preset


# All named presets to test. 'none' is included because it's a valid
# preset name that returns an identity transform.
PRESET_NAMES = ["none", "light", "medium", "heavy", "extreme"]


class TestPresetFactory:
    """Tests for the get_preset() factory function."""

    @pytest.mark.parametrize("name", PRESET_NAMES)
    def test_returns_compose(self, name):
        """Each preset name should return a valid Albumentations Compose."""
        preset = get_preset(name)
        assert hasattr(preset, "__call__"), f"{name} preset is not callable"

    def test_case_insensitive(self):
        """Preset names should be case-insensitive."""
        p1 = get_preset("MEDIUM")
        p2 = get_preset("medium")
        # Both should be valid Composes (we can't compare them directly,
        # but they should both be callable without error).
        assert callable(p1) and callable(p2)

    def test_unknown_raises(self):
        """An unknown preset name should raise ValueError."""
        with pytest.raises(ValueError, match="unknown augmentation preset"):
            get_preset("nonexistent")


class TestPresetBehavior:
    """Tests that each preset preserves the dataset contract."""

    @pytest.fixture
    def sample_image(self):
        """A synthetic 1024x1024 RGB image with varied pixel values."""
        rng = np.random.default_rng(42)
        return rng.integers(0, 256, size=(1024, 1024, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_mask(self):
        """A synthetic 1024x1024 binary mask (~25% positive pixels)."""
        rng = np.random.default_rng(42)
        return (rng.random((1024, 1024)) < 0.25).astype(np.uint8)

    @pytest.mark.parametrize("name", PRESET_NAMES)
    def test_preserves_image_shape(self, name, sample_image, sample_mask):
        """Augmented image should retain (1024, 1024, 3) shape."""
        preset = get_preset(name)
        result = preset(image=sample_image, mask=sample_mask)
        assert result["image"].shape == (1024, 1024, 3), \
            f"{name}: image shape {result['image'].shape}"

    @pytest.mark.parametrize("name", PRESET_NAMES)
    def test_preserves_mask_shape(self, name, sample_image, sample_mask):
        """Augmented mask should retain (1024, 1024) shape."""
        preset = get_preset(name)
        result = preset(image=sample_image, mask=sample_mask)
        assert result["mask"].shape == (1024, 1024), \
            f"{name}: mask shape {result['mask'].shape}"

    @pytest.mark.parametrize("name", PRESET_NAMES)
    def test_mask_stays_binary(self, name, sample_image, sample_mask):
        """Mask must contain only 0 and 1 after augmentation.

        This catches interpolation bugs: if Albumentations applies bilinear
        interpolation to the mask (e.g. during rotation), the values would
        become floats between 0 and 1, which breaks the loss computation.
        """
        preset = get_preset(name)
        result = preset(image=sample_image, mask=sample_mask)
        unique = np.unique(result["mask"])
        assert set(unique.tolist()).issubset({0, 1}), \
            f"{name}: mask has non-binary values {unique.tolist()}"

    @pytest.mark.parametrize("name", PRESET_NAMES)
    def test_all_zero_mask_survives(self, name, sample_image):
        """An all-zero mask (no buildings) should remain all-zero.

        This can happen if a training patch has < 5% coverage and
        augmentation moves the few positive pixels out of frame.
        """
        zero_mask = np.zeros((1024, 1024), dtype=np.uint8)
        preset = get_preset(name)
        result = preset(image=sample_image, mask=zero_mask)
        assert result["mask"].sum() == 0, \
            f"{name}: all-zero mask gained positive pixels"

    @pytest.mark.parametrize("name", PRESET_NAMES)
    def test_all_positive_mask_stays_populated(self, name, sample_image):
        """An all-positive mask should remain mostly positive.

        CoarseDropout in EXTREME might mask out some pixels, but the mask
        should never become all-zero from an all-positive input.
        """
        ones_mask = np.ones((1024, 1024), dtype=np.uint8)
        preset = get_preset(name)
        result = preset(image=sample_image, mask=ones_mask)
        assert result["mask"].sum() > 0, \
            f"{name}: all-positive mask became empty"
