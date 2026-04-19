"""
src/augment.py — Albumentations augmentation presets for aerial roof imagery.

Four preset ladders, ordered from mildest to most aggressive:

    LIGHT    — D4 dihedral group only (flips + 90° rotations). These are
               "free" for nadir aerial imagery because there is no preferred
               orientation — a roof seen from above looks the same rotated
               90° or flipped. This is the minimum any attempt should use.

    MEDIUM   — LIGHT + photometric and mild geometric augmentations. This
               is the default recipe, ported from an earlier Keras
               implementation. It has been validated on this specific
               dataset.

    HEAVY    — MEDIUM + shadow, gamma, CLAHE, and JPEG compression artifacts.
               These mimic real-world variation in aerial capture conditions
               (sun angle, atmospheric effects, lossy compression).

    EXTREME  — HEAVY + mild elastic transform, coarse dropout, and fog.
               Elastic transforms warp rigid building geometry, which may
               be unrealistic. CoarseDropout masks out random rectangular
               patches, which can occlude the roofs we're trying to segment.
               Expected to hurt more than it helps, included to confirm that.

GaussianBlur and GaussNoise are deliberately excluded from all presets
because the source imagery has very little blur variation — adding
artificial blur/noise would create unrealistic variance that doesn't match
any real deployment condition. This is a dataset-specific observation from
the reference code.

All presets operate on (image, mask) pairs jointly via Albumentations' mask
handling, so spatial transforms are applied consistently to both.

See README § Augmentation presets.
"""

import cv2
import albumentations as A


def get_preset(name: str) -> A.Compose:
    """Return an Albumentations Compose for the named preset.

    Args:
        name: one of 'light', 'medium', 'heavy', 'extreme', or 'none'.
            Case-insensitive.

    Returns:
        An A.Compose that accepts (image=..., mask=...) and returns the
        augmented pair.
    """
    name = name.lower()
    if name == "none":
        return A.Compose([])
    if name == "light":
        return _light()
    if name == "medium":
        return _medium()
    if name == "heavy":
        return _heavy()
    if name == "extreme":
        return _extreme()
    raise ValueError(
        f"unknown augmentation preset: {name!r}. "
        f"Choose from: none, light, medium, heavy, extreme"
    )


# ============================================================================
#  PRESET DEFINITIONS
# ============================================================================

def _light() -> A.Compose:
    """D4 dihedral group — the 8 symmetries of a square.

    Free for nadir aerial imagery: no preferred orientation, so every flip
    and 90° rotation is a valid view of the same scene. This is the strongest
    "free lunch" in aerial augmentation.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])


def _medium() -> A.Compose:
    """LIGHT + photometric/geometric augmentations from the reference code.

    This is the recipe that was validated on the Inria Austin data in the
    Keras reference implementation. The exact parameters (shift_limit=0.2,
    scale_limit=0.3, etc.) are preserved from the reference so that we can
    cross-check results.

    ShiftScaleRotate uses BORDER_REFLECT to avoid black borders from
    out-of-bounds pixels — black borders would be a learnable artifact
    that doesn't appear in real data.
    """
    return A.Compose([
        # --- D4 geometric (same as LIGHT) ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # --- mild affine (shift, scale, small-angle rotation) ---
        # rotate_limit=0.4 in the reference is in DEGREES (not radians),
        # so this is a very mild ±0.4° rotation. Combined with the 90°
        # rotations above, this gives coverage of almost all angles.
        A.ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.3,
            rotate_limit=0.4,
            border_mode=cv2.BORDER_REFLECT,
            p=0.7,
        ),

        # --- photometric (sun angle, camera, processing) ---
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.3,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=15,
            p=0.3,
        ),
    ])


def _heavy() -> A.Compose:
    """MEDIUM + shadow, gamma, CLAHE, JPEG artifacts.

    These augmentations mimic real variation in aerial capture:
      - RandomShadow: long shadows from sun angle at different times of day
      - RandomGamma: tone-curve variation across different camera/processing
      - CLAHE: contrast-limited adaptive histogram equalization, a common
        preprocessing step that varies across processing pipelines
      - ImageCompression: JPEG artifacts from lossy compression, which is
        common in real-world aerial imagery delivery
    """
    return A.Compose([
        # --- D4 geometric ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # --- mild affine ---
        A.ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.3,
            rotate_limit=0.4,
            border_mode=cv2.BORDER_REFLECT,
            p=0.7,
        ),

        # --- photometric (from MEDIUM) ---
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.3,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=15,
            p=0.3,
        ),

        # --- additional photometric (HEAVY-specific) ---
        A.RandomShadow(p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.2),
        A.ImageCompression(quality_lower=70, quality_upper=95, p=0.2),
    ])


def _extreme() -> A.Compose:
    """HEAVY + elastic transform, coarse dropout, fog.

    These are aggressive augmentations that may hurt more than they help:
      - ElasticTransform warps rigid building geometry into curves that
        don't exist in real aerial imagery. Kept very mild (alpha=30).
      - CoarseDropout masks out random rectangular patches, which can
        occlude the buildings we're trying to segment.
      - RandomFog simulates atmospheric conditions but at the cost of
        hiding building edges and color cues.

    This preset is included primarily to confirm that "more augmentation"
    is not always better — the hypothesis is that MEDIUM or HEAVY will
    outperform EXTREME on this dataset.
    """
    return A.Compose([
        # --- D4 geometric ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # --- mild affine ---
        A.ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.3,
            rotate_limit=0.4,
            border_mode=cv2.BORDER_REFLECT,
            p=0.7,
        ),

        # --- photometric ---
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.3,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=15,
            p=0.3,
        ),

        # --- HEAVY additions ---
        A.RandomShadow(p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.2),
        A.ImageCompression(quality_lower=70, quality_upper=95, p=0.2),

        # --- EXTREME additions ---
        A.ElasticTransform(
            alpha=30,
            sigma=5,
            border_mode=cv2.BORDER_REFLECT,
            p=0.2,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(32, 64),
            hole_width_range=(32, 64),
            p=0.2,
        ),
        A.RandomFog(
            fog_coef_range=(0.1, 0.3),
            p=0.15,
        ),
    ])
