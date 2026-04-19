"""
src/config.py — Per-attempt hyperparameter configurations.

Each attempt gets its own AttemptConfig instance that fully describes its
training recipe: model factory, learning rate, augmentation preset, dropout,
loss weighting, and training schedule. The config is the single source of
truth for what was run — it gets logged alongside the results so the
headline table in the README can be filled in directly from config + metrics.

Default values are starting points from the reference Keras/PyTorch code.
Per-model tuning (Phase 1/2 of the project plan) will override these —
the tuned configs replace these defaults before the final Phase 3 runs.

See README § Per-model tuning for what's tuned vs. locked.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch.nn as nn


@dataclass
class AttemptConfig:
    """Complete training recipe for one model attempt.

    Attributes:
        name: human-readable identifier (e.g. 'smp_unet_resnet34').
        model_factory: callable that returns an nn.Module. Called with
            no arguments — all model parameters are baked in.
        lr: initial learning rate for AdamW.
        augmentation_preset: name passed to augment.get_preset().
        decoder_dropout: dropout probability for decoder layers (SMP track).
            Ignored by SAM models which have their own dropout.
        bce_weight: weight for BCE in the combined loss.
        dice_weight: weight for Dice in the combined loss.
        epochs: maximum training epochs (early stopping may cut this short).
        batch_size: training batch size. May be lowered for SAM due to
            memory; effective batch preserved via gradient accumulation.
        samples_per_epoch: how many oversampled draws per training epoch.
        two_stage: whether to use the frozen→unfrozen training schedule.
        unfreeze_epoch: epoch at which to unfreeze the encoder (if two_stage).
        weight_decay: AdamW weight decay.
        patience_stop: early stopping patience (epochs without val IoU gain).
        patience_lr: ReduceLROnPlateau patience.
        lr_factor: ReduceLROnPlateau reduction factor.
        seed: random seed for reproducibility.
        patches_root: path to the patches/ directory.
    """
    name: str
    model_factory: Callable[[], nn.Module]
    lr: float = 1e-3
    augmentation_preset: str = "medium"
    decoder_dropout: float = 0.3
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    epochs: int = 60
    batch_size: int = 4
    samples_per_epoch: int = 400
    two_stage: bool = False
    unfreeze_epoch: int = 20
    weight_decay: float = 1e-4
    patience_stop: int = 10
    patience_lr: int = 3
    lr_factor: float = 0.5
    seed: int = 42
    patches_root: Path = field(default_factory=lambda: Path("patches"))


# ============================================================================
#  DEFAULT CONFIGS — starting points, to be refined during tuning phases
# ============================================================================
# Each config's model_factory is a lambda that defers the import so we
# don't load heavy models (SAM) at import time.

def _make_attempt_01() -> AttemptConfig:
    """Attempt #1: U-Net from scratch — baseline floor."""
    from src.models.unet_scratch import make_unet_scratch
    return AttemptConfig(
        name="unet_scratch",
        model_factory=make_unet_scratch,
        lr=1e-3,
        two_stage=False,
    )


def _make_attempt_02() -> AttemptConfig:
    """Attempt #2: SMP U-Net + ImageNet ResNet34 — transfer learning."""
    from src.models.smp_wrapper import make_smp_unet
    return AttemptConfig(
        name="smp_unet_resnet34",
        model_factory=lambda: make_smp_unet(decoder_dropout=0.3),
        lr=1e-3,
        two_stage=True,
        unfreeze_epoch=20,
    )


def _make_attempt_03() -> AttemptConfig:
    """Attempt #3: SMP DeepLabV3+ + ImageNet ResNet34 — architecture."""
    from src.models.smp_wrapper import make_smp_deeplabv3plus
    return AttemptConfig(
        name="smp_deeplabv3plus_resnet34",
        model_factory=lambda: make_smp_deeplabv3plus(decoder_dropout=0.3),
        lr=1e-3,
        two_stage=True,
        unfreeze_epoch=20,
    )


def _make_attempt_04() -> AttemptConfig:
    """Attempt #4: SAM ViT-B frozen + conv decoder — few-shot prior."""
    from src.models.sam_decoder import SAMFrozenConvDecoder
    return AttemptConfig(
        name="sam_frozen_conv",
        model_factory=SAMFrozenConvDecoder,
        lr=1e-4,
        batch_size=2,  # SAM ViT-B is memory-hungry at 1024x1024
        two_stage=False,  # encoder stays frozen throughout
    )


def _make_attempt_05() -> AttemptConfig:
    """Attempt #5: SAM ViT-B + U-Net decoder — richer decoder on SAM prior."""
    from src.models.sam_decoder import SAMUNetDecoder
    return AttemptConfig(
        name="sam_unet_decoder",
        model_factory=SAMUNetDecoder,
        lr=1e-4,
        batch_size=2,
        two_stage=False,
    )


# Registry mapping attempt number to config factory.
ATTEMPT_REGISTRY = {
    "01": _make_attempt_01,
    "02": _make_attempt_02,
    "03": _make_attempt_03,
    "04": _make_attempt_04,
    "05": _make_attempt_05,
}


def get_config(attempt: str) -> AttemptConfig:
    """Return the AttemptConfig for the given attempt number.

    Args:
        attempt: string like '01', '02', ..., '05'.

    Returns:
        A fresh AttemptConfig instance.
    """
    if attempt not in ATTEMPT_REGISTRY:
        raise ValueError(
            f"unknown attempt: {attempt!r}. "
            f"Choose from: {', '.join(sorted(ATTEMPT_REGISTRY))}"
        )
    return ATTEMPT_REGISTRY[attempt]()
