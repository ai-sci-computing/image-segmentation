"""
src/models/unet_scratch.py — Attempt #1: U-Net from scratch (no pretraining).

This is the baseline floor of the comparison. Using smp.Unet with
encoder_weights=None gives a clean U-Net with randomly initialized
ResNet34 encoder — no ImageNet prior, no pretrained features.

The model exists to answer: "how far can you get on 24 training samples
with a standard architecture but zero prior knowledge?" The answer sets
the floor that all pretrained approaches must beat.

Input:  (B, 3, 1024, 1024) float32 RGB
Output: (B, 1, 1024, 1024) float32 logits (pre-sigmoid)
"""

import segmentation_models_pytorch as smp
import torch.nn as nn


def make_unet_scratch(encoder_name: str = "resnet34") -> nn.Module:
    """Create a U-Net with a randomly initialized encoder (no pretraining).

    Args:
        encoder_name: which ResNet variant to use as the encoder backbone.
            Default 'resnet34' matches the pretrained attempts (#2, #3) so
            the only variable between #1 and #2 is the pretrained weights.

    Returns:
        An smp.Unet with classes=1 and no activation (raw logits).
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,  # no ImageNet pretraining — the whole point
        in_channels=3,
        classes=1,
        activation=None,  # raw logits; sigmoid applied in loss/metrics
    )
