"""
src/losses.py — Combined BCE + Dice loss for binary segmentation.

Ported from an earlier reference implementation and generalized to accept
tunable weights instead of a hardcoded 0.3/0.7 split.

The combination of BCE and Dice is standard for binary segmentation because:
  - BCE provides stable per-pixel gradients even when the prediction is far
    from the target.
  - Dice directly optimizes the overlap metric (related to IoU/F1), which
    matters more for segmentation quality than pixel-wise accuracy.

Using both together avoids the pathological cases of either alone:
  - Pure BCE can get stuck optimizing easy background pixels at the expense
    of the smaller foreground class.
  - Pure Dice can produce unstable gradients when the prediction or target
    is nearly empty (the denominator approaches zero).

The loss operates on raw logits (pre-sigmoid) for numerical stability:
BCE uses BCEWithLogitsLoss internally, and Dice applies sigmoid explicitly.
"""

import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """Weighted combination of BCE and Dice loss for binary segmentation.

    Args:
        bce_weight: weight for the BCE component (default 0.5).
        dice_weight: weight for the Dice component (default 0.5).
        smooth: smoothing term added to Dice numerator and denominator to
            avoid division by zero when both prediction and target are empty.
            Default 1.0 (Laplace smoothing).

    Input/output contract:
        pred:   (B, 1, H, W) float32 logits (pre-sigmoid)
        target: (B, 1, H, W) float32 in {0.0, 1.0}
        returns: scalar loss
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # BCE on raw logits — numerically stable via log-sum-exp internally.
        bce_loss = self.bce(pred, target)

        # Dice on sigmoid-activated predictions.
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice_loss = 1.0 - (
            (2.0 * intersection + self.smooth)
            / (pred_sigmoid.sum() + target.sum() + self.smooth)
        )

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
