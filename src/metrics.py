"""
src/metrics.py — IoU and F1 metrics for binary segmentation.

All metrics operate on tensors of raw logits (pre-sigmoid). They apply
sigmoid + thresholding internally, matching the loss function's convention.

Two levels of granularity:
  - `iou_score` / `f1_score`: compute a single scalar over a batch. Used
    for per-step logging during training.
  - `per_tile_metrics`: compute per-tile IoU and F1 over a full dataset.
    Used for the structured per-epoch diagnostic output and for the final
    headline results table.
"""

from typing import Dict, List

import torch
from torch.utils.data import DataLoader


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Compute Intersection-over-Union (Jaccard index) for a batch.

    Args:
        pred:   (B, 1, H, W) float32 logits (pre-sigmoid).
        target: (B, 1, H, W) float32 in {0.0, 1.0}.
        threshold: sigmoid output above this is considered positive.
        eps: small constant to avoid division by zero.

    Returns:
        Scalar IoU averaged over the batch.
    """
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return ((intersection + eps) / (union + eps)).item()


def f1_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Compute F1 score (Dice coefficient) for a batch.

    F1 = 2 * precision * recall / (precision + recall), which is equivalent
    to the Dice coefficient: 2 * |A ∩ B| / (|A| + |B|).

    Args:
        pred:   (B, 1, H, W) float32 logits (pre-sigmoid).
        target: (B, 1, H, W) float32 in {0.0, 1.0}.
        threshold: sigmoid output above this is considered positive.
        eps: small constant to avoid division by zero.

    Returns:
        Scalar F1 averaged over the batch.
    """
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum()
    return ((2.0 * intersection + eps) / (pred_bin.sum() + target.sum() + eps)).item()


@torch.no_grad()
def per_tile_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, List]:
    """Compute per-tile IoU and F1 over an entire dataset.

    This is used for the structured per-epoch diagnostic output (see
    README § How to run) and for the final results table. Each tile gets
    its own IoU and F1, plus a mean across all tiles.

    Args:
        model: the segmentation model in eval mode.
        loader: DataLoader for the split to evaluate (val or test).
        device: torch device to run inference on.
        threshold: sigmoid threshold for binarization.
        eps: smoothing for the metric denominators.

    Returns:
        dict with keys:
          'tile_ious':  list of per-tile IoU values
          'tile_f1s':   list of per-tile F1 values
          'mean_iou':   mean IoU across tiles
          'mean_f1':    mean F1 across tiles
    """
    model.eval()
    tile_ious = []
    tile_f1s = []

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        preds = model(imgs)

        # Compute per-sample metrics (each sample is one tile).
        pred_bin = (torch.sigmoid(preds) > threshold).float()
        for i in range(pred_bin.shape[0]):
            p = pred_bin[i]
            t = masks[i]
            inter = (p * t).sum()
            union = p.sum() + t.sum() - inter
            tile_ious.append(((inter + eps) / (union + eps)).item())
            tile_f1s.append(((2.0 * inter + eps) / (p.sum() + t.sum() + eps)).item())

    mean_iou = sum(tile_ious) / max(len(tile_ious), 1)
    mean_f1 = sum(tile_f1s) / max(len(tile_f1s), 1)

    return {
        "tile_ious": tile_ious,
        "tile_f1s": tile_f1s,
        "mean_iou": mean_iou,
        "mean_f1": mean_f1,
    }
