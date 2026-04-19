"""
src/predict.py — Generate and save predictions from a trained model.

After training, this module loads the best checkpoint and runs inference
on the val and test splits. For each tile it saves:
  - The predicted binary mask as PNG
  - An overlay image (original + red mask) for visual comparison

The output goes to results/<attempt>/predictions/{val,test}/ so that
different attempts can be compared side by side.

Can be called automatically at the end of train() or standalone:
    python -m src.predict --attempt 02 --checkpoint results/02/best.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.data import RoofPatchDataset


def generate_predictions(
    model: nn.Module,
    device: torch.device,
    patches_root: Path,
    output_dir: Path,
    threshold: float = 0.5,
) -> None:
    """Run inference on val and test splits, save masks and overlays.

    For each tile, saves:
      - predictions/{split}/masks/{stem}.png     — binary mask (0/255)
      - predictions/{split}/overlays/{stem}.png  — original image with
        red overlay on predicted roof areas

    Args:
        model: trained model in eval mode.
        device: torch device for inference.
        patches_root: path to the patches/ directory.
        output_dir: base results directory (e.g. results/02/).
        threshold: sigmoid threshold for binarization.
    """
    model.eval()

    for split in ("val", "test"):
        split_dir = patches_root / split
        if not split_dir.is_dir():
            continue

        ds = RoofPatchDataset(split_dir, transform=None)
        pred_masks_dir = output_dir / "predictions" / split / "masks"
        pred_overlay_dir = output_dir / "predictions" / split / "overlays"
        pred_masks_dir.mkdir(parents=True, exist_ok=True)
        pred_overlay_dir.mkdir(parents=True, exist_ok=True)

        print(f"  generating predictions for {split} ({len(ds)} tiles)...")

        for idx in range(len(ds)):
            stem = ds.stems[idx]
            img_tensor, mask_tensor = ds[idx]

            # Run inference
            with torch.no_grad():
                pred_logits = model(img_tensor.unsqueeze(0).to(device))
                pred_mask = (torch.sigmoid(pred_logits) > threshold).float()
                pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)

            # Save predicted mask as PNG (0/255 for visibility)
            cv2.imwrite(
                str(pred_masks_dir / f"{stem}.png"),
                pred_mask * 255,
            )

            # Create overlay: original image + red on predicted roof pixels
            # img_tensor is (3, H, W) float [0,1] RGB — convert back to
            # uint8 BGR for cv2
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            overlay = img_bgr.copy()
            mask_bool = pred_mask.astype(bool)
            overlay[mask_bool, 2] = 255   # red channel (BGR: index 2)
            overlay[mask_bool, 1] = 0
            overlay[mask_bool, 0] = 0
            blended = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)

            cv2.imwrite(
                str(pred_overlay_dir / f"{stem}.png"),
                blended,
            )

        print(f"  saved {len(ds)} predictions to {output_dir / 'predictions' / split}/")


def load_and_predict(
    checkpoint_path: Path,
    model_factory,
    device: torch.device,
    patches_root: Path = Path("patches"),
    output_dir: Optional[Path] = None,
) -> None:
    """Load a checkpoint and generate predictions.

    Args:
        checkpoint_path: path to best.pt.
        model_factory: callable returning the model architecture.
        device: torch device.
        patches_root: path to patches/.
        output_dir: where to save predictions. Defaults to the
            checkpoint's parent directory.
    """
    output_dir = output_dir or checkpoint_path.parent

    print(f"loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"  epoch: {ckpt['epoch']}, val_iou: {ckpt['val_iou']:.4f}")

    model = model_factory()
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    generate_predictions(model, device, patches_root, output_dir)


# ============================================================================
#  CLI ENTRY POINT
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate predictions from a trained checkpoint.",
    )
    parser.add_argument("--attempt", type=str, required=True, help="attempt number (01-05)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="path to best.pt")
    parser.add_argument("--patches", type=Path, default=Path("patches"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from src.config import get_config
    config = get_config(args.attempt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_and_predict(
        args.checkpoint, config.model_factory, device,
        args.patches, args.output or args.checkpoint.parent,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
