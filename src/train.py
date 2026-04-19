"""
src/train.py — Shared training loop for all 5 attempts.

Implements the training protocol described in README § Training protocol:
  - AdamW optimizer with configurable LR and weight decay
  - Two-stage frozen→unfrozen training for pretrained-encoder models
  - EarlyStopping on val IoU (patience from config)
  - ReduceLROnPlateau on val IoU (factor and patience from config)
  - Best checkpoint saved by val IoU
  - Structured per-epoch diagnostics for the Colab feedback loop

The --smoke flag overrides the config to run 2 epochs on a tiny subset
(samples_per_epoch=8, batch_size=2) for fast end-to-end verification.

Usage:
    # Full training (from a notebook or script)
    from src.config import get_config
    from src.train import train
    history = train(get_config('02'))

    # Smoke test (from command line)
    .venv/bin/python -m src.train --smoke --attempt 02
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.augment import get_preset
from src.config import AttemptConfig, get_config
from src.data import make_dataloaders
from src.diagnostics import format_epoch_log
from src.losses import BCEDiceLoss
from src.metrics import iou_score, f1_score, per_tile_metrics


def _get_device() -> torch.device:
    """Pick the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_gpu_memory() -> Optional[tuple]:
    """Return (used_GB, total_GB) for the current CUDA device, or None."""
    if not torch.cuda.is_available():
        return None
    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return used, total


def _save_history_csv(history: dict, path: Path) -> None:
    """Write the training history to a CSV file.

    Called after every epoch so that partial results survive Colab
    disconnects. Overwrites the file each time (not append) since
    the full history dict is always available in memory.
    """
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_iou", "val_f1",
                         "epoch_time_s", "total_time_s"])
        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                f"{history['train_loss'][i]:.6f}",
                f"{history['val_loss'][i]:.6f}",
                f"{history['val_iou'][i]:.6f}",
                f"{history['val_f1'][i]:.6f}",
                f"{history['epoch_time'][i]:.1f}" if i < len(history.get("epoch_time", [])) else "",
                f"{history['total_time'][i]:.1f}" if i < len(history.get("total_time", [])) else "",
            ])


def _freeze_encoder(model: nn.Module) -> None:
    """Freeze all parameters in the model's encoder (if it has one).

    Works for both SMP models (model.encoder) and SAM models
    (model.encoder). No-op if the model has no .encoder attribute.
    """
    if hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = False


def _unfreeze_encoder(model: nn.Module) -> None:
    """Unfreeze all parameters in the model's encoder.

    Called at the unfreeze_epoch in the two-stage training schedule.
    After unfreezing, the optimizer must be recreated to include the
    encoder parameters.
    """
    if hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = True


def train(
    config: AttemptConfig,
    smoke: bool = False,
    results_dir: Optional[Path] = None,
) -> Dict[str, List]:
    """Run the full training loop for one attempt.

    Args:
        config: the AttemptConfig describing the full recipe.
        smoke: if True, override config for a fast 2-epoch sanity check.
        results_dir: where to save checkpoints and history. Defaults to
            results/<config.name>/.

    Returns:
        dict with keys 'train_loss', 'val_loss', 'val_iou', 'val_f1',
        each a list of per-epoch values.
    """
    if smoke:
        config.epochs = 2
        config.samples_per_epoch = 8
        config.batch_size = 2
        config.patience_stop = 999  # don't early-stop in smoke mode

    device = _get_device()
    print(f"device: {device}")
    print(f"attempt: {config.name}")
    print(f"epochs: {config.epochs}, samples/epoch: {config.samples_per_epoch}, "
          f"batch: {config.batch_size}")

    # --- Data ---
    transform = get_preset(config.augmentation_preset)
    loaders = make_dataloaders(
        config.patches_root,
        train_transform=transform,
        batch_size=config.batch_size,
        samples_per_epoch=config.samples_per_epoch,
        num_workers=0 if smoke else 2,
        seed=config.seed,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Get tile names for per-tile diagnostics
    val_stems = loaders["val"].dataset.stems

    # --- Model ---
    model = config.model_factory()
    model = model.to(device)

    # For two-stage training, freeze the encoder initially.
    if config.two_stage:
        _freeze_encoder(model)
        print(f"encoder frozen for first {config.unfreeze_epoch} epochs")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"parameters: {trainable:,} trainable / {total:,} total")

    # --- Loss, optimizer, scheduler ---
    criterion = BCEDiceLoss(
        bce_weight=config.bce_weight,
        dice_weight=config.dice_weight,
    ).to(device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",         # maximize val IoU
        factor=config.lr_factor,
        patience=config.patience_lr,
        min_lr=1e-8,
    )

    # --- Results tracking ---
    results_dir = results_dir or Path("results") / config.name
    results_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_f1": [],
        "epoch_time": [],   # seconds per epoch
        "total_time": [],   # cumulative seconds since training started
    }
    best_val_iou = -1.0
    epochs_without_improvement = 0
    training_start = time.time()

    # --- Training loop ---
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        # Two-stage: unfreeze encoder at the configured epoch.
        # Recreate the optimizer to include the newly-unfrozen parameters.
        if config.two_stage and epoch == config.unfreeze_epoch + 1:
            _unfreeze_encoder(model)
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
            scheduler = ReduceLROnPlateau(
                optimizer, mode="max", factor=config.lr_factor,
                patience=config.patience_lr, min_lr=1e-8,
            )
            unfrozen_trainable = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            # Reset early stopping and best IoU tracking so the unfrozen
            # phase gets a fresh chance. Without this, patience accumulated
            # during the frozen plateau can trigger early stopping before
            # the unfrozen parameters have had any chance to improve.
            epochs_without_improvement = 0
            best_val_iou = -1.0

            print(f"\n--- epoch {epoch}: encoder unfrozen, "
                  f"trainable params: {unfrozen_trainable:,}, "
                  f"early stopping reset ---\n")

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1

        avg_train_loss = train_loss_sum / max(train_steps, 1)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_iou_sum = 0.0
        val_f1_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                preds = model(imgs)
                loss = criterion(preds, masks)

                val_loss_sum += loss.item()
                val_iou_sum += iou_score(preds, masks)
                val_f1_sum += f1_score(preds, masks)
                val_steps += 1

        avg_val_loss = val_loss_sum / max(val_steps, 1)
        avg_val_iou = val_iou_sum / max(val_steps, 1)
        avg_val_f1 = val_f1_sum / max(val_steps, 1)

        # Per-tile breakdown (computed every epoch — cheap with 6 tiles)
        tile_metrics = per_tile_metrics(model, val_loader, device)

        # Record history (including timing)
        epoch_time = time.time() - epoch_start
        total_time = time.time() - training_start
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_iou"].append(avg_val_iou)
        history["val_f1"].append(avg_val_f1)
        history["epoch_time"].append(epoch_time)
        history["total_time"].append(total_time)

        # LR scheduler step
        scheduler.step(avg_val_iou)
        current_lr = optimizer.param_groups[0]["lr"]

        # GPU memory info
        gpu_mem = _get_gpu_memory()

        step_time = (time.time() - epoch_start) / max(train_steps, 1)

        # --- Diagnostics ---
        log = format_epoch_log(
            attempt_name=config.name,
            epoch=epoch,
            total_epochs=config.epochs,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            val_iou=avg_val_iou,
            val_f1=avg_val_f1,
            per_tile_iou=tile_metrics["tile_ious"],
            tile_names=val_stems,
            train_samples=config.samples_per_epoch,
            batch_size=config.batch_size,
            n_val_tiles=len(val_stems),
            gpu_mem_used=gpu_mem[0] if gpu_mem else None,
            gpu_mem_total=gpu_mem[1] if gpu_mem else None,
            step_time=step_time,
            lr=current_lr,
        )
        print(log)

        # --- Save history CSV after every epoch ---
        # This ensures that even if Colab disconnects mid-run, we have
        # metrics for all completed epochs. The CSV is overwritten each
        # epoch (not appended) to keep it clean — it's small.
        _save_history_csv(history, results_dir / "history.csv")

        # --- Checkpoint ---
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            epochs_without_improvement = 0
            ckpt_path = results_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": avg_val_iou,
                "val_f1": avg_val_f1,
                "config_name": config.name,
            }, ckpt_path)
            print(f"  >> saved best checkpoint (iou={avg_val_iou:.4f}) to {ckpt_path}")
        else:
            epochs_without_improvement += 1

        # --- Early stopping ---
        if epochs_without_improvement >= config.patience_stop:
            print(f"\n  early stopping at epoch {epoch} "
                  f"(no improvement for {config.patience_stop} epochs)")
            break

        print()  # blank line between epochs

    # --- Final summary ---
    total_training_time = time.time() - training_start
    minutes = total_training_time / 60
    print(f"\n{'='*60}")
    print(f"attempt: {config.name}")
    print(f"best val IoU: {best_val_iou:.4f}")
    print(f"epochs run: {len(history['val_iou'])}")
    print(f"total time: {minutes:.1f} min ({total_training_time:.0f} s)")
    print(f"{'='*60}")

    # --- Generate predictions from best checkpoint ---
    # Load the best model (not the final epoch's model) and run inference
    # on val and test tiles. This produces masks and overlays that can be
    # compared across attempts.
    if not smoke and best_val_iou > 0:
        ckpt_path = results_dir / "best.pt"
        if ckpt_path.exists():
            from src.predict import generate_predictions
            print("\ngenerating predictions from best checkpoint...")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            generate_predictions(
                model, device, config.patches_root, results_dir,
            )

    return history


# ============================================================================
#  CLI ENTRY POINT — for smoke tests from the command line
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a roof segmentation model.",
    )
    parser.add_argument(
        "--attempt", type=str, required=True,
        help="attempt number (01-05)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="run a 2-epoch smoke test instead of full training",
    )
    args = parser.parse_args()

    config = get_config(args.attempt)
    train(config, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
