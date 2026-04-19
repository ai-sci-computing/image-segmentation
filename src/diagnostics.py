"""
src/diagnostics.py — Structured per-epoch log formatter.

Produces the copy-pasteable diagnostic output described in README § How to run.
The format is designed so that training output from Colab can be pasted into
a chat/terminal and parsed by a human or LLM to diagnose overfitting, LR
issues, class imbalance, etc.

Single responsibility: formatting. Called by train.py every epoch.
"""

from typing import Dict, List, Optional


def format_epoch_log(
    attempt_name: str,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    val_iou: float,
    val_f1: float,
    per_tile_iou: Optional[List[float]] = None,
    tile_names: Optional[List[str]] = None,
    train_samples: int = 400,
    batch_size: int = 4,
    n_val_tiles: int = 6,
    gpu_mem_used: Optional[float] = None,
    gpu_mem_total: Optional[float] = None,
    step_time: Optional[float] = None,
    lr: Optional[float] = None,
) -> str:
    """Format a structured diagnostic block for one epoch.

    All arguments with default None are omitted from the output if not
    provided — this keeps the format clean when running on CPU (no GPU
    memory info) or during smoke tests (no per-tile breakdown).

    Returns:
        A multi-line string ready to print. Example:

        === attempt: smp_unet_resnet34  epoch: 5/60  lr: 1.00e-03 ===
        train: loss=0.214 (n=400 augmented, bs=4, steps=100)
        val:   loss=0.287  iou=0.612  f1=0.748  (n=6 tiles)
        per_tile_iou: austin1=0.71  austin6=0.55  austin11=0.68
                      austin16=0.63  austin21=0.49  austin26=0.61
        worst_tile:   austin21 (iou=0.49)
        gpu_mem: 4.2 / 15.0 GB   step_time: 0.42s
    """
    steps = train_samples // batch_size
    lines = []

    # Header line
    header = f"=== attempt: {attempt_name}  epoch: {epoch}/{total_epochs}"
    if lr is not None:
        header += f"  lr: {lr:.2e}"
    header += " ==="
    lines.append(header)

    # Train metrics
    lines.append(
        f"train: loss={train_loss:.4f} "
        f"(n={train_samples} augmented, bs={batch_size}, steps={steps})"
    )

    # Val metrics
    lines.append(
        f"val:   loss={val_loss:.4f}  iou={val_iou:.4f}  f1={val_f1:.4f}  "
        f"(n={n_val_tiles} tiles)"
    )

    # Per-tile IoU breakdown (optional — may not be computed every epoch)
    if per_tile_iou is not None and tile_names is not None:
        tiles_str = "  ".join(
            f"{name}={iou:.2f}" for name, iou in zip(tile_names, per_tile_iou)
        )
        lines.append(f"per_tile_iou: {tiles_str}")

        # Worst tile
        worst_idx = min(range(len(per_tile_iou)), key=lambda i: per_tile_iou[i])
        lines.append(
            f"worst_tile:   {tile_names[worst_idx]} "
            f"(iou={per_tile_iou[worst_idx]:.2f})"
        )

    # GPU memory and step time (optional — not available on CPU)
    extras = []
    if gpu_mem_used is not None and gpu_mem_total is not None:
        extras.append(f"gpu_mem: {gpu_mem_used:.1f} / {gpu_mem_total:.1f} GB")
    if step_time is not None:
        extras.append(f"step_time: {step_time:.2f}s")
    if extras:
        lines.append("   ".join(extras))

    return "\n".join(lines)
