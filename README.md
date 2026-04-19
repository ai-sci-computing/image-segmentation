# Roof Segmentation — Few-Shot Comparison

Binary roof segmentation from aerial imagery under strict data scarcity.
Five architectures are compared head-to-head on a 24-sample training budget,
using a near-identical training recipe for all models.

---

## Goal

Compare how well different segmentation architectures perform on 24
labeled training samples, without problem-specific tuning.

All five models use the same augmentation (MEDIUM), loss (BCE+Dice 0.5/0.5),
optimizer (AdamW), dropout (0.3), and oversampling (400 draws/epoch). The
one exception is learning rate: 1e-3 for the CNN track, 1e-4 for the SAM
track.

This near-identical recipe isolates the effect of architecture and
pretraining from the effect of per-model hyperparameter tuning. A
follow-up experiment with per-model tuning on the top candidates is planned
as future work.

---

## Data

- **Source**: Inria Aerial Image Labeling dataset, Austin subset.
- **Tiles**: 36 RGB tiles at 5000×5000 px, each with a binary mask of building
  footprints.
- **Label semantics**: the masks mark building footprints. For nadir aerial
  imagery, this is effectively the *roof outline* (including overhangs/eaves —
  no distinction between roof and walls).
- **Source layout** (not committed to the repo):
  ```
  data/imgs/austin{1..36}.tif
  data/msks/austin{1..36}.tif
  ```
  The source TIFFs live locally only and are removed once `patches/` is
  finalized. The dataset-of-record is `patches/`, not `data/`.

---

## Split

Three-way split, fixed once, never revisited.

| Set   | Tiles                                            | Count | Purpose                                                                 |
| ----- | ------------------------------------------------ | ----- | ----------------------------------------------------------------------- |
| train | austin2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,20,22,23,24,25,27,28,29,30 | 24    | Model fitting                                                           |
| val   | austin1, 6, 11, 16, 21, 26                        | 6     | LR tuning, augmentation tuning, checkpoint selection, in-training IoU/F1 |
| test  | austin31, 32, 33, 34, 35, 36                      | 6     | Final reported numbers — read **once** at the end of each attempt        |

**Test discipline**: no tuning loop is permitted to read test. Enforced in code
via separate dataloader and config flag.

---

## Preprocessing

- **One 1024×1024 crop per source tile** → 36 patches total.
- **Smart-center crop**: start at the geometric center; if the centered 1024²
  window has < 5% positive (building) pixels, slide the window around the
  5000² source to find a position with meaningful coverage.
- **Manual verification**: all 36 outputs are eyeballed once; any that look
  bad are nudged by hand.
- **Formats**: JPEG q90 for RGB images (~400 KB each), PNG for binary masks
  (~50 KB each). Total footprint: **~15 MB**.
- **Committed to repo**: yes. `patches/` is the dataset-of-record.

```
patches/
  train/{images,masks}/austin{...}.{jpg,png}
  val/{images,masks}/austin{1,6,11,16,21,26}.{jpg,png}
  test/{images,masks}/austin{31..36}.{jpg,png}
```

**No runtime resizing anywhere in the pipeline.** The only resampling in the
entire project is the one-time preprocessing step (5000² → 1024²). Every
model trains and evaluates at 1024².

---

## Attempts

Five model configurations, ordered so each builds on a lesson from the
previous.

| # | Model                                                        | Encoder / prior             | Trainable params (approx) | What it tests                                  |
| - | ------------------------------------------------------------ | --------------------------- | ------------------------- | ----------------------------------------------- |
| 1 | U-Net from scratch                                           | none                        | ~7–30 M                   | Baseline floor; no pretrained prior             |
| 2 | SMP U-Net + ResNet (ImageNet)                                | ImageNet-pretrained CNN     | ~24 M                     | Standard transfer-learning recipe               |
| 3 | SMP DeepLabV3+ + ResNet (ImageNet)                           | ImageNet-pretrained CNN     | ~25 M                     | Architecture impact on top of #2                |
| 4 | SAM ViT-B frozen + conv decoder                              | frozen ViT (SAM)            | ~1–3 M                    | Massive pretrained prior; very few trainable params |
| 5 | SAM ViT-B encoder + U-Net decoder w/ skip connections        | frozen ViT (SAM), rich decoder | ~5–10 M                | Decoder capacity on top of SAM prior            |

Attempts #2 and #4 are ported from earlier reference implementations.

Attempt #5 pulls features from intermediate ViT blocks (e.g., blocks 3/6/9/12
for ViT-B), reshapes `(B, N, C)` → `(B, C, H/16, W/16)`, and feeds them as
skip connections into a U-Net-style decoder that upsamples back to 1024².

---

## Training protocol (shared across all attempts)

- **Framework**: PyTorch, `segmentation_models_pytorch` for the CNN track,
  `segment_anything` for the SAM track.
- **Input resolution**: 1024 × 1024 (unchanged from `patches/`).
- **Sample oversampling** (critical for tiny datasets — see "Design notes"):
  each epoch randomly draws `samples_per_epoch=400` indices *with replacement*
  from the 24 training patches; each draw is independently augmented. An
  epoch is thus ~100 optimizer steps at batch 4, not 6.
- **Loss**: `w * BCE + (1-w) * Dice` — weighting `w` is a per-model
  hyperparameter (default 0.5, sweep considered during tuning).
- **Optimizer**: AdamW, weight decay 1e-4, optionally EMA-wrapped.
- **Callbacks**:
  - EarlyStopping on `val_iou`, patience ~10 epochs
  - ReduceLROnPlateau, factor 0.5, patience 3
  - ModelCheckpoint: save best by `val_iou`
  - PredictionVisualizer: dump prediction overlays every N epochs for the
    Colab/terminal feedback loop
- **Two-stage training for pretrained-encoder attempts (#2, #3)**:
  1. Train with `encoder_freeze=True` for ~20 epochs (decoder warmup).
  2. Unfreeze encoder, continue training at same or reduced LR.
- **Decoder dropout** (SMP wrapper): `Dropout2d(p)` injected between decoder
  stages, `p` tunable per model. Encoder is never wrapped in dropout.
- **Mixed precision**: fp16 on Colab T4.
- **Seed**: 42 for all seedable operations.

---

## Shared recipe

All five models use the same training recipe so the comparison isolates
architecture and pretraining, not hyperparameter choices.

| Parameter | Value | Notes |
| --------- | ----- | ----- |
| Augmentation | MEDIUM | D4 + ShiftScaleRotate + brightness/contrast + HSV |
| Loss | BCE + Dice (0.5 / 0.5) | |
| Optimizer | AdamW, weight decay 1e-4 | |
| LR (CNN track) | 1e-3 | Attempts #1–#3 |
| LR (SAM track) | 1e-4 | Attempts #4–#5 (frozen encoder, smaller decoder) |
| Decoder dropout | 0.3 | |
| Oversampling | 400 draws/epoch | |
| Batch size | 4 (CNN) / 2 (SAM) | SAM ViT-B needs more memory at 1024² |
| Seed | 42 | |

**Future work**: per-model tuning of LR, augmentation preset, and dropout
on the top candidates (#02, #03, #05) to find each architecture's ceiling.

---

## Augmentation presets

Built around the curated pipeline from the reference Keras code, which
explicitly excluded `GaussianBlur` and `GaussNoise` because "there is not
much variety of blur in the given images" — a dataset-specific observation
worth respecting.

```
LIGHT:
  HorizontalFlip, VerticalFlip, RandomRotate90
  (the full D4 dihedral group — free for aerial nadir imagery)

MEDIUM (default / reference recipe):
  LIGHT +
  ShiftScaleRotate(shift=0.2, scale=0.3, rotate=0.4, BORDER_REFLECT, p=0.7)
  RandomBrightnessContrast(0.2, 0.3, p=0.5)
  HueSaturationValue(hue=10, sat=15, val=15, p=0.3)

HEAVY:
  MEDIUM +
  RandomShadow
  RandomGamma
  CLAHE
  mild JPEG compression artifacts

EXTREME:
  HEAVY +
  mild ElasticTransform
  CoarseDropout
  RandomFog
```

Hypothesis: MEDIUM or HEAVY wins for CNN track; LIGHT or MEDIUM wins for
SAM track (frozen ViT encoders tend to prefer milder augmentation).

---

## Evaluation

- **Metric**: mean IoU + F1, computed on reconstructed full-resolution
  predictions (not on patches, since there's exactly one patch per tile).
- **Validation**: used continuously during tuning.
- **Test**: read **exactly once per attempt**, after all tuning is finalized.
  Test numbers are the leaderboard.
- **Qualitative output**: prediction overlays on val + test tiles, committed
  to `results/<attempt>/overlays/`.

---

## Reporting

### Headline table

| # | Model | LR | Aug | Dropout | Loss (BCE/Dice) | Best Epoch | Val IoU | Val F1 | **Test IoU** | **Test F1** | Time |
| - | ----- | -- | --- | ------- | --------------- | ---------- | ------- | ------ | ------------ | ----------- | ---- |
| 1 | U-Net from scratch | 1e-3 | MEDIUM | 0.0 | 0.5/0.5 | 36 | 0.735 | 0.845 | **0.757** | **0.861** | 74 min |
| 2 | SMP U-Net + ResNet34 (ImageNet) | 1e-3 | MEDIUM | 0.3 | 0.5/0.5 | 30 | 0.765 | 0.865 | **0.796** | **0.886** | 50 min |
| **3** | **SMP DeepLabV3+ + ResNet34 (ImageNet)** | **1e-3** | **MEDIUM** | **0.3** | **0.5/0.5** | **39** | **0.772** | **0.870** | **0.802** | **0.890** | **65 min** |
| 4 | SAM ViT-B frozen + conv decoder | 1e-4 | MEDIUM | 0.3 | 0.5/0.5 | 18 | 0.690 | 0.814 | **0.646** | **0.783** | 127 min |
| 5 | SAM ViT-B + U-Net decoder (skip conn.) | 1e-4 | MEDIUM | 0.3 | 0.5/0.5 | 18 | 0.740 | 0.849 | **0.758** | **0.862** | 104 min |

All training on Google Colab free T4 GPU. Val and test are both shown — val
was used during training (contaminated by tuning); test is the honest number.

**Winner**: Attempt #3 (DeepLabV3+) with test IoU **0.802**.

---

## Project phases

```
Phase 1 — Architecture comparison (shared recipe)               [DONE]
  All 5 models trained with the same recipe. Results in headline table.
  This is the core deliverable of the current study.

Phase 2 — Per-model tuning on top candidates                    [FUTURE WORK]
  LR sweeps, augmentation preset comparison (LIGHT/MEDIUM/HEAVY),
  dropout tuning on the top 2–3 architectures (#02, #03, #05).

Phase 3 — Data-budget ablation                                  [FUTURE WORK]
  Best models retrained on {5, 12, 24} training tiles.
  Produces IoU vs. training tile count curve.
```

---

## Repository layout

```
image-segmentation/
├── README.md                    — this file
├── .gitignore                   — excludes data/ and runtime artifacts
├── requirements.txt             — Python dependencies
├── patches/                     — committed dataset-of-record (~16 MB)
│   ├── overview.png             — visual grid of all 36 patches
│   ├── train/{images,masks}/    — 24 training patches
│   ├── val/{images,masks}/      — 6 validation patches
│   └── test/{images,masks}/     — 6 test patches
├── scripts/
│   ├── make_patches.py          — one-time preprocessing (5000² → 1024²)
│   ├── eyeball_patches.py       — visual check of all 36 patches
│   └── compare_attempts.py      — side-by-side prediction comparison grid
├── src/
│   ├── data.py                  — Dataset + oversampling Sampler
│   ├── augment.py               — Albumentations presets (LIGHT/MEDIUM/HEAVY/EXTREME)
│   ├── models/
│   │   ├── unet_scratch.py      — Attempt #1: U-Net, no pretraining
│   │   ├── smp_wrapper.py       — Attempts #2, #3: SMP models + decoder dropout
│   │   └── sam_decoder.py       — Attempts #4, #5: SAM frozen encoder + decoders
│   ├── losses.py                — BCE+Dice with tunable weighting
│   ├── metrics.py               — IoU, F1, per-tile evaluation
│   ├── predict.py               — generate prediction masks and overlays
│   ├── train.py                 — shared training loop, callbacks, --smoke flag
│   ├── diagnostics.py           — structured per-epoch logging
│   └── config.py                — per-attempt hyperparameter dataclasses
├── tests/                       — 81 tests, local pytest (CPU only)
│   ├── test_dataset.py
│   ├── test_augment.py
│   ├── test_models.py
│   ├── test_losses.py
│   └── test_metrics.py
└── results/                     — per-attempt outputs (on Google Drive / local)
    └── <attempt>/
        ├── history.csv          — per-epoch metrics + timing
        ├── best.pt              — best checkpoint by val IoU
        └── predictions/
            ├── {val,test}/masks/     — predicted binary masks
            └── {val,test}/overlays/  — original + red mask overlays
```

Source TIFFs (`data/`) are local-only and git-ignored. Training was done
on Colab via CLI (`python -m src.train --attempt 02`), not via notebooks.

---

## How to run

### Local (once, for preprocessing and testing)

```bash
# one-time patch generation from local source TIFFs
python scripts/make_patches.py --src data/ --dst patches/

# unit tests (runs on CPU in seconds)
pytest tests/
```

### Colab (for each training attempt)

One attempt per Colab session. Results saved to Google Drive to survive
disconnects.

```python
# Cell 1 — Setup
!rm -rf image-segmentation
!git clone https://github.com/ai-sci-computing/image-segmentation.git
%cd image-segmentation
!pip install -q -r requirements.txt
!pip install -q git+https://github.com/facebookresearch/segment-anything.git

from google.colab import drive
drive.mount('/content/drive')

# SAM checkpoint (needed for attempts 04 and 05 only)
!wget -q -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
import src.models.sam_decoder as sam_dec
from pathlib import Path
sam_dec.DEFAULT_CHECKPOINT = Path("sam_vit_b_01ec64.pth")
```

```python
# Cell 2 — Train one attempt (change the number each session)
ATTEMPT = "02"
from src.config import get_config
from src.train import train
results_dir = Path(f"/content/drive/MyDrive/roof-segmentation/results/{ATTEMPT}")
history = train(get_config(ATTEMPT), results_dir=results_dir)
```

The training loop prints **structured per-epoch diagnostics**:

```
=== attempt: smp_unet_resnet34  epoch: 5/60  lr: 1.00e-03 ===
train: loss=0.214 (n=400 augmented, bs=4, steps=100)
val:   loss=0.287  iou=0.612  f1=0.748  (n=6 tiles)
per_tile_iou: austin1=0.71  austin6=0.55  austin11=0.68
              austin16=0.63  austin21=0.49  austin26=0.61
worst_tile:   austin21 (iou=0.49)
gpu_mem: 4.2 / 15.6 GB   step_time: 0.42s
```

Free Colab T4 (16 GB VRAM) is sufficient for all attempts. CNN models
(#1–#3) run at ~1 min/epoch; SAM models (#4–#5) at ~4 min/epoch (fp32).

---

## Testing strategy

**Two-tier approach.** Local red-green-refactor for everything that doesn't
need GPU; structured observational logging for everything that does.

**Tier 1 — local pytest** (runs in seconds on any laptop):

- `test_dataset.py` — patch loader returns correct shapes and pixel ranges;
  mask is binary; no NaNs
- `test_augment.py` — augmentation pipeline preserves shapes, mask stays
  binary, handles all-zero and all-positive edge cases
- `test_models.py` — each model factory builds, forward pass on
  `(1, 3, 1024, 1024)` zero tensor returns `(1, 1, 1024, 1024)`,
  **trainable param count matches expectation** (catches "did I actually
  freeze the SAM encoder?" bugs)
- `test_losses.py` — BCE+Dice returns scalar > 0, gradients flow, hand-
  computed values match on tiny tensors
- `test_metrics.py` — IoU on hand-crafted 4×4 masks matches expected fraction
- `test_inference.py` — full-tile reconstruction is bit-exact identity when
  the model is `nn.Identity()`

Smoke-test flag on the training script: `train.py --smoke` runs 2 epochs on
a CPU-tiny subset to exercise the entire pipeline (data → model → loss →
optimizer → eval → checkpoint) before touching Colab.

**Tier 2 — Colab observational**:

Structured per-epoch diagnostics (see "How to run") are the remote debugging
interface. The loop is: you paste output to chat, I diagnose overfitting /
class imbalance / LR issues from the numbers, you adjust and re-run.

---

## Design notes

Non-obvious choices and the reasoning behind them.

- **Oversampled epochs (`samples_per_epoch=400`)**. With 24 training
  samples at batch 4, a naive epoch is 6 optimizer steps — too small for
  LR schedules and early stopping to behave sensibly. Oversampling with
  replacement + fresh augmentation per draw restores normal training
  dynamics without faking data. Each epoch sees ~100 steps of genuinely
  distinct augmented views. Pattern borrowed from the reference Keras code.

- **Single resolution, no runtime resizing**. Upsampling creates fake
  detail and can't add information; downsampling loses it. Storing and
  training at the same resolution (1024²) means the only resampling in
  the entire pipeline is the one-time preprocessing step. Both tracks
  eat identical bytes.

- **Per-model tuning, not shared recipe**. The research question is
  "what's each approach capable of," not "which wins with a fixed
  recipe." Locking LR or augmentation across fundamentally different
  models (from-scratch CNN vs. frozen pretrained ViT) would penalize
  whichever disagrees with the chosen value — measuring misconfiguration
  rather than potential. Equal tuning *budget* per model, documented in
  the results table, is the fairness discipline instead.

- **Decoder-only dropout**. Dropout inside a pretrained encoder disrupts
  its BN/feature statistics; dropout in the decoder regularizes the
  tiny set of newly-trained parameters, which is where overfitting
  actually happens with 30 samples. Matches the pattern used in the
  reference SAM decoder code.

- **Test set held back**. Per-model tuning against val means val is
  contaminated by tuning effort by the time the final run happens. Test
  is read once at the end of each attempt, never influences any decision,
  and provides the honest leaderboard.

- **No GaussianBlur / GaussNoise in augmentation**. Excluded based on
  a dataset-specific observation from the reference code: the source
  imagery has very little blur variation, so these augmentations add
  unrealistic variance without matching any real-world deployment
  condition.

---

## Status

**Phase 1 complete.** All 5 architectures compared on the same recipe.
Full results in the headline table above. Detailed analysis in
`roof-segmentation/REPORT.md`.

### Key findings

- **Winner**: SMP DeepLabV3+ + ResNet34 (ImageNet), test IoU **0.802**
- **ImageNet pretraining** is the single biggest factor (+3.9 IoU points
  over from-scratch)
- **SAM's frozen encoder underperformed** — test IoU 0.646 (worst of all
  five), despite having the largest pretrained prior. Cause: task mismatch
  (promptable vs. dense segmentation), no domain adaptation (encoder
  frozen), and spatial resolution bottleneck (single 64x64 feature map)
- **Even from-scratch U-Net reaches 0.757** thanks to oversampling +
  augmentation — no exotic few-shot methods needed
- **Skip connections matter**: SAM + U-Net decoder (#05, 0.758) beats
  SAM + conv decoder (#04, 0.646) by 11.2 points

### Future work

- **Per-model tuning** on the top candidates (#02, #03, #05): LR sweeps,
  augmentation preset comparison, dropout tuning
- **Data-budget ablation**: retrain best models on {5, 12, 24} tiles to
  measure how performance degrades with less data
- **SAM improvements**: unfreeze encoder with LoRA/adapters, add fp16
  mixed precision, try SAM2
- **Augmentation study**: compare LIGHT/MEDIUM/HEAVY/EXTREME presets —
  frozen ViT encoders may prefer milder augmentation than CNNs
