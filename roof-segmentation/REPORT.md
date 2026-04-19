# Roof Segmentation — Few-Shot Comparison: Results Report

## Summary

Five segmentation architectures were compared on the Inria Austin aerial
imagery dataset with only **24 training samples** (1024x1024 patches),
all using the **same training recipe** (MEDIUM augmentation, BCE+Dice loss,
AdamW, default dropout). The goal was to find which architecture gives the
best results *out of the box*, without problem-specific hyperparameter
tuning. Each model was evaluated on a held-out validation set (6 tiles)
during training, and a separate test set (6 tiles) read only once for the
final reported numbers.

**Winner: SMP DeepLabV3+ with ImageNet-pretrained ResNet34 encoder
(Attempt #03)**, achieving **test IoU 0.802** and **test F1 0.890** — a
remarkably strong result for a model trained on just 24 labeled patches.

The most surprising finding: the SAM-based models (frozen ViT-B encoder +
custom decoders) performed *worse* than the conventional CNN models, despite
having access to a much larger pretrained prior. SAM's encoder was trained
for promptable segmentation, not dense semantic segmentation — and freezing
it meant it never adapted to aerial building features.

---

## Headline Results

| # | Model | Best Epoch | Val IoU | Val F1 | **Test IoU** | **Test F1** | Training Time |
|---|-------|-----------|---------|--------|-------------|------------|---------------|
| 1 | U-Net from scratch | 36 | 0.735 | 0.845 | 0.757 | 0.861 | ~74 min |
| 2 | SMP U-Net + ResNet34 (ImageNet) | 30 | 0.765 | 0.865 | 0.796 | 0.886 | ~50 min |
| **3** | **SMP DeepLabV3+ + ResNet34 (ImageNet)** | **39** | **0.772** | **0.870** | **0.802** | **0.890** | **~65 min** |
| 4 | SAM ViT-B frozen + conv decoder | 18 | 0.690 | 0.814 | 0.646 | 0.783 | ~127 min |
| 5 | SAM ViT-B + U-Net decoder (skip connections) | 18 | 0.740 | 0.849 | 0.758 | 0.862 | ~104 min |

All times on Google Colab free T4 GPU. Attempts 04 and 05 ran at fp32
without mixed precision, contributing to their longer training times.

---

## Analysis

### 1. ImageNet pretraining is the single biggest factor

The jump from Attempt #01 (no pretraining, test IoU 0.757) to Attempt #02
(ImageNet-pretrained encoder, test IoU 0.796) is **+3.9 percentage points**
— the largest single improvement in the comparison. With only 24 training
patches, a randomly initialized encoder struggles to learn both low-level
features (edges, textures) and high-level semantics (what a roof looks
like). The ImageNet prior provides the low-level features for free, letting
the model focus its limited training budget on learning the task-specific
semantics.

### 2. Architecture matters, but less than pretraining

Attempt #03 (DeepLabV3+ with ASPP, test IoU 0.802) beats Attempt #02
(U-Net, test IoU 0.796) by **+0.6 points**. The ASPP module's multi-scale
context helps with the varied roof sizes in Austin (small houses vs. large
commercial buildings), but the gain is modest compared to the pretraining
effect.

### 3. SAM's frozen encoder underperforms — the biggest surprise

The hypothesis going in was that SAM's massive pretraining (11 million
images, 1 billion masks) would provide a stronger prior than ImageNet's
1.2 million natural images, especially for few-shot scenarios. The results
contradict this:

- **SAM frozen + conv decoder (#04): test IoU 0.646** — worst of all five,
  15.6 points below the best CNN.
- **SAM + U-Net decoder (#05): test IoU 0.758** — competitive with the
  from-scratch U-Net, but still below both pretrained CNN models.

**Why SAM underperformed:**

1. **Task mismatch.** SAM was trained for *promptable* segmentation (given
   a point or box, segment that specific object). We discarded the prompt
   pathway entirely and used the encoder as a generic feature extractor for
   *dense semantic* segmentation. SAM's encoder features may not be optimal
   for this task without the prompt that SAM was designed to use.

2. **Frozen encoder = no domain adaptation.** The SAM encoder was completely
   frozen — it never saw a single aerial image during training. Its features
   reflect SAM's web-crawled training data (mostly natural images at varied
   scales), not nadir aerial imagery of buildings. The CNN models, by
   contrast, had their encoders unfrozen at epoch 20, allowing them to
   adapt their features to the specific characteristics of the Austin
   aerial tiles (uniform scale, nadir viewpoint, specific color palette).

3. **Spatial resolution bottleneck.** SAM ViT-B produces a single 64x64
   feature map from 1024x1024 input (16x downsampling). There are no
   intermediate-resolution feature maps like a CNN's multi-scale pyramid.
   The decoder must reconstruct all fine spatial detail from this single
   coarse grid. The SMP U-Net, by comparison, has skip connections at
   five different resolutions, preserving sharp building boundaries.

4. **Skip connections partially compensate.** Attempt #05 (SAM + U-Net
   decoder with ViT skip connections) improved substantially over #04
   (+11.2 points on test), confirming that the spatial resolution
   bottleneck is a real problem. However, all ViT blocks output at the
   same 64x64 resolution — the "skip connections" are multi-depth (early
   vs. late features) but not multi-scale (all 64x64), so they can't
   fully compensate for the lack of high-resolution spatial information.

5. **Insufficient training budget.** Both SAM models ran for only 28
   epochs before early stopping (no improvement for 10 epochs). They
   also ran at fp32 without mixed precision, making each epoch ~4.5
   minutes vs. ~1 minute for the CNN models. The SAM models may have
   benefited from more training time or a different learning rate schedule,
   but free Colab's time constraints prevented longer runs.

### 4. Per-tile analysis reveals consistent patterns

**austin6** (val) is the hardest tile for every model — it consistently
has the lowest IoU across all attempts (0.53–0.68). This tile likely
contains atypical building patterns (very dense, unusual shapes, or mixed
vegetation) that are underrepresented in the 24 training patches.

**austin36** (test) shows the largest gap between CNN and SAM models.
The SAM frozen decoder (#04) achieves only 0.53 IoU here while the
DeepLabV3+ (#03) reaches 0.76. Visual inspection shows that #04 bleeds
extensively into streets and yards between buildings, lacking the boundary
precision that the CNN's multi-scale skip connections provide.

**austin33** (test) is one tile where SAM performs relatively well — both
SAM models reach 0.76–0.83, close to the CNN models (0.78–0.86). This
tile may have larger, more distinct buildings where the 64x64 feature
resolution is sufficient to delineate boundaries.

### 5. Two-stage training works well for CNNs

The frozen-then-unfrozen training strategy is clearly visible in the
training curves for Attempts #02 and #03:
- **Epochs 1–20** (encoder frozen): steady improvement from ~0.63 to
  ~0.70 val IoU as the decoder learns to interpret ImageNet features.
- **Epoch 21** (unfreeze): a brief dip as the optimizer adjusts to the
  much larger parameter space, followed by a push to ~0.74–0.78.
- **Epochs 25–40**: gradual refinement, with the model converging at
  its best around epoch 30–39.

The epoch timing reflects this too: ~60s per epoch while frozen (decoder
only), jumping to ~90–100s after unfreezing (full model).

### 6. Efficiency comparison

| Model | Time to best epoch | Time per epoch | IoU per GPU-hour |
|-------|--------------------|---------------|-----------------|
| #01 U-Net scratch | 58 min | 97s | 0.78 |
| #02 SMP U-Net | 35 min | 60–89s | 1.36 |
| **#03 SMP DeepLab** | **48 min** | **47–101s** | **1.00** |
| #04 SAM frozen | 82 min | 273s | 0.47 |
| #05 SAM U-Net dec | 67 min | 222s | 0.68 |

The SMP U-Net (#02) offers the best IoU-per-GPU-hour — it reaches 0.796
test IoU in just 35 minutes. If compute budget is constrained (as it was
on free Colab), this is the pragmatic choice. DeepLabV3+ (#03) is slightly
better in absolute quality but takes longer.

---

## Training Curves

### Convergence behavior

- **Attempts 01–03** all converge smoothly. Early stopping triggered around
  epoch 46 (#01), 40 (#02), and 49 (#03). Training loss continues to
  decrease after validation plateaus, indicating some overfitting but not
  catastrophic — the augmentation pipeline is doing its job.

- **Attempt 04** shows very slow learning: val IoU creeps from 0.654 to
  0.688 over 28 epochs (only +3.4 points). The frozen ViT-B encoder
  features are apparently not easily separable for binary building
  segmentation without domain adaptation.

- **Attempt 05** learns faster than #04 (0.685 → 0.739 in 28 epochs),
  confirming that the skip connections provide richer signal to the
  decoder. But it still plateaus well below the CNN models.

### Two-stage transition

For Attempts #02 and #03, the unfreeze at epoch 21 is clearly visible:
- Train loss spikes briefly (optimizer suddenly has 24M parameters instead
  of 3M).
- Val IoU dips for 1–2 epochs, then climbs past the frozen-phase ceiling.
- Epoch time roughly doubles (encoder gradients now computed).

---

## Visual Comparison

See the generated comparison grids:
- `comparison_val.png` — all 6 validation tiles, ground truth (green) vs.
  5 attempts (red overlays)
- `comparison_test.png` — all 6 test tiles, same layout

Key visual observations:
- **CNN models (01–03)** produce clean, sharp building boundaries with
  relatively few false positives in streets/vegetation.
- **SAM frozen (#04)** produces blobby, over-segmented masks that bleed
  into adjacent streets and yards. Building boundaries are poorly defined.
- **SAM + U-Net decoder (#05)** is noticeably sharper than #04 but still
  softer than the CNN models, especially on smaller buildings.
- All models struggle with **very small structures** (garden sheds, garages)
  and **buildings partially occluded by trees**.

---

## Conclusions

1. **With a shared default recipe, ImageNet-pretrained CNNs win clearly.**
   SMP DeepLabV3+ reaches 0.802 test IoU on 24 patches without any
   problem-specific tuning — strong enough for practical use.

2. **Architecture matters less than pretraining.** The gap between U-Net
   and DeepLabV3+ (both ImageNet, same recipe) is only 0.6 points. The
   gap between no-pretraining and ImageNet is 3.9 points.

3. **SAM's frozen encoder underperforms on a default recipe.** The
   "massive pretrained prior" hypothesis did not hold under these
   conditions. SAM was trained for promptable segmentation, not dense
   semantic segmentation, and freezing the encoder prevents domain
   adaptation. Per-model tuning (unfreezing, LoRA, different LR/aug)
   might close the gap — this is the main question for future work.

4. **Skip connections matter.** Both within CNN architectures (U-Net's
   multi-scale skips) and the SAM track (ViT block skip connections),
   richer decoder architectures consistently outperform simpler ones.
   The 11.2-point jump from #04 to #05 is the largest architectural
   effect in the comparison.

5. **24 samples + oversampling + augmentation is enough to train from
   scratch.** The from-scratch U-Net reaches 0.757 test IoU without
   any pretrained prior — no exotic few-shot methods needed.

---

## Future work

The current results use a shared recipe for all models. The natural
next step is **per-model tuning on the top candidates**:

- **Candidates**: #02 (SMP U-Net), #03 (DeepLabV3+), #05 (SAM + U-Net decoder)
- **Tuning axes**: learning rate, augmentation preset (LIGHT vs. MEDIUM
  vs. HEAVY), decoder dropout rate, loss weighting
- **SAM-specific**: unfreeze the encoder with LoRA/adapters or a very low
  LR (1e-5) to allow domain adaptation; add fp16 mixed precision for
  ~2x training speed; try SAM2
- **Data-budget ablation**: retrain the best 1–2 models on {5, 12, 24}
  training tiles to measure how performance degrades with less data —
  this would produce the project's headline figure

---

## Files

```
roof-segmentation/
├── REPORT.md                    ← this file
├── comparison_val.png           ← side-by-side visual comparison (val tiles)
├── comparison_test.png          ← side-by-side visual comparison (test tiles)
└── results/
    ├── 01/                      ← U-Net scratch
    │   ├── best.pt, history.csv
    │   └── predictions/{val,test}/{masks,overlays}/
    ├── 02/                      ← SMP U-Net + ResNet34
    │   ├── best.pt, history.csv
    │   └── predictions/{val,test}/{masks,overlays}/
    ├── 03/                      ← SMP DeepLabV3+ + ResNet34 (winner)
    │   ├── best.pt, history.csv
    │   └── predictions/{val,test}/{masks,overlays}/
    ├── 04/                      ← SAM frozen + conv decoder
    │   ├── best.pt, history.csv
    │   └── predictions/{val,test}/{masks,overlays}/
    └── 05/                      ← SAM ViT-B + U-Net decoder
        ├── best.pt, history.csv
        └── predictions/{val,test}/{masks,overlays}/
```
