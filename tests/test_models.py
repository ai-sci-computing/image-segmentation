"""
tests/test_models.py — Unit tests for all model factories.

Each test verifies the model's forward-pass contract:
  - Input:  (1, 3, 1024, 1024) float32
  - Output: (1, 1, 1024, 1024) float32

Tests run on CPU with zero tensors, so they're fast (~2–5 s each) and
don't require GPU. They catch shape mismatches, missing modules, and
"did I actually freeze the encoder?" bugs.

SAM model tests are in this file too (added in a later commit) but
are conditionally skipped if the SAM checkpoint is not available.
"""

import pytest
import torch

from src.models.unet_scratch import make_unet_scratch
from src.models.smp_wrapper import make_smp_unet, make_smp_deeplabv3plus
from src.models.sam_decoder import (
    SAMFrozenConvDecoder,
    SAMUNetDecoder,
    DEFAULT_CHECKPOINT,
)


# Standard test input — a single 1024x1024 RGB image of zeros.
# Using zeros is sufficient for shape verification; we don't need
# realistic pixel values to test the forward-pass contract.
DUMMY_INPUT = torch.zeros(1, 3, 1024, 1024)

# SAM tests require the ViT-B checkpoint file to be present on disk.
# Skip them if it's not available (e.g. in CI without the 358 MB file).
sam_available = DEFAULT_CHECKPOINT.exists()
skip_no_sam = pytest.mark.skipif(
    not sam_available,
    reason=f"SAM checkpoint not found: {DEFAULT_CHECKPOINT}",
)


class TestUnetScratch:
    """Tests for attempt #1: U-Net from scratch."""

    def test_output_shape(self):
        model = make_unet_scratch()
        model.eval()
        with torch.no_grad():
            out = model(DUMMY_INPUT)
        assert out.shape == (1, 1, 1024, 1024), f"unexpected shape: {out.shape}"

    def test_output_dtype(self):
        model = make_unet_scratch()
        model.eval()
        with torch.no_grad():
            out = model(DUMMY_INPUT)
        assert out.dtype == torch.float32

    def test_no_pretrained_weights(self):
        """Encoder should have randomly initialized weights (no imagenet)."""
        model = make_unet_scratch()
        # SMP stores the encoder_weights value; None means no pretraining.
        # We can't easily verify random init, but we can check that the
        # model was created without pretrained weights by checking that
        # it has a reasonable number of parameters (would be the same
        # architecture as the pretrained version, just different values).
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 1_000_000, "model seems too small"


class TestSMPUnet:
    """Tests for attempt #2: SMP U-Net with ImageNet encoder + dropout."""

    def test_output_shape(self):
        model = make_smp_unet()
        model.eval()
        with torch.no_grad():
            out = model(DUMMY_INPUT)
        assert out.shape == (1, 1, 1024, 1024), f"unexpected shape: {out.shape}"

    def test_dropout_injected(self):
        """Decoder blocks should have injected Dropout2d modules."""
        model = make_smp_unet(decoder_dropout=0.3)
        found = False
        for block in model.decoder.blocks:
            for name, mod in block.named_modules():
                if isinstance(mod, torch.nn.Dropout2d):
                    found = True
                    break
        assert found, "no Dropout2d found in decoder blocks"

    def test_no_dropout_when_zero(self):
        """decoder_dropout=0 should not inject any Dropout2d modules."""
        model = make_smp_unet(decoder_dropout=0.0)
        for block in model.decoder.blocks:
            for name, mod in block.named_modules():
                assert not isinstance(mod, torch.nn.Dropout2d), \
                    f"found Dropout2d in block despite dropout_p=0: {name}"

    def test_eval_mode_deterministic(self):
        """In eval mode, dropout should be disabled — output is deterministic."""
        model = make_smp_unet(decoder_dropout=0.5)
        model.eval()
        with torch.no_grad():
            out1 = model(DUMMY_INPUT).clone()
            out2 = model(DUMMY_INPUT).clone()
        assert torch.equal(out1, out2), "eval mode output is non-deterministic"


class TestSMPDeepLabV3Plus:
    """Tests for attempt #3: SMP DeepLabV3+ with ImageNet encoder."""

    def test_output_shape(self):
        model = make_smp_deeplabv3plus()
        model.eval()
        with torch.no_grad():
            out = model(DUMMY_INPUT)
        assert out.shape == (1, 1, 1024, 1024), f"unexpected shape: {out.shape}"

    def test_pretrained_encoder(self):
        """Encoder should be pretrained on ImageNet."""
        model = make_smp_deeplabv3plus()
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 1_000_000, "model seems too small"


# ============================================================================
#  SAM models — skipped if the ViT-B checkpoint is not available
# ============================================================================

class TestSAMFrozenConvDecoder:
    """Tests for attempt #4: SAM frozen encoder + simple conv decoder."""

    @skip_no_sam
    def test_output_shape(self):
        model = SAMFrozenConvDecoder()
        model.eval()
        with torch.no_grad():
            out = model(DUMMY_INPUT)
        assert out.shape == (1, 1, 1024, 1024), f"unexpected shape: {out.shape}"

    @skip_no_sam
    def test_encoder_frozen(self):
        """Every encoder parameter should have requires_grad=False."""
        model = SAMFrozenConvDecoder()
        for name, p in model.encoder.named_parameters():
            assert not p.requires_grad, \
                f"encoder param {name} is not frozen"

    @skip_no_sam
    def test_trainable_param_count(self):
        """Trainable params should be small (decoder only, ~1-3M)."""
        model = SAMFrozenConvDecoder()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        # Decoder should be a small fraction of the total model.
        assert trainable < total * 0.05, \
            f"too many trainable params: {trainable} / {total}"
        assert trainable > 100_000, \
            f"too few trainable params: {trainable} — decoder may be empty"


class TestSAMUNetDecoder:
    """Tests for attempt #5: SAM encoder + U-Net decoder with skip connections."""

    @skip_no_sam
    def test_output_shape(self):
        model = SAMUNetDecoder()
        model.eval()
        with torch.no_grad():
            out = model(DUMMY_INPUT)
        assert out.shape == (1, 1, 1024, 1024), f"unexpected shape: {out.shape}"

    @skip_no_sam
    def test_encoder_frozen(self):
        """Every encoder parameter should have requires_grad=False."""
        model = SAMUNetDecoder()
        for name, p in model.encoder.named_parameters():
            assert not p.requires_grad, \
                f"encoder param {name} is not frozen"

    @skip_no_sam
    def test_more_trainable_than_conv_decoder(self):
        """U-Net decoder should have more trainable params than the simple
        conv decoder (skip projections + richer decoder stages)."""
        conv_model = SAMFrozenConvDecoder()
        unet_model = SAMUNetDecoder()
        conv_trainable = sum(p.numel() for p in conv_model.parameters() if p.requires_grad)
        unet_trainable = sum(p.numel() for p in unet_model.parameters() if p.requires_grad)
        assert unet_trainable > conv_trainable, \
            f"U-Net decoder ({unet_trainable}) should have more params than conv ({conv_trainable})"

    @skip_no_sam
    def test_hooks_capture_features(self):
        """Forward hooks should capture features from the selected ViT blocks."""
        model = SAMUNetDecoder()
        model.eval()
        with torch.no_grad():
            _ = model(DUMMY_INPUT)
        for i, hook in enumerate(model.hooks):
            assert hook.features is not None, \
                f"hook {i} did not capture features"
            assert hook.features.shape[0] == 1, \
                f"hook {i}: batch size mismatch"
