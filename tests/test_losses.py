"""
tests/test_losses.py — Unit tests for BCEDiceLoss.

Tests verify the loss contract (scalar output, gradient flow, correct
values on hand-crafted inputs) rather than training convergence.
"""

import torch
import pytest

from src.losses import BCEDiceLoss


class TestBCEDiceLoss:

    def test_returns_scalar(self):
        """Loss should be a 0-dim tensor."""
        loss_fn = BCEDiceLoss()
        pred = torch.randn(2, 1, 4, 4)
        target = torch.ones(2, 1, 4, 4)
        loss = loss_fn(pred, target)
        assert loss.dim() == 0, f"expected scalar, got shape {loss.shape}"

    def test_positive_loss(self):
        """Loss should be > 0 for non-perfect predictions."""
        loss_fn = BCEDiceLoss()
        pred = torch.randn(2, 1, 4, 4)
        target = torch.ones(2, 1, 4, 4)
        loss = loss_fn(pred, target)
        assert loss.item() > 0, "loss should be positive for random predictions"

    def test_gradients_flow(self):
        """Gradients should flow back to the input tensor."""
        loss_fn = BCEDiceLoss()
        pred = torch.randn(2, 1, 4, 4, requires_grad=True)
        target = torch.ones(2, 1, 4, 4)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None, "no gradient on pred"
        assert not torch.isnan(pred.grad).any(), "gradient contains NaN"

    def test_perfect_prediction_low_loss(self):
        """Loss should be near zero when prediction perfectly matches target.

        Using large positive logits (pre-sigmoid) for the positive class
        gives sigmoid -> ~1.0, which matches a target of 1.0.
        """
        loss_fn = BCEDiceLoss()
        pred = torch.full((1, 1, 4, 4), 10.0)   # sigmoid(10) ≈ 1.0
        target = torch.ones(1, 1, 4, 4)
        loss = loss_fn(pred, target)
        assert loss.item() < 0.01, f"expected near-zero loss, got {loss.item()}"

    def test_hand_computed_all_ones(self):
        """Verify BCE+Dice matches hand-computed values on a trivial case.

        Setup: 2x2 image, all logits = 0, all targets = 1.
          sigmoid(0) = 0.5 everywhere

          BCE = -log(0.5) = 0.6931...
          Dice_intersection = 4 * 0.5 = 2.0
          Dice = 1 - (2*2.0 + 1) / (4*0.5 + 4*1.0 + 1) = 1 - 5/7 = 2/7
          Combined = 0.5 * 0.6931 + 0.5 * (2/7) ≈ 0.4894
        """
        loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, smooth=1.0)
        pred = torch.zeros(1, 1, 2, 2)
        target = torch.ones(1, 1, 2, 2)
        loss = loss_fn(pred, target)

        import math
        bce = -math.log(0.5)                    # 0.6931
        dice = 1.0 - (2 * 2.0 + 1) / (2.0 + 4.0 + 1)  # 2/7
        expected = 0.5 * bce + 0.5 * dice

        assert abs(loss.item() - expected) < 1e-4, \
            f"expected {expected:.6f}, got {loss.item():.6f}"

    def test_custom_weights(self):
        """Different BCE/Dice weights should produce different losses."""
        pred = torch.randn(2, 1, 4, 4)
        target = torch.ones(2, 1, 4, 4)

        loss_equal = BCEDiceLoss(0.5, 0.5)(pred, target).item()
        loss_bce_heavy = BCEDiceLoss(0.8, 0.2)(pred, target).item()
        loss_dice_heavy = BCEDiceLoss(0.2, 0.8)(pred, target).item()

        # Unless BCE and Dice happen to be exactly equal (astronomically
        # unlikely for random predictions), the three losses should differ.
        assert not (loss_equal == loss_bce_heavy == loss_dice_heavy), \
            "all weight combos gave the same loss — suspicious"

    def test_no_nans_on_empty_target(self):
        """Loss should not produce NaN when target is all-zero.

        This can happen if a training patch has no buildings after
        augmentation. The smooth term in Dice prevents 0/0.
        """
        loss_fn = BCEDiceLoss()
        pred = torch.randn(1, 1, 4, 4)
        target = torch.zeros(1, 1, 4, 4)
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss), "loss is NaN on all-zero target"
