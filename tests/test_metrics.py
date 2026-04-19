"""
tests/test_metrics.py — Unit tests for IoU and F1 metrics.

Hand-crafted test cases with known expected values verify that the metric
implementations are correct. Tests use small 4x4 tensors where the
intersection, union, and sums can be computed by hand.
"""

import torch
import pytest

from src.metrics import iou_score, f1_score


class TestIoU:
    """Tests for the iou_score function."""

    def test_perfect_prediction(self):
        """IoU should be ~1.0 when prediction matches target perfectly."""
        # Large positive logits → sigmoid ≈ 1.0 → matches target = 1.0
        pred = torch.full((1, 1, 4, 4), 10.0)
        target = torch.ones(1, 1, 4, 4)
        iou = iou_score(pred, target)
        assert abs(iou - 1.0) < 1e-4, f"expected ~1.0, got {iou}"

    def test_perfect_negative(self):
        """IoU should be ~1.0 when both pred and target are all-zero.

        The eps term prevents 0/0, so IoU = eps/eps ≈ 1.0 for the case
        where there are no positives in either prediction or target.
        """
        pred = torch.full((1, 1, 4, 4), -10.0)  # sigmoid ≈ 0
        target = torch.zeros(1, 1, 4, 4)
        iou = iou_score(pred, target)
        assert iou > 0.99, f"expected ~1.0 for true-negative case, got {iou}"

    def test_no_overlap(self):
        """IoU should be near 0 when prediction and target don't overlap."""
        # Top-left half predicted, bottom-right half is target
        pred = torch.full((1, 1, 4, 4), -10.0)
        pred[0, 0, :2, :] = 10.0   # predict top two rows
        target = torch.zeros(1, 1, 4, 4)
        target[0, 0, 2:, :] = 1.0  # target bottom two rows
        iou = iou_score(pred, target)
        assert iou < 0.01, f"expected ~0 for no overlap, got {iou}"

    def test_half_overlap(self):
        """IoU should match hand-computed value for partial overlap.

        pred:   top 3 rows positive (12 pixels)
        target: bottom 3 rows positive (12 pixels)
        intersection: middle 2 rows (8 pixels)
        union: all 4 rows (16 pixels)
        IoU = 8/16 = 0.5
        """
        pred = torch.full((1, 1, 4, 4), -10.0)
        pred[0, 0, :3, :] = 10.0
        target = torch.zeros(1, 1, 4, 4)
        target[0, 0, 1:, :] = 1.0
        iou = iou_score(pred, target)
        assert abs(iou - 0.5) < 0.01, f"expected ~0.5, got {iou}"


class TestF1:
    """Tests for the f1_score function."""

    def test_perfect_prediction(self):
        """F1 should be ~1.0 when prediction matches target perfectly."""
        pred = torch.full((1, 1, 4, 4), 10.0)
        target = torch.ones(1, 1, 4, 4)
        f1 = f1_score(pred, target)
        assert abs(f1 - 1.0) < 1e-4, f"expected ~1.0, got {f1}"

    def test_no_overlap(self):
        """F1 should be near 0 when prediction and target don't overlap."""
        pred = torch.full((1, 1, 4, 4), -10.0)
        pred[0, 0, :2, :] = 10.0
        target = torch.zeros(1, 1, 4, 4)
        target[0, 0, 2:, :] = 1.0
        f1 = f1_score(pred, target)
        assert f1 < 0.01, f"expected ~0, got {f1}"

    def test_half_overlap(self):
        """F1 should match hand-computed value for partial overlap.

        Same setup as IoU half-overlap test:
        intersection = 8, pred_sum = 12, target_sum = 12
        F1 = 2*8 / (12 + 12) = 16/24 = 0.6667
        """
        pred = torch.full((1, 1, 4, 4), -10.0)
        pred[0, 0, :3, :] = 10.0
        target = torch.zeros(1, 1, 4, 4)
        target[0, 0, 1:, :] = 1.0
        f1 = f1_score(pred, target)
        assert abs(f1 - 2 / 3) < 0.01, f"expected ~0.667, got {f1}"

    def test_iou_f1_relationship(self):
        """F1 should always be >= IoU for the same prediction.

        This is a mathematical property: F1 = 2*IoU / (1 + IoU).
        """
        pred = torch.randn(2, 1, 8, 8)
        target = (torch.randn(2, 1, 8, 8) > 0).float()
        iou = iou_score(pred, target)
        f1 = f1_score(pred, target)
        assert f1 >= iou - 1e-6, f"F1 ({f1}) < IoU ({iou}) — violates relationship"
