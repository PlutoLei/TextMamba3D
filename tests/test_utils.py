"""
Comprehensive pytest tests for utils/ modules:
  - utils.metrics (dice_score, dice_score_brats_regions, hausdorff_distance_95, hausdorff_distance_95_brats_regions)
  - utils.tta (tta_predict, FLIP_AXES_8, FLIP_AXES_4)
  - utils.sliding_window (gaussian_weight_3d)
"""

import sys
import os
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `utils.*` imports resolve correctly
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.metrics import (
    BRATS_REGIONS,
    dice_score,
    dice_score_brats_regions,
    hausdorff_distance_95,
    hausdorff_distance_95_brats_regions,
)
from utils.tta import FLIP_AXES_4, FLIP_AXES_8, tta_predict
from utils.sliding_window import gaussian_weight_3d


# ===================================================================
#  1. METRICS TESTS
# ===================================================================


class TestDiceScore:
    """Tests for utils.metrics.dice_score"""

    NUM_CLASSES = 4  # background(0) + 3 foreground classes (BraTS convention)

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _make_onehot_pred(label: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Convert an integer label map (B, D, H, W) to a one-hot logit tensor
        (B, num_classes, D, H, W) where the correct class has a very large
        logit so argmax recovers the label."""
        B, D, H, W = label.shape
        logits = torch.full((B, num_classes, D, H, W), -100.0)
        label_idx = label.view(B, 1, D, H, W).long()
        logits.scatter_(1, label_idx, 100.0)
        return logits

    # -- tests ------------------------------------------------------------

    def test_perfect_prediction(self):
        """When pred == target for every voxel, all per-class dice == 1.0."""
        target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
        target[0, :2, :, :] = 1
        target[0, 2:, :2, :] = 2
        target[0, 2:, 2:, :] = 3
        pred = self._make_onehot_pred(target, self.NUM_CLASSES)
        scores = dice_score(pred, target, self.NUM_CLASSES)
        for c in range(1, self.NUM_CLASSES):
            assert scores[f"dice_class_{c}"] == pytest.approx(1.0, abs=1e-6)
        assert scores["dice_mean"] == pytest.approx(1.0, abs=1e-6)

    def test_both_empty(self):
        """If both pred and target are all-background, dice should be 1.0
        for every foreground class (BraTS convention)."""
        target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
        pred = self._make_onehot_pred(target, self.NUM_CLASSES)
        scores = dice_score(pred, target, self.NUM_CLASSES)
        for c in range(1, self.NUM_CLASSES):
            assert scores[f"dice_class_{c}"] == 1.0
        assert scores["dice_mean"] == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap(self):
        """Completely disjoint pred and target should give dice ~ 0."""
        target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
        target[0, :2, :, :] = 1  # target has class 1 in first half

        # pred puts class 1 in second half
        pred = torch.zeros(1, self.NUM_CLASSES, 4, 4, 4)
        pred[:, 0, :2, :, :] = 100.0   # background where target has 1
        pred[:, 1, 2:, :, :] = 100.0   # class 1 where target has background
        pred[:, 0, 2:, :, :] = -100.0  # suppress background in second half
        pred[:, 1, :2, :, :] = -100.0  # suppress class 1 in first half

        scores = dice_score(pred, target, self.NUM_CLASSES)
        assert scores["dice_class_1"] == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap(self):
        """Partial overlap should give 0 < dice < 1."""
        target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
        target[0, :2, :, :] = 1

        pred_logits = torch.zeros(1, self.NUM_CLASSES, 4, 4, 4)
        pred_logits[:, 0] = -100.0
        pred_logits[:, 1] = 100.0  # predict class 1 everywhere
        # class 1 predicted everywhere, target only in first half => partial overlap

        scores = dice_score(pred_logits, target, self.NUM_CLASSES)
        d = scores["dice_class_1"]
        assert 0.0 < d < 1.0

    def test_return_keys(self):
        """Returned dict should have dice_class_1 .. dice_class_(N-1) and dice_mean."""
        target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
        pred = torch.zeros(1, self.NUM_CLASSES, 4, 4, 4)
        scores = dice_score(pred, target, self.NUM_CLASSES)
        for c in range(1, self.NUM_CLASSES):
            assert f"dice_class_{c}" in scores
        assert "dice_mean" in scores

    def test_batch_dimension(self):
        """Dice should work with batch size > 1."""
        target = torch.zeros(2, 4, 4, 4, dtype=torch.long)
        target[0, :2] = 1
        target[1, 2:] = 2
        pred = self._make_onehot_pred(target, self.NUM_CLASSES)
        scores = dice_score(pred, target, self.NUM_CLASSES)
        # With perfect prediction, dice should be 1 for classes present
        assert scores["dice_class_1"] == pytest.approx(1.0, abs=1e-6)
        assert scores["dice_class_2"] == pytest.approx(1.0, abs=1e-6)


class TestDiceScoreBratsRegions:
    """Tests for utils.metrics.dice_score_brats_regions"""

    @staticmethod
    def _make_onehot_pred(label: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
        """Convert (B, D, H, W) label to (B, num_classes, D, H, W) logits."""
        B, D, H, W = label.shape
        logits = torch.full((B, num_classes, D, H, W), -100.0)
        label_idx = label.view(B, 1, D, H, W).long()
        logits.scatter_(1, label_idx, 100.0)
        return logits

    def test_perfect_prediction(self):
        """Perfect prediction => dice 1.0 for all BraTS regions."""
        target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
        target[0, 0:1, :, :] = 1  # NCR/NET
        target[0, 1:2, :, :] = 2  # ED
        target[0, 2:3, :, :] = 3  # ET
        pred = self._make_onehot_pred(target)
        scores = dice_score_brats_regions(pred, target)
        for region in BRATS_REGIONS:
            assert scores[f"dice_{region}"] == pytest.approx(1.0, abs=1e-6)
        assert scores["dice_mean"] == pytest.approx(1.0, abs=1e-6)

    def test_both_empty(self):
        """All background => dice 1.0 for every region (BraTS convention)."""
        target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
        pred = self._make_onehot_pred(target)
        scores = dice_score_brats_regions(pred, target)
        for region in BRATS_REGIONS:
            assert scores[f"dice_{region}"] == 1.0

    def test_region_keys(self):
        """Returned dict should contain dice_ET, dice_TC, dice_WT, dice_mean."""
        target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
        pred = torch.zeros(1, 4, 4, 4, 4)
        scores = dice_score_brats_regions(pred, target)
        expected_keys = {"dice_ET", "dice_TC", "dice_WT", "dice_mean"}
        assert expected_keys == set(scores.keys())

    def test_region_grouping_ET(self):
        """ET region includes only class 3."""
        target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
        target[0, :2, :, :] = 3  # half is ET

        pred_logits = torch.full((1, 4, 4, 4, 4), -100.0)
        # predict class 3 in exactly the same voxels
        pred_logits[:, 0, :2, :, :] = -100.0
        pred_logits[:, 3, :2, :, :] = 100.0
        pred_logits[:, 0, 2:, :, :] = 100.0

        scores = dice_score_brats_regions(pred_logits, target)
        assert scores["dice_ET"] == pytest.approx(1.0, abs=1e-6)

    def test_region_grouping_TC(self):
        """TC region includes classes 1 and 3."""
        target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
        target[0, :2, :, :] = 1
        target[0, 2:3, :, :] = 3
        # TC target covers slices 0-2

        pred_logits = self._make_onehot_pred(target)
        scores = dice_score_brats_regions(pred_logits, target)
        assert scores["dice_TC"] == pytest.approx(1.0, abs=1e-6)

    def test_region_grouping_WT(self):
        """WT region includes classes 1, 2, and 3."""
        target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
        target[0, 0:1, :, :] = 1
        target[0, 1:2, :, :] = 2
        target[0, 2:3, :, :] = 3
        # WT target covers slices 0-2

        pred_logits = self._make_onehot_pred(target)
        scores = dice_score_brats_regions(pred_logits, target)
        assert scores["dice_WT"] == pytest.approx(1.0, abs=1e-6)


class TestHausdorffDistance95:
    """Tests for utils.metrics.hausdorff_distance_95"""

    def test_identical_masks(self):
        """Identical non-empty masks => HD95 == 0."""
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[2:6, 2:6, 2:6] = 1
        hd = hausdorff_distance_95(mask, mask)
        assert hd == pytest.approx(0.0, abs=1e-6)

    def test_empty_pred(self):
        """Empty prediction => HD95 is NaN."""
        pred = np.zeros((8, 8, 8), dtype=np.uint8)
        target = np.ones((8, 8, 8), dtype=np.uint8)
        hd = hausdorff_distance_95(pred, target)
        assert np.isnan(hd)

    def test_empty_target(self):
        """Empty target => HD95 is NaN."""
        pred = np.ones((8, 8, 8), dtype=np.uint8)
        target = np.zeros((8, 8, 8), dtype=np.uint8)
        hd = hausdorff_distance_95(pred, target)
        assert np.isnan(hd)

    def test_both_empty(self):
        """Both empty => HD95 is NaN."""
        empty = np.zeros((8, 8, 8), dtype=np.uint8)
        hd = hausdorff_distance_95(empty, empty)
        assert np.isnan(hd)

    def test_different_masks_positive(self):
        """Non-overlapping masks => HD95 > 0."""
        pred = np.zeros((16, 16, 16), dtype=np.uint8)
        target = np.zeros((16, 16, 16), dtype=np.uint8)
        pred[:4, :4, :4] = 1
        target[12:, 12:, 12:] = 1
        hd = hausdorff_distance_95(pred, target)
        assert hd > 0.0
        assert not np.isnan(hd)

    def test_shifted_masks(self):
        """Slightly shifted masks => small positive HD95."""
        pred = np.zeros((8, 8, 8), dtype=np.uint8)
        target = np.zeros((8, 8, 8), dtype=np.uint8)
        pred[2:5, 2:5, 2:5] = 1
        target[3:6, 3:6, 3:6] = 1  # shifted by 1 voxel
        hd = hausdorff_distance_95(pred, target)
        assert 0 < hd < 10  # should be small but positive


class TestHausdorffDistance95BratsRegions:
    """Tests for utils.metrics.hausdorff_distance_95_brats_regions"""

    def test_return_keys(self):
        """Should return hd95_ET, hd95_TC, hd95_WT."""
        pred = np.zeros((8, 8, 8), dtype=np.uint8)
        target = np.zeros((8, 8, 8), dtype=np.uint8)
        scores = hausdorff_distance_95_brats_regions(pred, target)
        expected = {"hd95_ET", "hd95_TC", "hd95_WT"}
        assert expected == set(scores.keys())

    def test_identical_segmentations(self):
        """Identical multi-class segmentations => hd95 == 0 for each region."""
        seg = np.zeros((8, 8, 8), dtype=np.uint8)
        seg[0:2, :, :] = 1
        seg[2:4, :, :] = 2
        seg[4:6, :, :] = 3
        scores = hausdorff_distance_95_brats_regions(seg, seg)
        for region in BRATS_REGIONS:
            assert scores[f"hd95_{region}"] == pytest.approx(0.0, abs=1e-6)

    def test_both_empty_returns_nan(self):
        """All background => NaN for all regions."""
        empty = np.zeros((8, 8, 8), dtype=np.uint8)
        scores = hausdorff_distance_95_brats_regions(empty, empty)
        for region in BRATS_REGIONS:
            assert np.isnan(scores[f"hd95_{region}"])

    def test_region_class_mapping(self):
        """Verify regions correctly aggregate classes."""
        pred = np.zeros((8, 8, 8), dtype=np.uint8)
        target = np.zeros((8, 8, 8), dtype=np.uint8)
        # Only class 3 present in both
        pred[2:6, 2:6, 2:6] = 3
        target[2:6, 2:6, 2:6] = 3
        scores = hausdorff_distance_95_brats_regions(pred, target)
        # ET=[3], TC=[1,3], WT=[1,2,3] — class 3 contributes to all three
        assert scores["hd95_ET"] == pytest.approx(0.0, abs=1e-6)
        assert scores["hd95_TC"] == pytest.approx(0.0, abs=1e-6)
        assert scores["hd95_WT"] == pytest.approx(0.0, abs=1e-6)


# ===================================================================
#  2. TTA TESTS
# ===================================================================


class DummyModel(nn.Module):
    """A trivial model that returns the input image expanded to `num_classes`
    channels via a 1x1x1x1 convolution-like broadcast. This lets us verify
    TTA flipping/averaging logic without needing real weights.

    Input:  (B, C_in, D, H, W)
    Output: (B, num_classes, D, H, W)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 4):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, num_classes, kernel_size=1, bias=False)
        # Initialize to identity-like: each output channel copies input channel 0
        nn.init.constant_(self.conv.weight, 1.0)

    def forward(self, image, text_ids=None, attention_mask=None, use_text=True):
        return self.conv(image)


class IdentityModel(nn.Module):
    """Even simpler: returns input unchanged (requires in_channels == num_classes)."""

    def forward(self, image, text_ids=None, attention_mask=None, use_text=True):
        return image


class TestTTAPredict:
    """Tests for utils.tta.tta_predict"""

    def test_output_shape_flip8(self):
        """Output shape should match (B, C, D, H, W) with num_flips=8."""
        model = DummyModel(in_channels=1, num_classes=4)
        model.eval()
        image = torch.randn(1, 1, 8, 8, 8)
        out = tta_predict(model, image, num_flips=8)
        assert out.shape == (1, 4, 8, 8, 8)

    def test_output_shape_flip4(self):
        """Output shape should match (B, C, D, H, W) with num_flips=4."""
        model = DummyModel(in_channels=1, num_classes=4)
        model.eval()
        image = torch.randn(1, 1, 8, 8, 8)
        out = tta_predict(model, image, num_flips=4)
        assert out.shape == (1, 4, 8, 8, 8)

    def test_invalid_num_flips(self):
        """Invalid num_flips should raise ValueError."""
        model = DummyModel()
        image = torch.randn(1, 1, 4, 4, 4)
        with pytest.raises(ValueError, match="num_flips must be 4 or 8"):
            tta_predict(model, image, num_flips=3)

    def test_invalid_num_flips_16(self):
        """Another invalid value."""
        model = DummyModel()
        image = torch.randn(1, 1, 4, 4, 4)
        with pytest.raises(ValueError):
            tta_predict(model, image, num_flips=16)

    def test_output_is_probability(self):
        """Output should be valid softmax probabilities: sum to 1 along dim=1,
        all values >= 0."""
        model = DummyModel(in_channels=1, num_classes=4)
        model.eval()
        image = torch.randn(1, 1, 6, 6, 6)
        with torch.no_grad():
            out = tta_predict(model, image, num_flips=8)
        # All values non-negative (average of softmax outputs)
        assert (out >= 0).all()
        # Sum along class dim should be ~1.0 at every voxel
        channel_sum = out.sum(dim=1)
        assert torch.allclose(channel_sum, torch.ones_like(channel_sum), atol=1e-5)

    def test_output_is_probability_flip4(self):
        """Same probability check for num_flips=4."""
        model = DummyModel(in_channels=1, num_classes=4)
        model.eval()
        image = torch.randn(1, 1, 6, 6, 6)
        with torch.no_grad():
            out = tta_predict(model, image, num_flips=4)
        assert (out >= 0).all()
        channel_sum = out.sum(dim=1)
        assert torch.allclose(channel_sum, torch.ones_like(channel_sum), atol=1e-5)

    def test_symmetric_model_invariance(self):
        """For a model whose output is invariant to flips (e.g. constant output),
        TTA should return the same result as a single forward pass."""
        model = DummyModel(in_channels=1, num_classes=4)
        # Make the conv produce a constant output regardless of input pattern
        nn.init.constant_(model.conv.weight, 0.0)
        model.eval()

        image = torch.randn(1, 1, 4, 4, 4)
        with torch.no_grad():
            single = F.softmax(model(image), dim=1)
            tta_out = tta_predict(model, image, num_flips=8)
        assert torch.allclose(single, tta_out, atol=1e-5)

    def test_text_args_forwarded(self):
        """Verify text_ids and attention_mask are forwarded to the model."""

        class RecordingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.recorded_text_ids = []
                self.recorded_masks = []
                self.recorded_use_text = []

            def forward(self, image, text_ids=None, attention_mask=None, use_text=True):
                self.recorded_text_ids.append(text_ids)
                self.recorded_masks.append(attention_mask)
                self.recorded_use_text.append(use_text)
                return torch.zeros(image.shape[0], 4, *image.shape[2:])

        model = RecordingModel()
        image = torch.randn(1, 1, 4, 4, 4)
        text_ids = torch.tensor([[1, 2, 3]])
        attn_mask = torch.tensor([[1, 1, 1]])

        tta_predict(model, image, text_ids=text_ids, attention_mask=attn_mask, use_text=False, num_flips=4)

        # Should have been called 4 times (num_flips=4 => 4 augmentations)
        assert len(model.recorded_text_ids) == 4
        for tid in model.recorded_text_ids:
            assert torch.equal(tid, text_ids)
        for mask in model.recorded_masks:
            assert torch.equal(mask, attn_mask)
        for ut in model.recorded_use_text:
            assert ut is False

    def test_flip_axes_constants(self):
        """Verify FLIP_AXES_8 has 8 entries and FLIP_AXES_4 has 4 entries."""
        assert len(FLIP_AXES_8) == 8
        assert len(FLIP_AXES_4) == 4
        # First entry in both should be no-flip (empty list)
        assert FLIP_AXES_8[0] == []
        assert FLIP_AXES_4[0] == []

    def test_non_cubic_input(self):
        """TTA should work with non-cubic spatial dimensions."""
        model = DummyModel(in_channels=1, num_classes=2)
        model.eval()
        image = torch.randn(1, 1, 4, 6, 8)
        with torch.no_grad():
            out = tta_predict(model, image, num_flips=8)
        assert out.shape == (1, 2, 4, 6, 8)


# ===================================================================
#  3. SLIDING WINDOW (GAUSSIAN WEIGHT) TESTS
# ===================================================================


class TestGaussianWeight3D:
    """Tests for utils.sliding_window.gaussian_weight_3d"""

    def test_output_shape_cubic(self):
        """Output shape should match the requested cubic patch_size."""
        patch = (16, 16, 16)
        w = gaussian_weight_3d(patch)
        assert w.shape == torch.Size(patch)

    def test_output_shape_non_cubic(self):
        """Output shape should match non-cubic patch_size."""
        patch = (8, 12, 16)
        w = gaussian_weight_3d(patch)
        assert w.shape == torch.Size(patch)

    def test_output_shape_small(self):
        """Works with very small patch sizes."""
        patch = (2, 2, 2)
        w = gaussian_weight_3d(patch)
        assert w.shape == torch.Size(patch)

    def test_center_value_is_one(self):
        """The center voxel should have value 1.0 (the maximum)."""
        patch = (16, 16, 16)
        w = gaussian_weight_3d(patch)
        center = tuple(s // 2 for s in patch)
        assert w[center].item() == pytest.approx(1.0, abs=1e-6)

    def test_center_value_odd_size(self):
        """Center value is 1.0 for odd-sized patches."""
        patch = (15, 15, 15)
        w = gaussian_weight_3d(patch)
        center = tuple(s // 2 for s in patch)
        assert w[center].item() == pytest.approx(1.0, abs=1e-6)

    def test_center_value_non_cubic(self):
        """Center value is 1.0 for non-cubic patches."""
        patch = (8, 12, 16)
        w = gaussian_weight_3d(patch)
        center = tuple(s // 2 for s in patch)
        # For even sizes, center is at s//2 which may not be exactly (s-1)/2.
        # The maximum should still be 1.0 due to normalization.
        assert w.max().item() == pytest.approx(1.0, abs=1e-6)

    def test_corner_values_less_than_center(self):
        """Corner voxels should have values strictly less than the center."""
        patch = (16, 16, 16)
        w = gaussian_weight_3d(patch)
        center_val = w[8, 8, 8].item()
        corners = [
            w[0, 0, 0].item(),
            w[0, 0, -1].item(),
            w[0, -1, 0].item(),
            w[-1, 0, 0].item(),
            w[-1, -1, -1].item(),
        ]
        for corner in corners:
            assert corner < center_val

    def test_values_in_valid_range(self):
        """All values should be in [1e-4, 1.0]."""
        patch = (16, 16, 16)
        w = gaussian_weight_3d(patch)
        assert w.min().item() >= 1e-4 - 1e-8  # small tolerance for float
        assert w.max().item() <= 1.0 + 1e-8

    def test_values_in_valid_range_non_cubic(self):
        """Valid range check for non-cubic patches."""
        patch = (8, 12, 20)
        w = gaussian_weight_3d(patch)
        assert w.min().item() >= 1e-4 - 1e-8
        assert w.max().item() <= 1.0 + 1e-8

    def test_symmetry(self):
        """The Gaussian weight should be symmetric about the center."""
        patch = (16, 16, 16)
        w = gaussian_weight_3d(patch)
        # Flip along each axis and compare
        assert torch.allclose(w, w.flip(0), atol=1e-6)
        assert torch.allclose(w, w.flip(1), atol=1e-6)
        assert torch.allclose(w, w.flip(2), atol=1e-6)

    def test_dtype_is_float32(self):
        """Output should be float32 tensor."""
        w = gaussian_weight_3d((8, 8, 8))
        assert w.dtype == torch.float32

    def test_sigma_scale_affects_output(self):
        """Different sigma_scale values should produce different weights."""
        patch = (16, 16, 16)
        w1 = gaussian_weight_3d(patch, sigma_scale=0.125)
        w2 = gaussian_weight_3d(patch, sigma_scale=0.25)
        # Both should have max == 1.0
        assert w1.max().item() == pytest.approx(1.0, abs=1e-6)
        assert w2.max().item() == pytest.approx(1.0, abs=1e-6)
        # But corner values should differ (larger sigma => flatter => larger corners)
        assert w2[0, 0, 0].item() > w1[0, 0, 0].item()

    def test_monotonic_decay_from_center(self):
        """Values should decrease (or stay equal) as we move from center to edge
        along any axis."""
        patch = (16, 16, 16)
        w = gaussian_weight_3d(patch)
        c = 8  # center index
        # Along axis 0, holding other dims at center
        for i in range(c, patch[0] - 1):
            assert w[i, c, c].item() >= w[i + 1, c, c].item() - 1e-7

    def test_large_patch_size(self):
        """Should work with larger patch sizes without error."""
        patch = (64, 64, 64)
        w = gaussian_weight_3d(patch)
        assert w.shape == torch.Size(patch)
        assert w.max().item() == pytest.approx(1.0, abs=1e-6)
        assert w.min().item() >= 1e-4 - 1e-8
