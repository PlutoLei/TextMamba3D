"""
Integration tests for TextMamba3D.

Covers:
  - EarlyStopping logic (from train.py)
  - End-to-end forward/backward pipeline
  - Inference normalization helpers

All volumes use 16^3 spatial size to keep tests fast.
"""

import sys
import os

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so bare imports work.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train import EarlyStopping
from models.textmamba3d import TextMamba3D
from losses import CombinedLoss
from data.transforms import get_train_transforms, get_val_transforms


# ========================== fixtures ========================================

SMALL_IMG_SIZE = (16, 16, 16)
IN_CHANNELS = 4
OUT_CHANNELS = 4
EMBED_DIM = 32
DEPTHS = [1, 1]
TEXT_EMBED_DIM = 64
TEXT_DEPTH = 1
BATCH = 1


@pytest.fixture()
def small_model():
    """Return a lightweight TextMamba3D on CPU."""
    model = TextMamba3D(
        img_size=SMALL_IMG_SIZE,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        embed_dim=EMBED_DIM,
        depths=DEPTHS,
        text_embed_dim=TEXT_EMBED_DIM,
        text_depth=TEXT_DEPTH,
        use_pretrained_text=False,
    )
    model.train()
    return model


@pytest.fixture()
def criterion():
    """Return the combined loss used in training."""
    return CombinedLoss(
        dice_weight=1.0,
        ce_weight=1.0,
        edge_weight=1.0,
        contrastive_weight=0.5,
        class_weights=[0.25, 3.0, 1.0, 4.0],
    )


@pytest.fixture()
def random_batch():
    """Produce a (image, mask) pair with correct shapes."""
    image = torch.randn(BATCH, IN_CHANNELS, *SMALL_IMG_SIZE)
    mask = torch.randint(0, OUT_CHANNELS, (BATCH, *SMALL_IMG_SIZE))
    return image, mask


@pytest.fixture()
def random_batch_with_text():
    """Produce (image, mask, text_ids) for text-conditioned forward."""
    image = torch.randn(BATCH, IN_CHANNELS, *SMALL_IMG_SIZE)
    mask = torch.randint(0, OUT_CHANNELS, (BATCH, *SMALL_IMG_SIZE))
    text_ids = torch.randint(0, 1000, (BATCH, 12))
    return image, mask, text_ids


# ========================== TestEarlyStopping ===============================


class TestEarlyStopping:
    """Unit tests for the EarlyStopping helper in train.py."""

    def test_first_call_never_stops(self):
        """The very first call should record best_score and return False."""
        es = EarlyStopping(patience=5, min_delta=0.001, mode="max")
        assert es(0.50) is False
        assert es.best_score == 0.50
        assert es.counter == 0

    def test_improvement_resets_counter(self):
        """A clear improvement should keep counter at zero."""
        es = EarlyStopping(patience=3, min_delta=0.001, mode="max")
        es(0.50)
        # Each call strictly improves
        assert es(0.60) is False
        assert es.counter == 0
        assert es(0.70) is False
        assert es.counter == 0

    def test_patience_triggers_stop(self):
        """After `patience` calls without improvement, early_stop == True."""
        patience = 4
        es = EarlyStopping(patience=patience, min_delta=0.001, mode="max")
        es(0.80)  # baseline

        for i in range(patience - 1):
            result = es(0.80)  # no improvement
            assert result is False, f"Should not stop at call {i + 1}"

        # The patience-th stagnant call should trigger
        assert es(0.80) is True
        assert es.early_stop is True

    def test_min_delta_threshold(self):
        """Improvement smaller than min_delta is not counted."""
        es = EarlyStopping(patience=3, min_delta=0.01, mode="max")
        es(0.80)

        # Tiny improvement that is below min_delta
        assert es(0.8005) is False
        assert es.counter == 1, "Improvement below min_delta should increment counter"

        # Another tiny improvement
        assert es(0.801) is False
        assert es.counter == 2

        # Third stagnant call triggers stop (patience=3)
        assert es(0.801) is True

    def test_mode_min(self):
        """mode='min' treats lower scores as improvements."""
        es = EarlyStopping(patience=3, min_delta=0.001, mode="min")
        es(1.00)

        # Decrease is improvement in min mode
        assert es(0.90) is False
        assert es.best_score == 0.90
        assert es.counter == 0

        # Increase is not improvement
        assert es(0.95) is False
        assert es.counter == 1

    def test_counter_reset_on_improvement(self):
        """After several stagnant calls, a real improvement resets counter."""
        es = EarlyStopping(patience=5, min_delta=0.001, mode="max")
        es(0.70)

        # Stagnate for 3 calls
        es(0.70)
        es(0.70)
        es(0.70)
        assert es.counter == 3

        # Now improve
        assert es(0.80) is False
        assert es.counter == 0
        assert es.best_score == 0.80


# ========================== TestEndToEndPipeline ============================


class TestEndToEndPipeline:
    """Integration tests that exercise model + loss + transforms together."""

    def test_forward_backward_with_text(self, small_model, criterion, random_batch_with_text):
        """Forward pass with text conditioning, then backward through combined loss."""
        image, mask, text_ids = random_batch_with_text
        logits = small_model(image, text_ids, use_text=True)

        assert logits.shape == (BATCH, OUT_CHANNELS, *SMALL_IMG_SIZE)

        losses = criterion(logits, mask)
        losses['total'].backward()

        grads_found = sum(
            1 for p in small_model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grads_found > 0, "No gradients found after backward"

    def test_forward_backward_without_text(self, small_model, criterion, random_batch):
        """Forward + backward without text conditioning."""
        image, mask = random_batch
        logits = small_model(image, text_ids=None, use_text=False)

        assert logits.shape == (BATCH, OUT_CHANNELS, *SMALL_IMG_SIZE)

        losses = criterion(logits, mask)
        losses['total'].backward()

        grads_found = sum(
            1 for p in small_model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grads_found > 0, "No gradients found after backward"

    def test_transforms_then_model(self, small_model):
        """Apply training transforms to synthetic data, then feed through model."""
        train_tf = get_train_transforms(SMALL_IMG_SIZE)

        # Transforms expect (image: Tensor[C,D,H,W], mask: Tensor[D,H,W])
        image = torch.randn(IN_CHANNELS, 32, 32, 32)
        mask = torch.randint(0, OUT_CHANNELS, (32, 32, 32))

        image_t, mask_t = train_tf(image, mask)

        assert image_t.shape[-3:] == SMALL_IMG_SIZE

        image_t = image_t.unsqueeze(0).float()
        with torch.no_grad():
            logits = small_model(image_t, use_text=False)
        assert logits.shape == (1, OUT_CHANNELS, *SMALL_IMG_SIZE)

    def test_val_transforms_deterministic(self):
        """Validation transforms should produce identical output for identical input."""
        val_tf = get_val_transforms(SMALL_IMG_SIZE)

        image = torch.randn(IN_CHANNELS, 32, 32, 32)
        mask = torch.randint(0, OUT_CHANNELS, (32, 32, 32))

        img1, mask1 = val_tf(image.clone(), mask.clone())
        img2, mask2 = val_tf(image.clone(), mask.clone())

        assert torch.equal(img1, img2), "Val transforms should be deterministic"
        assert torch.equal(mask1, mask2), "Val transforms should be deterministic"

    def test_full_pipeline_train_step(self, small_model, criterion, random_batch_with_text):
        """Simulate a complete training step: forward, loss, backward, optimizer.step."""
        image, mask, text_ids = random_batch_with_text
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)

        # Snapshot all trainable parameters
        snapshots = {
            name: p.clone().detach()
            for name, p in small_model.named_parameters()
            if p.requires_grad
        }

        optimizer.zero_grad()
        logits = small_model(image, text_ids, use_text=True)
        losses = criterion(logits, mask)
        losses['total'].backward()
        optimizer.step()

        # At least some parameters should have changed
        changed = sum(
            1 for name, p in small_model.named_parameters()
            if p.requires_grad and not torch.equal(snapshots[name], p.detach())
        )
        assert changed > 0, "No parameters changed after optimizer step"
        assert torch.isfinite(losses['total']), f"Loss is not finite: {losses['total'].item()}"


# ========================== TestInferenceNormalization ======================


class TestInferenceNormalization:
    """Tests for the per-channel z-score normalization used during inference.

    The normalization logic is reproduced here to decouple tests from the
    full InferenceDataLoader (which requires NIfTI file paths).
    """

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        """Replicate InferenceDataLoader._normalize exactly."""
        image = image.copy()
        for i in range(image.shape[0]):
            nonzero = image[i][image[i] > 0]
            if len(nonzero) > 0:
                image[i] = (image[i] - nonzero.mean()) / (nonzero.std() + 1e-8)
        return image

    def test_zscore_normalization(self):
        """After normalization, the non-zero voxels should be roughly mean-0 std-1."""
        rng = np.random.RandomState(42)
        # Create an image with positive values only (simulating brain intensities)
        image = rng.uniform(100, 200, size=(IN_CHANNELS, 16, 16, 16)).astype(np.float32)

        normed = self._normalize(image)

        for ch in range(IN_CHANNELS):
            vals = normed[ch][normed[ch] != 0]
            if len(vals) > 0:
                assert abs(vals.mean()) < 0.15, f"Channel {ch} mean far from 0: {vals.mean():.4f}"
                assert abs(vals.std() - 1.0) < 0.15, f"Channel {ch} std far from 1: {vals.std():.4f}"

    def test_all_zero_channel(self):
        """A channel that is entirely zero should remain zero after normalization."""
        image = np.zeros((IN_CHANNELS, 16, 16, 16), dtype=np.float32)
        # Only fill channels 1 and 2 with nonzero data
        image[1] = np.random.uniform(50, 150, size=(16, 16, 16)).astype(np.float32)
        image[2] = np.random.uniform(50, 150, size=(16, 16, 16)).astype(np.float32)

        normed = self._normalize(image)

        # Channel 0 and 3 were all-zero and should stay zero
        np.testing.assert_array_equal(normed[0], 0.0)
        np.testing.assert_array_equal(normed[3], 0.0)

    def test_normalization_shape_preserved(self):
        """Output shape must exactly match input shape."""
        shapes = [
            (IN_CHANNELS, 16, 16, 16),
            (1, 8, 8, 8),
            (3, 32, 32, 32),
        ]
        for shape in shapes:
            image = np.random.uniform(10, 100, size=shape).astype(np.float32)
            normed = self._normalize(image)
            assert normed.shape == shape, f"Shape mismatch: {normed.shape} != {shape}"
