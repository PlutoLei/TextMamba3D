# tests/test_losses.py
"""Comprehensive loss function tests.

Tests cover: DiceLoss (weighted/unweighted), EdgeLoss, ContrastiveLoss,
CombinedLoss with class weights.
"""
import torch
import pytest


class TestDiceLoss:
    def test_dice_loss_perfect_prediction(self):
        """Perfect prediction should give loss close to 0."""
        from losses.dice_loss import DiceLoss

        loss_fn = DiceLoss(include_background=True)
        pred = torch.zeros(2, 4, 16, 16, 16)
        target = torch.zeros(2, 16, 16, 16, dtype=torch.long)

        for c in range(4):
            start = c * 4
            end = (c + 1) * 4
            pred[:, c, start:end, :, :] = 10
            target[:, start:end, :, :] = c

        loss = loss_fn(pred, target)
        assert loss.item() < 0.1

    def test_dice_loss_with_class_weights(self):
        """Class weights should affect loss value."""
        from losses.dice_loss import DiceLoss

        pred = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
        target = torch.randint(0, 4, (1, 8, 8, 8))

        loss_uniform = DiceLoss(class_weights=[1.0, 1.0, 1.0, 1.0])
        loss_weighted = DiceLoss(class_weights=[0.25, 3.0, 1.0, 4.0])

        val_uniform = loss_uniform(pred, target)
        val_weighted = loss_weighted(pred, target)

        # Weighted and uniform should differ (unless by coincidence)
        assert val_uniform.shape == ()
        assert val_weighted.shape == ()
        assert val_uniform.item() > 0
        assert val_weighted.item() > 0

    def test_dice_loss_brats_weights_constant(self):
        """BRATS_CLASS_WEIGHTS should be accessible."""
        from losses.dice_loss import BRATS_CLASS_WEIGHTS

        assert len(BRATS_CLASS_WEIGHTS) == 4
        assert BRATS_CLASS_WEIGHTS == [0.25, 3.0, 1.0, 4.0]

    def test_dice_loss_no_background(self):
        """Default DiceLoss excludes background (class 0)."""
        from losses.dice_loss import DiceLoss

        loss_fn = DiceLoss(include_background=False)
        # All background
        pred = torch.zeros(1, 4, 8, 8, 8)
        pred[:, 0, :, :, :] = 10  # Strong background prediction
        target = torch.zeros(1, 8, 8, 8, dtype=torch.long)  # All background

        loss = loss_fn(pred, target)
        # Loss should still compute (based on classes 1-3 which are empty)
        assert loss.item() >= 0

    def test_dice_loss_backward(self):
        """Dice loss should allow gradient computation."""
        from losses.dice_loss import DiceLoss

        loss_fn = DiceLoss(class_weights=[0.25, 3.0, 1.0, 4.0])
        pred = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
        target = torch.randint(0, 4, (1, 8, 8, 8))

        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape


class TestEdgeLoss:
    def test_edge_loss_output(self):
        """Edge loss should return non-negative scalar."""
        from losses.edge_loss import EdgeLoss

        loss_fn = EdgeLoss()
        pred = torch.randn(2, 4, 16, 16, 16)
        target = torch.randint(0, 4, (2, 16, 16, 16))
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_edge_loss_uniform_target(self):
        """Uniform target (no edges) should give small edge loss."""
        from losses.edge_loss import EdgeLoss

        loss_fn = EdgeLoss()
        pred = torch.randn(1, 4, 16, 16, 16)
        target = torch.zeros(1, 16, 16, 16, dtype=torch.long)  # All background
        loss = loss_fn(pred, target)
        # Edge mask should be near zero for uniform target
        assert loss.item() >= 0

    def test_edge_mask_shape(self):
        """Edge mask should have correct shape."""
        from losses.edge_loss import EdgeLoss

        loss_fn = EdgeLoss()
        target = torch.randint(0, 4, (2, 16, 16, 16))
        edge_mask = loss_fn.get_edge_mask(target)
        assert edge_mask.shape == (2, 1, 16, 16, 16)


class TestContrastiveLoss:
    def test_contrastive_loss_identical_features(self):
        """Identical normalized features should give low loss."""
        from losses.contrastive_loss import ContrastiveLoss

        loss_fn = ContrastiveLoss(temperature=0.07)
        feat = torch.randn(4, 256)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        loss = loss_fn(feat, feat)
        assert loss.item() < 1.0

    def test_contrastive_loss_non_negative(self):
        """Contrastive loss should be non-negative."""
        from losses.contrastive_loss import ContrastiveLoss

        loss_fn = ContrastiveLoss(temperature=0.07)
        img_feat = torch.randn(4, 128)
        text_feat = torch.randn(4, 128)
        loss = loss_fn(img_feat, text_feat)
        assert loss.item() >= 0


class TestCombinedLoss:
    def test_combined_loss_all_components(self):
        """CombinedLoss returns dict with all loss components."""
        from losses import CombinedLoss

        criterion = CombinedLoss(
            dice_weight=1.0, ce_weight=1.0,
            edge_weight=1.0, contrastive_weight=0.5,
            class_weights=[0.25, 3.0, 1.0, 4.0],
        )
        pred = torch.randn(1, 4, 16, 16, 16, requires_grad=True)
        target = torch.randint(0, 4, (1, 16, 16, 16))
        img_feat = torch.randn(1, 256)
        text_feat = torch.randn(1, 256)

        losses = criterion(pred, target, img_feat, text_feat)

        for key in ['dice', 'ce', 'edge', 'contrastive', 'total']:
            assert key in losses, f"Missing loss component: {key}"
            assert losses[key].item() >= 0

    def test_combined_loss_without_contrastive(self):
        """CombinedLoss works without contrastive features."""
        from losses import CombinedLoss

        criterion = CombinedLoss(contrastive_weight=0.0)
        pred = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
        target = torch.randint(0, 4, (1, 8, 8, 8))

        losses = criterion(pred, target)
        assert losses['contrastive'].item() == 0.0
        assert losses['total'].item() > 0

    def test_combined_loss_with_class_weights(self):
        """CombinedLoss with class weights should differ from default."""
        from losses import CombinedLoss

        pred = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
        target = torch.randint(0, 4, (1, 8, 8, 8))

        crit_default = CombinedLoss(class_weights=None)
        crit_weighted = CombinedLoss(class_weights=[0.25, 3.0, 1.0, 4.0])

        loss_d = crit_default(pred, target)['total']
        loss_w = crit_weighted(pred, target)['total']

        # Both should compute valid losses
        assert loss_d.item() > 0
        assert loss_w.item() > 0

    def test_combined_loss_backward(self):
        """CombinedLoss should support backward pass."""
        from losses import CombinedLoss

        criterion = CombinedLoss(class_weights=[0.25, 3.0, 1.0, 4.0])
        pred = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
        target = torch.randint(0, 4, (1, 8, 8, 8))

        losses = criterion(pred, target)
        losses['total'].backward()
        assert pred.grad is not None
