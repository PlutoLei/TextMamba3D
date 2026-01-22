# tests/test_losses.py
import torch
import pytest


class TestDiceLoss:
    def test_dice_loss_perfect_prediction(self):
        """Perfect prediction should give loss close to 0."""
        from losses.dice_loss import DiceLoss

        loss_fn = DiceLoss(include_background=True)
        # Create perfect predictions: each voxel's class has high logits
        # Use 4 classes, divide volume into 4 regions
        pred = torch.zeros(2, 4, 16, 16, 16)
        target = torch.zeros(2, 16, 16, 16, dtype=torch.long)

        # Split along first spatial dimension into 4 quadrants
        for c in range(4):
            start = c * 4
            end = (c + 1) * 4
            pred[:, c, start:end, :, :] = 10  # High logits for class c in its region
            target[:, start:end, :, :] = c     # Target is class c in that region

        loss = loss_fn(pred, target)
        assert loss.item() < 0.1


class TestEdgeLoss:
    def test_edge_loss_output(self):
        """Test edge loss computation."""
        from losses.edge_loss import EdgeLoss

        loss_fn = EdgeLoss()
        pred = torch.randn(2, 4, 16, 16, 16)
        target = torch.randint(0, 4, (2, 16, 16, 16))

        loss = loss_fn(pred, target)
        assert loss.item() >= 0


class TestContrastiveLoss:
    def test_contrastive_loss_identical_features(self):
        """Identical normalized features should give low loss."""
        from losses.contrastive_loss import ContrastiveLoss

        loss_fn = ContrastiveLoss(temperature=0.07)
        feat = torch.randn(4, 256)
        feat = feat / feat.norm(dim=-1, keepdim=True)

        loss = loss_fn(feat, feat)
        assert loss.item() < 1.0
