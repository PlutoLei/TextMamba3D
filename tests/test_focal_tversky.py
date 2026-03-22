# tests/test_focal_tversky.py
import torch


class TestFocalTverskyLoss:
    def test_output_is_scalar(self):
        from losses.focal_tversky_loss import FocalTverskyLoss
        ftl = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
        pred = torch.randn(2, 4, 8, 8, 8)
        target = torch.randint(0, 4, (2, 8, 8, 8))
        loss = ftl(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_perfect_prediction_low_loss(self):
        from losses.focal_tversky_loss import FocalTverskyLoss
        ftl = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
        pred = torch.zeros(1, 4, 16, 16, 16)
        target = torch.zeros(1, 16, 16, 16, dtype=torch.long)
        for c in range(4):
            start = c * 4
            end = (c + 1) * 4
            pred[:, c, start:end, :, :] = 10.0
            target[:, start:end, :, :] = c
        loss = ftl(pred, target)
        assert loss.item() < 0.1

    def test_excludes_background_by_default(self):
        from losses.focal_tversky_loss import FocalTverskyLoss
        # All-background target: foreground classes absent → loss should be ~0
        pred = torch.zeros(1, 4, 8, 8, 8)
        pred[:, 0] = 10.0
        target = torch.zeros(1, 8, 8, 8, dtype=torch.long)
        ftl_no_bg = FocalTverskyLoss(include_background=False)
        ftl_with_bg = FocalTverskyLoss(include_background=True)
        loss_no_bg = ftl_no_bg(pred, target)
        loss_with_bg = ftl_with_bg(pred, target)
        assert abs(loss_no_bg.item()) < 1e-6  # No foreground → zero loss
        assert loss_with_bg.item() < loss_no_bg.item() + 0.01  # Background well-predicted

    def test_class_weights_affect_loss(self):
        from losses.focal_tversky_loss import FocalTverskyLoss
        torch.manual_seed(123)
        pred = torch.randn(1, 4, 8, 8, 8)
        target = torch.randint(0, 4, (1, 8, 8, 8))
        ftl_uniform = FocalTverskyLoss(class_weights=[1.0, 1.0, 1.0, 1.0])
        ftl_weighted = FocalTverskyLoss(class_weights=[0.25, 3.0, 1.0, 4.0])
        loss_u = ftl_uniform(pred, target)
        loss_w = ftl_weighted(pred, target)
        assert loss_u.shape == ()
        assert loss_w.shape == ()
        assert abs(loss_u.item() - loss_w.item()) > 1e-6  # Weights change the loss

    def test_asymmetry_fn_penalty(self):
        from losses.focal_tversky_loss import FocalTverskyLoss
        torch.manual_seed(42)
        # Pred strongly favors class 0, but target has class 1 in half the volume.
        # High beta penalizes false negatives more -> higher loss when FN dominates.
        # Use include_background=False so background FP dynamics don't mask the effect.
        pred = torch.zeros(1, 4, 8, 8, 8)
        pred[:, 0] = 10.0
        target = torch.zeros(1, 8, 8, 8, dtype=torch.long)
        target[:, 4:8, :, :] = 1
        ftl_high_beta = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.0, include_background=False)
        ftl_low_beta = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.0, include_background=False)
        loss_high = ftl_high_beta(pred, target)
        loss_low = ftl_low_beta(pred, target)
        assert loss_high.item() > loss_low.item()

    def test_backward_pass(self):
        from losses.focal_tversky_loss import FocalTverskyLoss
        ftl = FocalTverskyLoss(class_weights=[0.25, 3.0, 1.0, 4.0])
        pred = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
        target = torch.randint(0, 4, (1, 8, 8, 8))
        loss = ftl(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_gamma_focal_effect(self):
        """Higher gamma suppresses easy-sample loss → lower total loss for random preds."""
        from losses.focal_tversky_loss import FocalTverskyLoss
        torch.manual_seed(99)
        pred = torch.randn(1, 4, 8, 8, 8)
        target = torch.randint(0, 4, (1, 8, 8, 8))
        ftl_low_gamma = FocalTverskyLoss(gamma=1.0)
        ftl_high_gamma = FocalTverskyLoss(gamma=2.0)
        loss_low = ftl_low_gamma(pred, target)
        loss_high = ftl_high_gamma(pred, target)
        assert loss_low.item() >= 0
        assert loss_high.item() >= 0
        # (1-TI)^2 < (1-TI)^1 when TI > 0, so higher gamma → lower loss
        assert loss_high.item() < loss_low.item()


class TestFTLIntegration:
    def test_combined_loss_uses_ftl(self):
        from losses import CombinedLoss
        from losses.focal_tversky_loss import FocalTverskyLoss

        criterion = CombinedLoss(
            use_ftl=True,
            ftl_alpha=0.3, ftl_beta=0.7, ftl_gamma=1.33,
            class_weights=[0.25, 3.0, 1.0, 4.0],
        )
        assert isinstance(criterion.dice_loss, FocalTverskyLoss)

        pred = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
        target = torch.randint(0, 4, (1, 8, 8, 8))
        losses = criterion(pred, target)
        assert losses['total'].item() > 0
        losses['total'].backward()
        assert pred.grad is not None

    def test_combined_loss_default_uses_dice(self):
        from losses import CombinedLoss
        from losses.dice_loss import DiceLoss

        criterion = CombinedLoss(class_weights=[0.25, 3.0, 1.0, 4.0])
        assert isinstance(criterion.dice_loss, DiceLoss)

    def test_combined_loss_ftl_deep_supervision(self):
        from losses import CombinedLoss

        criterion = CombinedLoss(
            use_ftl=True,
            class_weights=[0.25, 3.0, 1.0, 4.0],
        )
        pred = torch.randn(1, 4, 16, 16, 16, requires_grad=True)
        target = torch.randint(0, 4, (1, 16, 16, 16))
        aux = [torch.randn(1, 4, 8, 8, 8), torch.randn(1, 4, 4, 4, 4)]
        losses = criterion(pred, target, aux_preds=aux)
        assert losses['deep_supervision'].item() > 0
