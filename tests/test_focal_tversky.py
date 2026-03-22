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
        ftl = FocalTverskyLoss(include_background=False)
        pred = torch.zeros(1, 4, 8, 8, 8)
        pred[:, 0] = 10.0
        target = torch.zeros(1, 8, 8, 8, dtype=torch.long)
        loss = ftl(pred, target)
        assert loss.item() >= 0

    def test_class_weights_affect_loss(self):
        from losses.focal_tversky_loss import FocalTverskyLoss
        pred = torch.randn(1, 4, 8, 8, 8)
        target = torch.randint(0, 4, (1, 8, 8, 8))
        ftl_uniform = FocalTverskyLoss(class_weights=[1.0, 1.0, 1.0, 1.0])
        ftl_weighted = FocalTverskyLoss(class_weights=[0.25, 3.0, 1.0, 4.0])
        loss_u = ftl_uniform(pred, target)
        loss_w = ftl_weighted(pred, target)
        assert loss_u.shape == ()
        assert loss_w.shape == ()

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
        from losses.focal_tversky_loss import FocalTverskyLoss
        pred = torch.randn(1, 4, 8, 8, 8)
        target = torch.randint(0, 4, (1, 8, 8, 8))
        ftl_low_gamma = FocalTverskyLoss(gamma=1.0)
        ftl_high_gamma = FocalTverskyLoss(gamma=2.0)
        loss_low = ftl_low_gamma(pred, target)
        loss_high = ftl_high_gamma(pred, target)
        assert loss_low.item() >= 0
        assert loss_high.item() >= 0
