# losses/focal_tversky_loss.py
"""Focal Tversky Loss for class-imbalanced medical image segmentation.

Reference: Abraham & Khan, "A Novel Focal Tversky Loss Function with
Improved Attention U-Net for Lesion Segmentation", ISBI 2019.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss with asymmetric FP/FN weighting.

    Args:
        alpha: Weight for false positives (over-segmentation). Default 0.3.
        beta: Weight for false negatives (under-segmentation). Default 0.7.
        gamma: Focal parameter. Values >1 down-weight easy samples. Default 1.33.
        smooth: Smoothing term to avoid division by zero. Default 1.0.
        include_background: Whether to include class 0 in loss. Default False.
        class_weights: Per-class weights [C] including background. Default None.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        smooth: float = 1.0,
        include_background: bool = False,
        class_weights: list[float] | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.include_background = include_background

        if class_weights is not None:
            self.register_buffer(
                'class_weights',
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, D, H, W] logits
            target: [B, D, H, W] class indices
        Returns:
            Scalar loss
        """
        num_classes = pred.shape[1]
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes)
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()

        start_idx = 0 if self.include_background else 1
        if start_idx >= num_classes:
            return pred.new_zeros(())

        # Vectorized per-class TP/FP/FN over [B, D, H, W].
        tp = (pred_soft * target_onehot).sum(dim=(0, 2, 3, 4))  # [C]
        fp = (pred_soft * (1 - target_onehot)).sum(dim=(0, 2, 3, 4))  # [C]
        fn = ((1 - pred_soft) * target_onehot).sum(dim=(0, 2, 3, 4))  # [C]

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        ftl_tensor = (1 - tversky_index).pow(self.gamma)[start_idx:]

        present_mask = target_onehot.sum(dim=(0, 2, 3, 4))[start_idx:] >= 1e-6
        ftl_tensor = torch.where(present_mask, ftl_tensor, torch.zeros_like(ftl_tensor))

        weight_tensor = pred.new_ones(num_classes)
        if self.class_weights is not None:
            num_weighted = min(num_classes, self.class_weights.numel())
            weight_tensor[:num_weighted] = self.class_weights[:num_weighted].to(
                device=pred.device,
                dtype=pred.dtype,
            )
        weight_tensor = weight_tensor[start_idx:]

        weight_sum = weight_tensor.sum()
        if weight_sum < 1e-8:
            return pred.new_zeros(())
        weight_tensor = weight_tensor / weight_sum

        return (ftl_tensor * weight_tensor).sum()
