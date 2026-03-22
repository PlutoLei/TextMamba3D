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

        ftl_scores = []
        weights = []
        for i in range(start_idx, num_classes):
            p_i = pred_soft[:, i]
            g_i = target_onehot[:, i]

            if g_i.sum() < 1e-6:
                ftl_scores.append(pred.new_zeros(()))
                weights.append(pred.new_zeros(()))
                continue

            tp = (p_i * g_i).sum()
            fp = (p_i * (1 - g_i)).sum()
            fn = ((1 - p_i) * g_i).sum()

            tversky_index = (tp + self.smooth) / (
                tp + self.alpha * fp + self.beta * fn + self.smooth
            )
            focal_tversky = (1 - tversky_index) ** self.gamma

            ftl_scores.append(focal_tversky)

            if self.class_weights is not None and i < len(self.class_weights):
                weights.append(
                    self.class_weights[i].to(device=pred.device, dtype=pred.dtype)
                )
            else:
                weights.append(pred.new_ones(()))

        if not ftl_scores:
            return pred.new_zeros(())

        ftl_tensor = torch.stack(ftl_scores)
        weight_tensor = torch.stack(weights)
        weight_sum = weight_tensor.sum()
        if weight_sum < 1e-8:
            return pred.new_zeros(())
        weight_tensor = weight_tensor / weight_sum

        return (ftl_tensor * weight_tensor).sum()
