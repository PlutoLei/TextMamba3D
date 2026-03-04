# losses/dice_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# BraTS default class weights (inverse frequency, clinically motivated)
# Class 0 (background): excluded from Dice by default
# Class 1 (necrotic/non-enhancing tumor core): rare, high weight
# Class 2 (peritumoral edema): moderate frequency
# Class 3 (GD-enhancing tumor): rarest, clinically most critical
BRATS_CLASS_WEIGHTS = [0.25, 3.0, 1.0, 4.0]


class DiceLoss(nn.Module):
    """Class-weighted Dice loss for segmentation.

    Supports per-class weighting to handle class imbalance common in
    medical segmentation (e.g., BraTS tumor subregions).
    """

    def __init__(
        self,
        smooth: float = 1e-4,
        include_background: bool = False,
        class_weights: list[float] | None = None,
    ):
        super().__init__()
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
        pred = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes)
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()

        start_idx = 0 if self.include_background else 1

        dice_scores = []
        weights = []
        for i in range(start_idx, num_classes):
            pred_i = pred[:, i]
            target_i = target_onehot[:, i]
            if target_i.sum().item() < 1e-6:
                dice_scores.append(pred.new_tensor(1.0))
                weights.append(pred.new_zeros(()))
            else:
                intersection = (pred_i * target_i).sum()
                union = pred_i.sum() + target_i.sum()
                dice = (2 * intersection + self.smooth) / (union + self.smooth)
                dice_scores.append(dice)

                if self.class_weights is not None and i < len(self.class_weights):
                    weights.append(self.class_weights[i].to(device=pred.device, dtype=pred.dtype))
                else:
                    weights.append(pred.new_ones(()))

        dice_tensor = torch.stack(dice_scores)
        weight_tensor = torch.stack(weights)
        weight_sum = weight_tensor.sum()
        if weight_sum.item() < 1e-8:
            return pred.new_zeros(())
        weight_tensor = weight_tensor / weight_sum  # Normalize

        return 1 - (dice_tensor * weight_tensor).sum()
