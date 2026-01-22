# losses/dice_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""

    def __init__(self, smooth: float = 1e-5, include_background: bool = False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

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
        for i in range(start_idx, num_classes):
            pred_i = pred[:, i]
            target_i = target_onehot[:, i]
            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)

        return 1 - torch.stack(dice_scores).mean()
