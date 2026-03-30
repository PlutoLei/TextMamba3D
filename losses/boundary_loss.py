"""Boundary Loss for brain tumor segmentation.

Based on Kervadec et al. "Boundary loss for highly unbalanced segmentation" (MIDL 2019).
Uses signed distance transform of ground truth boundaries as supervision signal.
Architecture-agnostic — works with any model outputting softmax probabilities.

Reference: https://github.com/LIVIAETS/boundary-loss (MIT License)
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_distance_map(mask: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Compute signed distance transform for each class.

    Args:
        mask: [D, H, W] integer segmentation mask
        num_classes: number of classes

    Returns:
        [C, D, H, W] signed distance maps (negative inside, positive outside)
    """
    dist_maps = np.zeros((num_classes, *mask.shape), dtype=np.float32)
    for c in range(num_classes):
        binary = (mask == c).astype(np.uint8)
        if binary.sum() == 0:
            # Class not present — all positive (far from boundary)
            dist_maps[c] = np.ones_like(binary, dtype=np.float32) * 1e4
            continue
        if binary.sum() == binary.size:
            # Class fills entire volume — all negative (inside)
            dist_maps[c] = -np.ones_like(binary, dtype=np.float32) * 1e4
            continue
        # Positive distance outside, negative inside
        pos_dist = distance_transform_edt(1 - binary)
        neg_dist = distance_transform_edt(binary)
        dist_maps[c] = pos_dist - neg_dist
    return dist_maps


class BoundaryLoss(nn.Module):
    """Boundary loss using signed distance transforms.

    L_boundary = sum over classes: mean(softmax_probs * dist_map)

    Minimized when predicted probabilities are high inside true regions
    (where dist_map is negative) and low outside (where dist_map is positive).
    """

    def __init__(self, include_background: bool = False):
        super().__init__()
        self.include_background = include_background

    def forward(self, pred: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, D, H, W] softmax probabilities
            dist_maps: [B, C, D, H, W] signed distance maps

        Returns:
            scalar boundary loss
        """
        start = 0 if self.include_background else 1
        pc = pred[:, start:]
        dc = dist_maps[:, start:]
        loss = (pc * dc).mean()
        return loss
