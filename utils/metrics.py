# utils/metrics.py
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict:
    """Compute Dice score per class."""
    pred_argmax = pred.argmax(dim=1)

    scores = {}
    for c in range(1, num_classes):  # Skip background
        pred_c = (pred_argmax == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2 * intersection) / (union + 1e-8)
        scores[f'dice_class_{c}'] = dice.item()

    scores['dice_mean'] = np.mean(list(scores.values()))
    return scores


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute 95th percentile Hausdorff distance."""
    if pred.sum() == 0 or target.sum() == 0:
        return np.nan

    # Distance transform
    pred_dist = distance_transform_edt(~pred)
    target_dist = distance_transform_edt(~target)

    # Surface distances
    pred_surface = pred_dist[target > 0]
    target_surface = target_dist[pred > 0]

    all_distances = np.concatenate([pred_surface, target_surface])

    return np.percentile(all_distances, 95)
