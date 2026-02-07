# utils/metrics.py
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


# BraTS 标准区域定义 (基于重映射后的标签: 0=bg, 1=necrotic, 2=edema, 3=enhancing)
BRATS_REGIONS = {
    'ET': [3],        # Enhancing Tumor
    'TC': [1, 3],     # Tumor Core = necrotic + enhancing
    'WT': [1, 2, 3],  # Whole Tumor = necrotic + edema + enhancing
}


def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict:
    """Compute Dice score per class.

    When both pred and target are empty for a class, Dice = 1.0 (BraTS standard).
    """
    pred_argmax = pred.argmax(dim=1)

    scores = {}
    for c in range(1, num_classes):  # Skip background
        pred_c = (pred_argmax == c).float()
        target_c = (target == c).float()

        pred_sum = pred_c.sum()
        target_sum = target_c.sum()

        # BraTS standard: both empty → perfect match
        if pred_sum == 0 and target_sum == 0:
            scores[f'dice_class_{c}'] = 1.0
            continue

        intersection = (pred_c * target_c).sum()
        union = pred_sum + target_sum
        dice = (2 * intersection) / (union + 1e-8)
        scores[f'dice_class_{c}'] = dice.item()

    scores['dice_mean'] = np.mean(list(scores.values()))
    return scores


def dice_score_brats_regions(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute Dice score per BraTS region (ET, TC, WT).

    Regions:
        ET (Enhancing Tumor) = class 3
        TC (Tumor Core)      = class 1 + 3
        WT (Whole Tumor)     = class 1 + 2 + 3
    """
    pred_argmax = pred.argmax(dim=1)
    scores = {}

    for region_name, classes in BRATS_REGIONS.items():
        pred_region = sum((pred_argmax == c).float() for c in classes)
        target_region = sum((target == c).float() for c in classes)

        pred_region = (pred_region > 0).float()
        target_region = (target_region > 0).float()

        pred_sum = pred_region.sum()
        target_sum = target_region.sum()

        if pred_sum == 0 and target_sum == 0:
            scores[f'dice_{region_name}'] = 1.0
            continue

        intersection = (pred_region * target_region).sum()
        union = pred_sum + target_sum
        dice = (2 * intersection) / (union + 1e-8)
        scores[f'dice_{region_name}'] = dice.item()

    scores['dice_mean'] = np.mean(list(scores.values()))
    return scores


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute 95th percentile Hausdorff distance.

    Args:
        pred: binary mask (any numeric type, >0 = positive)
        target: binary mask (any numeric type, >0 = positive)
    """
    if pred.sum() == 0 or target.sum() == 0:
        return np.nan

    # Cast to bool to ensure correct logical negation
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)

    # Distance transform on background (inverted mask)
    pred_dist = distance_transform_edt(~pred_bool)
    target_dist = distance_transform_edt(~target_bool)

    # Surface distances
    pred_surface = pred_dist[target_bool]
    target_surface = target_dist[pred_bool]

    all_distances = np.concatenate([pred_surface, target_surface])

    return np.percentile(all_distances, 95)


def hausdorff_distance_95_brats_regions(
    pred_np: np.ndarray, target_np: np.ndarray
) -> dict:
    """Compute HD95 per BraTS region (ET, TC, WT)."""
    scores = {}
    for region_name, classes in BRATS_REGIONS.items():
        pred_region = np.isin(pred_np, classes).astype(np.uint8)
        target_region = np.isin(target_np, classes).astype(np.uint8)
        hd95 = hausdorff_distance_95(pred_region, target_region)
        scores[f'hd95_{region_name}'] = hd95
    return scores
