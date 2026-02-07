# utils/sliding_window.py
"""Sliding window utilities for 3D medical image segmentation."""

import numpy as np
import torch


def gaussian_weight_3d(
    patch_size: tuple[int, int, int],
    sigma_scale: float = 0.125,
) -> torch.Tensor:
    """Create 3D Gaussian importance map for sliding window inference.

    Center voxels receive higher weight, which reduces stitching artifacts
    at patch borders when aggregating overlapping predictions.

    Args:
        patch_size: Spatial dimensions (D, H, W) of the patch.
        sigma_scale: Fraction of patch size used as Gaussian sigma.
            Smaller values produce a more peaked (center-focused) weighting.

    Returns:
        Tensor of shape [D, H, W] with values in [1e-4, 1.0].
    """
    coords = [np.arange(s) - (s - 1) / 2.0 for s in patch_size]
    grid = np.stack(np.meshgrid(*coords, indexing='ij'), axis=-1)
    sigma = np.array(patch_size) * sigma_scale
    gauss = np.exp(-0.5 * np.sum((grid / sigma) ** 2, axis=-1))
    gauss = gauss / gauss.max()
    gauss = np.clip(gauss, 1e-4, 1.0)
    return torch.from_numpy(gauss).float()
