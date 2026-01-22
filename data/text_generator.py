# data/text_generator.py
import torch
import numpy as np
from typing import Dict, Tuple


class DiagnosisTextGenerator:
    """Generate diagnosis text from segmentation mask."""

    # Brain region mapping (simplified)
    BRAIN_REGIONS = {
        (0.0, 0.5): {"x": "右侧", "y": "后", "z": "下"},
        (0.5, 1.0): {"x": "左侧", "y": "前", "z": "上"},
    }

    LOBE_MAP = {
        "前上": "额叶",
        "前下": "额叶",
        "后上": "顶叶",
        "后下": "枕叶",
    }

    def __init__(self, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.voxel_spacing = voxel_spacing

    def _get_region(self, centroid: np.ndarray, shape: tuple) -> str:
        """Map centroid to brain region."""
        norm = centroid / np.array(shape)

        x_side = "左侧" if norm[0] > 0.5 else "右侧"
        y_pos = "前" if norm[1] > 0.5 else "后"
        z_pos = "上" if norm[2] > 0.5 else "下"

        lobe = self.LOBE_MAP.get(y_pos + z_pos, "脑实质")

        return f"{x_side}{lobe}"

    def _compute_volumes(self, mask: torch.Tensor) -> Dict[str, float]:
        """Compute volumes of each region in cm³."""
        voxel_vol = np.prod(self.voxel_spacing) / 1000  # mm³ to cm³

        volumes = {
            "necrotic": (mask == 1).sum().item() * voxel_vol,
            "edema": (mask == 2).sum().item() * voxel_vol,
            "enhancing": (mask == 4).sum().item() * voxel_vol,
        }
        volumes["total"] = sum(volumes.values())

        return volumes

    def _analyze_boundary(self, mask: torch.Tensor) -> str:
        """Analyze boundary characteristics."""
        # Simple gradient-based analysis
        tumor_mask = (mask > 0).float()

        # Compute gradient magnitude
        grad_x = torch.diff(tumor_mask, dim=0).abs()
        grad_y = torch.diff(tumor_mask, dim=1).abs()
        grad_z = torch.diff(tumor_mask, dim=2).abs()

        edge_sum = grad_x.sum() + grad_y.sum() + grad_z.sum()
        tumor_surface = edge_sum.item()
        tumor_volume = tumor_mask.sum().item()

        if tumor_volume == 0:
            return "未见明显病灶"

        # Surface to volume ratio indicates boundary complexity
        ratio = tumor_surface / (tumor_volume ** (2/3) + 1e-8)

        if ratio > 10:
            return "边界不规则，呈浸润性生长"
        elif ratio > 6:
            return "边界欠清，与周围组织分界不明确"
        else:
            return "边界尚清"

    def _get_grade(self, volumes: Dict[str, float]) -> str:
        """Estimate tumor grade based on volumes."""
        enhancing_ratio = volumes["enhancing"] / (volumes["total"] + 1e-8)

        if enhancing_ratio > 0.3:
            return "高级别胶质瘤(HGG)"
        else:
            return "考虑低级别胶质瘤(LGG)可能"

    def generate(self, mask: torch.Tensor) -> str:
        """
        Generate diagnosis text from segmentation mask.

        Args:
            mask: [D, H, W] segmentation mask
                0: background
                1: necrotic/non-enhancing
                2: edema
                4: enhancing tumor
        Returns:
            Diagnosis text string
        """
        if mask.sum() == 0:
            return "MRI平扫未见明显异常信号。"

        # Find tumor centroid
        tumor_coords = torch.nonzero(mask > 0).float()
        centroid = tumor_coords.mean(dim=0).numpy()

        # Get region
        region = self._get_region(centroid, mask.shape)

        # Compute volumes
        volumes = self._compute_volumes(mask)

        # Analyze boundary
        boundary_desc = self._analyze_boundary(mask)

        # Get grade
        grade = self._get_grade(volumes)

        # Generate text
        text = (
            f"MRI示{region}占位性病变，"
            f"大小约{volumes['total']:.1f}cm³，"
            f"其中强化区域约{volumes['enhancing']:.1f}cm³，"
            f"周围水肿区域约{volumes['edema']:.1f}cm³。"
            f"{boundary_desc}，"
            f"{grade}。"
        )

        return text

    def generate_batch(self, masks: torch.Tensor) -> list:
        """Generate texts for a batch of masks."""
        return [self.generate(m) for m in masks]
