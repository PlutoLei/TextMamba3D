# data/text_generator.py
import torch
import numpy as np
import random
from typing import Dict, Tuple, Optional


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

    # 同义词替换映射（用于文本噪声增强）
    SYNONYMS = {
        "左侧": ["左侧", "左", "左边", "左半球"],
        "右侧": ["右侧", "右", "右边", "右半球"],
        "额叶": ["额叶", "额部", "额区"],
        "顶叶": ["顶叶", "顶部", "顶区"],
        "枕叶": ["枕叶", "枕部", "枕区"],
        "脑实质": ["脑实质", "脑组织", "脑内"],
        "占位性病变": ["占位性病变", "占位", "病灶", "异常信号"],
        "边界不规则，呈浸润性生长": ["边界不规则，呈浸润性生长", "边界不规则", "呈浸润性生长", "边界模糊不清"],
        "边界欠清，与周围组织分界不明确": ["边界欠清，与周围组织分界不明确", "边界欠清", "分界不清", "边界不清晰"],
        "边界尚清": ["边界尚清", "边界清晰", "边界较清", "分界清楚"],
        "高级别胶质瘤(HGG)": ["高级别胶质瘤(HGG)", "高级别胶质瘤", "HGG可能", "考虑高级别胶质瘤"],
        "考虑低级别胶质瘤(LGG)可能": ["考虑低级别胶质瘤(LGG)可能", "低级别胶质瘤可能", "LGG可能", "考虑低级别胶质瘤"],
    }

    # 体积描述模板（用于模糊化数值）
    VOLUME_TEMPLATES = {
        "precise": "大小约{vol:.1f}cm³",
        "round": "大小约{vol:.0f}cm³",
        "vague_small": "体积较小",
        "vague_medium": "体积中等",
        "vague_large": "体积较大",
        "range": "大小约{vol_low:.0f}-{vol_high:.0f}cm³",
    }

    def __init__(
        self,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        noise_prob: float = 0.5,
        enable_noise: bool = True
    ):
        self.voxel_spacing = voxel_spacing
        self.noise_prob = noise_prob  # 应用噪声的概率
        self.enable_noise = enable_noise  # 是否启用噪声

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

        # 注意: mask 已经映射为 0, 1, 2, 3 (原始 4->3)
        volumes = {
            "necrotic": (mask == 1).sum().item() * voxel_vol,
            "edema": (mask == 2).sum().item() * voxel_vol,
            "enhancing": ((mask == 3) | (mask == 4)).sum().item() * voxel_vol,  # 兼容两种标签
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

    def _apply_synonym_noise(self, text: str) -> str:
        """随机替换同义词，模拟真实医生描述的多样性。"""
        for original, synonyms in self.SYNONYMS.items():
            if original in text and random.random() < self.noise_prob:
                replacement = random.choice(synonyms)
                text = text.replace(original, replacement, 1)
        return text

    def _apply_volume_noise(self, volume: float) -> str:
        """对体积数值应用噪声，模拟真实描述中的模糊性。"""
        if random.random() > self.noise_prob:
            return self.VOLUME_TEMPLATES["precise"].format(vol=volume)

        noise_type = random.choice(["round", "vague", "range", "omit"])

        if noise_type == "round":
            return self.VOLUME_TEMPLATES["round"].format(vol=volume)
        elif noise_type == "vague":
            if volume < 10:
                return self.VOLUME_TEMPLATES["vague_small"]
            elif volume < 50:
                return self.VOLUME_TEMPLATES["vague_medium"]
            else:
                return self.VOLUME_TEMPLATES["vague_large"]
        elif noise_type == "range":
            vol_low = volume * random.uniform(0.8, 0.95)
            vol_high = volume * random.uniform(1.05, 1.2)
            return self.VOLUME_TEMPLATES["range"].format(vol_low=vol_low, vol_high=vol_high)
        else:  # omit
            return ""

    def _apply_structure_noise(
        self,
        region: str,
        volumes: Dict[str, float],
        boundary_desc: str,
        grade: str
    ) -> str:
        """应用结构噪声，随机省略或重排信息，模拟真实报告的多样性。"""
        components = []

        # 位置信息（大概率保留）
        if random.random() < 0.9:
            region_text = self._apply_synonym_noise(f"MRI示{region}占位性病变")
            components.append(region_text)
        else:
            components.append("MRI示脑内占位性病变")

        # 总体积（可能省略或模糊化）
        vol_text = self._apply_volume_noise(volumes['total'])
        if vol_text:
            components.append(vol_text)

        # 强化区域（可能省略）
        if random.random() < 0.7:
            enh_vol = volumes['enhancing']
            if random.random() < self.noise_prob:
                components.append(f"可见强化")
            else:
                components.append(f"其中强化区域约{enh_vol:.1f}cm³")

        # 水肿区域（可能省略）
        if random.random() < 0.6:
            edema_vol = volumes['edema']
            if random.random() < self.noise_prob:
                if edema_vol > 20:
                    components.append("周围水肿明显")
                else:
                    components.append("周围可见水肿")
            else:
                components.append(f"周围水肿区域约{edema_vol:.1f}cm³")

        # 边界描述（可能简化）
        boundary_text = self._apply_synonym_noise(boundary_desc)
        components.append(boundary_text)

        # 分级（可能省略或简化）
        if random.random() < 0.8:
            grade_text = self._apply_synonym_noise(grade)
            components.append(grade_text)

        # 组合文本，使用不同的连接符
        if random.random() < 0.5:
            text = "，".join(components) + "。"
        else:
            text = "；".join(components) + "。"

        return text

    def generate(self, mask: torch.Tensor, apply_noise: Optional[bool] = None) -> str:
        """
        Generate diagnosis text from segmentation mask.

        Args:
            mask: [D, H, W] segmentation mask
                0: background
                1: necrotic/non-enhancing
                2: edema
                4: enhancing tumor
            apply_noise: 是否应用噪声增强。None 时使用 self.enable_noise
        Returns:
            Diagnosis text string
        """
        if mask.sum() == 0:
            return "MRI平扫未见明显异常信号。"

        # 确定是否应用噪声
        use_noise = apply_noise if apply_noise is not None else self.enable_noise

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

        # Generate text with or without noise
        if use_noise and random.random() < self.noise_prob:
            text = self._apply_structure_noise(region, volumes, boundary_desc, grade)
        else:
            # 原始精确模板
            text = (
                f"MRI示{region}占位性病变，"
                f"大小约{volumes['total']:.1f}cm³，"
                f"其中强化区域约{volumes['enhancing']:.1f}cm³，"
                f"周围水肿区域约{volumes['edema']:.1f}cm³。"
                f"{boundary_desc}，"
                f"{grade}。"
            )

        return text

    def generate_clean(self, mask: torch.Tensor) -> str:
        """生成无噪声的精确诊断文本（用于评估）。"""
        return self.generate(mask, apply_noise=False)

    def generate_noisy(self, mask: torch.Tensor) -> str:
        """生成带噪声的诊断文本（用于训练增强）。"""
        return self.generate(mask, apply_noise=True)

    def generate_batch(self, masks: torch.Tensor, apply_noise: Optional[bool] = None) -> list:
        """Generate texts for a batch of masks."""
        return [self.generate(m, apply_noise) for m in masks]
