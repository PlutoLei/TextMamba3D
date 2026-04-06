# losses/text_supervision.py
"""R-Super inspired: text as supervision, not as input.

Instead of feeding text into the model as a conditioning signal (which the
pretrained backbone ignores), use text to create LOSS constraints that the
model MUST satisfy.

Key insight: the segmentation loss can be satisfied without text (backbone
already knows the answer). But text supervision loss CANNOT be satisfied
without making predictions consistent with what the text says.

Reference: R-Super (MICCAI 2025 Best Paper)

Three auxiliary losses:
1. RegionPresenceLoss: text says "enhancing tumor present" → model must predict ET > 0
2. VolumeConsistencyLoss: text describes tumor size → model's predicted volume should match
3. TextSegConsistencyLoss: combines both into a single module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re


class TextSegConsistencyLoss(nn.Module):
    """Consistency loss between text descriptions and segmentation predictions.

    Parses text to extract expected tumor properties, then penalizes the model
    if its segmentation contradicts the text. Text shapes the loss landscape,
    not the feature space — completely bypasses modality competition.
    """

    def __init__(self, presence_weight: float = 1.0, volume_weight: float = 0.5):
        super().__init__()
        self.presence_weight = presence_weight
        self.volume_weight = volume_weight
        # BraTS classes: 0=background, 1=NCR/NET, 2=ED, 3=ET
        # Regions: ET=class3, TC=class1+class3, WT=class1+class2+class3

    def _parse_text_batch(self, text_ids: torch.Tensor, tokenizer) -> dict:
        """Parse text tokens back to strings and extract tumor properties.

        Returns dict with per-sample boolean tensors for region presence.
        """
        B = text_ids.size(0)
        device = text_ids.device

        # Default: assume all regions present (safe default)
        has_et = torch.ones(B, device=device)
        has_edema = torch.ones(B, device=device)
        size_hint = torch.ones(B, device=device) * 0.5  # normalized 0-1

        if tokenizer is None:
            return {'has_et': has_et, 'has_edema': has_edema, 'size_hint': size_hint}

        texts = tokenizer.batch_decode(text_ids, skip_special_tokens=True)
        for i, text in enumerate(texts):
            text_lower = text.lower()
            # ET presence
            if any(k in text_lower for k in ['no enhancing', 'without enhancement',
                                               'non-enhancing only', 'no contrast']):
                has_et[i] = 0.0
            # Edema
            if any(k in text_lower for k in ['no edema', 'without edema',
                                               'no peritumoral']):
                has_edema[i] = 0.0
            # Size hints
            if any(k in text_lower for k in ['small', 'tiny', 'minimal']):
                size_hint[i] = 0.2
            elif any(k in text_lower for k in ['large', 'extensive', 'massive']):
                size_hint[i] = 0.8

        return {'has_et': has_et, 'has_edema': has_edema, 'size_hint': size_hint}

    def forward(
        self,
        pred: torch.Tensor,
        text_ids: torch.Tensor | None = None,
        tokenizer=None,
    ) -> torch.Tensor:
        """
        Args:
            pred: [B, C, D, H, W] raw logits (4 classes)
            text_ids: [B, M] text token IDs
            tokenizer: tokenizer for decoding text_ids
        Returns:
            scalar loss
        """
        if text_ids is None:
            return torch.tensor(0.0, device=pred.device)

        B = pred.size(0)
        probs = pred.softmax(dim=1)  # [B, C, D, H, W]

        # Region probabilities (spatial average)
        # ET = class 3
        et_prob = probs[:, 3].mean(dim=(1, 2, 3))   # [B]
        # Edema = class 2
        ed_prob = probs[:, 2].mean(dim=(1, 2, 3))   # [B]
        # Total tumor volume fraction
        tumor_frac = (1 - probs[:, 0]).mean(dim=(1, 2, 3))  # [B]

        # Parse text for expected properties
        text_props = self._parse_text_batch(text_ids, tokenizer)

        loss = torch.tensor(0.0, device=pred.device)

        # 1. Region presence loss
        # If text says ET present (has_et=1), et_prob should be > 0
        # If text says ET absent (has_et=0), et_prob should be ~0
        if self.presence_weight > 0:
            # BCE between predicted region probability and text-derived label
            et_target = text_props['has_et']
            ed_target = text_props['has_edema']
            presence_loss = (
                F.binary_cross_entropy(et_prob.clamp(1e-7, 1-1e-7), et_target)
                + F.binary_cross_entropy(ed_prob.clamp(1e-7, 1-1e-7), ed_target)
            ) / 2
            loss = loss + self.presence_weight * presence_loss

        # 2. Volume consistency loss
        # Text-derived size hint should correlate with predicted tumor fraction
        if self.volume_weight > 0:
            size_target = text_props['size_hint']
            # Soft L1 between normalized predicted volume and text hint
            volume_loss = F.smooth_l1_loss(tumor_frac, size_target * 0.1)
            loss = loss + self.volume_weight * volume_loss

        return loss
