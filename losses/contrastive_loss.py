# losses/contrastive_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Contrastive loss for text-image alignment."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, img_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        img_feat = F.normalize(img_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        logits = img_feat @ text_feat.T / self.temperature
        B = img_feat.shape[0]
        labels = torch.arange(B, device=img_feat.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2


class ForegroundContrastiveLoss(nn.Module):
    """Foreground-aware contrastive loss for bottleneck image tokens and text CLS."""

    def __init__(
        self,
        temperature: float = 0.07,
        feat_dim: int = 384,
        text_dim: int = 256,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.img_proj = nn.Sequential(
            nn.Linear(feat_dim, text_dim),
            nn.LayerNorm(text_dim),
        )

    def forward(
        self,
        pixel_feat: torch.Tensor,
        text_feat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE using foreground-weighted token pooling."""
        fg_mask = (mask > 0).to(dtype=pixel_feat.dtype).unsqueeze(1)
        fg_mask_down = F.adaptive_max_pool3d(fg_mask, output_size=(4, 4, 4))
        fg_weights = fg_mask_down.flatten(start_dim=1)

        has_foreground = fg_weights.sum(dim=1, keepdim=True) > 0
        uniform_weights = torch.ones_like(fg_weights)
        weights = torch.where(has_foreground, fg_weights, uniform_weights)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

        pooled_feat = (pixel_feat * weights.unsqueeze(-1)).sum(dim=1)
        img_feat = self.img_proj(pooled_feat)

        img_feat = F.normalize(img_feat, dim=-1)
        text_feat = F.normalize(text_feat.detach(), dim=-1)

        logits = img_feat @ text_feat.T / self.temperature
        batch_size = pixel_feat.shape[0]
        labels = torch.arange(batch_size, device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2
