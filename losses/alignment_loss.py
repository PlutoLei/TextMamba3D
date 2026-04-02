import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for text-image alignment.

    Pulls matching text-image pairs together, pushes non-matching apart.
    Uses in-batch negatives (batch size = number of negatives).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, img_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        img_feat = F.normalize(img_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        logits = img_feat @ text_feat.T / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2
