# models/fusion.py
import torch
import torch.nn as nn
from .mamba_block import MambaLayer


# ---------------------------------------------------------------------------
# FiLM: Feature-wise Linear Modulation for multi-scale text guidance
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: output = gamma * input + beta.

    Predicts per-channel scale and shift from a conditioning vector,
    then applies them to every spatial token.
    """

    def __init__(self, feat_dim: int, cond_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, feat_dim)
        self.beta_proj = nn.Linear(cond_dim, feat_dim)

        # Initialize gamma close to 1 and beta close to 0
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] spatial feature tokens
            cond: [B, D_cond] conditioning vector (text global feature)
        Returns:
            [B, L, D] modulated features
        """
        gamma = self.gamma_proj(cond).unsqueeze(1)  # [B, 1, D]
        beta = self.beta_proj(cond).unsqueeze(1)     # [B, 1, D]
        return gamma * x + beta


class MultiScaleFiLM(nn.Module):
    """Apply FiLM modulation at multiple encoder scales.

    One FiLMLayer per encoder stage, conditioned on text global feature.
    Modulates skip connections before they enter the decoder.
    """

    def __init__(self, stage_dims: list[int], text_dim: int):
        """
        Args:
            stage_dims: Channel dimensions for each encoder stage, e.g. [96, 192, 384, 768]
            text_dim: Dimension of text global feature
        """
        super().__init__()
        self.film_layers = nn.ModuleList([
            FiLMLayer(feat_dim=dim, cond_dim=text_dim)
            for dim in stage_dims
        ])

    def forward(
        self,
        features: list[torch.Tensor],
        text_global: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Args:
            features: List of encoder features [stage0, stage1, ..., stageN]
            text_global: [B, D_text] text global feature
        Returns:
            List of FiLM-modulated features (same shapes)
        """
        return [
            film(feat, text_global)
            for film, feat in zip(self.film_layers, features)
        ]


# ---------------------------------------------------------------------------
# MambaFusion: Deep bottleneck fusion via causal Mamba
# ---------------------------------------------------------------------------

class MambaFusion(nn.Module):
    """Fuse image and text features using Mamba.

    Strategy: Concatenate [text, image] tokens, process with Mamba,
    then extract image portion. Text at the front guides image features
    through Mamba's causal nature.
    """

    def __init__(
        self,
        img_dim: int,
        text_dim: int,
        hidden_dim: int,
        depth: int = 2,
        d_state: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Project to common dimension
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Mamba fusion layers
        self.mamba_fusion = MambaLayer(
            dim=hidden_dim,
            depth=depth,
            d_state=d_state,
            dropout=dropout,
        )

        # Project back to image dimension
        self.out_proj = nn.Linear(hidden_dim, img_dim)
        self.norm = nn.LayerNorm(img_dim)

    def forward(
        self,
        img_feat: torch.Tensor,
        text_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            img_feat: [B, N, D_img] image features
            text_feat: [B, M, D_text] text features
        Returns:
            [B, N, D_img] fused image features
        """
        B, N, _ = img_feat.shape
        M = text_feat.shape[1]

        # Project to common space
        img_h = self.img_proj(img_feat)    # [B, N, hidden_dim]
        text_h = self.text_proj(text_feat)  # [B, M, hidden_dim]

        # Concatenate: [text, image] - text guides image
        concat = torch.cat([text_h, img_h], dim=1)  # [B, M+N, hidden_dim]

        # Mamba fusion
        fused = self.mamba_fusion(concat)  # [B, M+N, hidden_dim]

        # Extract image portion
        img_fused = fused[:, M:, :]  # [B, N, hidden_dim]

        # Project back and residual
        out = self.out_proj(img_fused)
        out = self.norm(out + img_feat)

        return out
