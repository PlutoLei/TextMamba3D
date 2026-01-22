# models/fusion.py
import torch
import torch.nn as nn
from .mamba_block import MambaLayer


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
