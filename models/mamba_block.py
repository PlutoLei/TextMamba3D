# models/mamba_block.py
import torch
import torch.nn as nn
from einops import rearrange

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class MambaBlock(nn.Module):
    """Single Mamba block with residual connection."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        if Mamba is not None:
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Fallback: simple linear layer for testing without mamba-ssm
            self.mamba = nn.Sequential(
                nn.Linear(dim, dim * expand),
                nn.GELU(),
                nn.Linear(dim * expand, dim),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input tensor
        Returns:
            [B, L, D] output tensor
        """
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class MambaLayer(nn.Module):
    """Stack of Mamba blocks."""

    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(dim, d_state, d_conv, expand, dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
