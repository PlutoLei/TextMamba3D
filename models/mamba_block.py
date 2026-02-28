# models/mamba_block.py
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from einops import rearrange

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    Mamba = None
    MAMBA_AVAILABLE = False


def _create_ssm(dim: int, d_state: int, d_conv: int, expand: int, dropout: float = 0.0):
    """Create a Mamba SSM or MLP fallback."""
    if MAMBA_AVAILABLE:
        try:
            return Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        except Exception as e:
            print(f"Warning: Mamba init failed ({e}), using fallback MLP")
    return nn.Sequential(
        nn.Linear(dim, dim * expand),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * expand, dim),
    )


class MambaBlock(nn.Module):
    """Single unidirectional Mamba block with residual connection."""

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
        self.mamba = _create_ssm(dim, d_state, d_conv, expand, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, D] -> [B, L, D]"""
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class MambaLayer(nn.Module):
    """Stack of unidirectional Mamba blocks."""

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


# ---------------------------------------------------------------------------
# Bidirectional Mamba — forward + backward scans with learned merge
# ---------------------------------------------------------------------------

class BiMambaBlock(nn.Module):
    """Bidirectional Mamba block: forward + backward SSM scans.

    For 3D vision tasks, unidirectional scanning loses spatial context —
    tokens later in the sequence can't influence earlier ones.
    BiMamba solves this by scanning both directions and merging.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.forward_ssm = _create_ssm(dim, d_state, d_conv, expand, dropout)
        self.backward_ssm = _create_ssm(dim, d_state, d_conv, expand, dropout)
        self.merge = nn.Linear(dim * 2, dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, D] -> [B, L, D]"""
        residual = x
        x = self.norm(x)

        # Forward scan
        x_fwd = self.forward_ssm(x)

        # Backward scan: reverse → process → reverse back
        x_bwd = self.backward_ssm(x.flip(1)).flip(1)

        # Merge forward + backward
        x = self.merge(torch.cat([x_fwd, x_bwd], dim=-1))
        x = self.gelu(x)
        x = self.dropout(x)
        return x + residual


class BiMambaLayer(nn.Module):
    """Stack of bidirectional Mamba blocks."""

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
            BiMambaBlock(dim, d_state, d_conv, expand, dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Cross-Scan BiMamba for 3D — scans along 3 spatial axis orderings
# ---------------------------------------------------------------------------

class CrossScanBiMamba3DBlock(nn.Module):
    """Cross-scan bidirectional Mamba for 3D volumes.

    Scans bidirectionally along 3 different spatial axis orderings:
      - D-H-W (depth-first)
      - H-W-D (height-first)
      - W-D-H (width-first)
    Total: 6 scan directions, merged with learned projection.

    This provides comprehensive spatial coverage: tokens that are far
    apart in one ordering may be adjacent in another.
    """

    def __init__(
        self,
        dim: int,
        spatial_dims: tuple,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims  # (D, H, W) at this stage
        self.norm = nn.LayerNorm(dim)

        # One BiMamba per axis ordering (each does fwd + bwd = 2 directions)
        self.dhw_fwd = _create_ssm(dim, d_state, d_conv, expand, dropout)
        self.dhw_bwd = _create_ssm(dim, d_state, d_conv, expand, dropout)
        self.hwd_fwd = _create_ssm(dim, d_state, d_conv, expand, dropout)
        self.hwd_bwd = _create_ssm(dim, d_state, d_conv, expand, dropout)
        self.wdh_fwd = _create_ssm(dim, d_state, d_conv, expand, dropout)
        self.wdh_bwd = _create_ssm(dim, d_state, d_conv, expand, dropout)

        # Merge 6 directions -> dim
        self.merge = nn.Linear(dim * 6, dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def _reorder(self, x: torch.Tensor, src: str, dst: str) -> torch.Tensor:
        """Reorder flattened spatial tokens between different axis orderings."""
        D, H, W = self.spatial_dims
        return rearrange(x, f'b ({src}) c -> b ({dst}) c', d=D, h=H, w=W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, D*H*W, C] -> [B, D*H*W, C]"""
        residual = x
        x = self.norm(x)
        D, H, W = self.spatial_dims

        # Scan 1: D-H-W ordering (native)
        out_dhw_f = self.dhw_fwd(x)
        out_dhw_b = self.dhw_bwd(x.flip(1)).flip(1)

        # Scan 2: H-W-D ordering
        x_hwd = self._reorder(x, 'd h w', 'h w d')
        out_hwd_f = self._reorder(self.hwd_fwd(x_hwd), 'h w d', 'd h w')
        out_hwd_b = self._reorder(self.hwd_bwd(x_hwd.flip(1)).flip(1), 'h w d', 'd h w')

        # Scan 3: W-D-H ordering
        x_wdh = self._reorder(x, 'd h w', 'w d h')
        out_wdh_f = self._reorder(self.wdh_fwd(x_wdh), 'w d h', 'd h w')
        out_wdh_b = self._reorder(self.wdh_bwd(x_wdh.flip(1)).flip(1), 'w d h', 'd h w')

        # Merge all 6 directions
        merged = torch.cat([
            out_dhw_f, out_dhw_b,
            out_hwd_f, out_hwd_b,
            out_wdh_f, out_wdh_b,
        ], dim=-1)
        x = self.merge(merged)
        x = self.gelu(x)
        x = self.dropout(x)
        return x + residual


class CrossScanBiMamba3DLayer(nn.Module):
    """Stack of cross-scan BiMamba blocks for 3D volumes."""

    def __init__(
        self,
        dim: int,
        depth: int,
        spatial_dims: tuple,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            CrossScanBiMamba3DBlock(dim, spatial_dims, d_state, d_conv, expand, dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x
