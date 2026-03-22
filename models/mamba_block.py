# models/mamba_block.py
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    Mamba = None
    MAMBA_AVAILABLE = False

try:
    from mamba_ssm import Mamba3
    MAMBA3_AVAILABLE = True
    MAMBA3_IS_REAL = True
except (ImportError, AttributeError):
    try:
        from mamba_ssm import Mamba2 as Mamba3
        MAMBA3_AVAILABLE = True
        MAMBA3_IS_REAL = False
        import warnings
        warnings.warn("Real Mamba3 not available, falling back to Mamba2.", stacklevel=2)
    except ImportError:
        Mamba3 = None
        MAMBA3_AVAILABLE = False
        MAMBA3_IS_REAL = False


def _auto_headdim(d_inner: int) -> int:
    """Pick largest valid headdim from preferred list."""
    for hd in [64, 48, 32, 24, 16]:
        if d_inner % hd == 0:
            return hd
    return 1


def _create_ssm(
    dim: int,
    d_state: int,
    d_conv: int,
    expand: int,
    dropout: float = 0.0,
    use_mamba3: bool = False,
    headdim: int | None = None,
    rope_fraction: float | None = None,
    chunk_size: int | None = None,
    is_mimo: bool = False,
):
    """Create a Mamba SSM (v1 or v3/fallback) or MLP fallback.

    Args:
        dim: Model dimension (d_model).
        d_state: SSM state expansion factor.
        d_conv: Local convolution width (Mamba-1 only; ignored by Mamba3 path).
        expand: Block expansion factor.
        dropout: Dropout rate (unused by SSM; kept for API consistency).
        use_mamba3: If True, use Mamba3 or the Mamba2 compatibility fallback.
        headdim: Head dimension for Mamba3/Mamba2. Auto-selected if None.
            Must divide d_inner (dim * expand). Raises ValueError if invalid.
    """
    # Mamba3 Triton backward kernel crashes for d_model<96 (d_inner<192, nheads<3).
    # Verified: dim=48 fails on all seq_len/batch combos; dim>=96 works.
    # Fallback to Mamba-1 for small dims — only affects Stage 0 (embed_dim=48).
    _MAMBA3_MIN_DIM = 96

    if use_mamba3:
        if not MAMBA3_AVAILABLE:
            raise ImportError(
                "use_mamba3=True but mamba_ssm.Mamba3/Mamba2 is not available. "
                "Install mamba-ssm with Mamba3 support or a compatible Mamba2 fallback."
            )
        if dim < _MAMBA3_MIN_DIM:
            import warnings
            d_inner = int(dim * expand)
            hd = headdim or _auto_headdim(d_inner)
            warnings.warn(
                f"Mamba3 backward crashes for d_model={dim} < {_MAMBA3_MIN_DIM}. "
                f"Falling back to Mamba-2 for this layer.",
                stacklevel=2,
            )
            try:
                from mamba_ssm import Mamba2
                return Mamba2(d_model=dim, d_state=d_state, expand=expand, headdim=hd)
            except (ImportError, Exception):
                pass  # fall through to Mamba-1
        else:
            d_inner = int(dim * expand)
            hd = headdim or _auto_headdim(d_inner)
            if d_inner % hd != 0:
                raise ValueError(
                    f"headdim={hd} does not divide d_inner={d_inner} "
                    f"(dim={dim}, expand={expand}). "
                    f"Valid headdim values: {[h for h in [64, 48, 32, 24, 16] if d_inner % h == 0]}"
                )
            kwargs = dict(d_model=dim, d_state=d_state, expand=expand, headdim=hd)
            if MAMBA3_IS_REAL:
                if rope_fraction is not None:
                    kwargs["rope_fraction"] = rope_fraction
                if chunk_size is not None:
                    kwargs["chunk_size"] = chunk_size
                if is_mimo:
                    kwargs["is_mimo"] = True
            return Mamba3(**kwargs)

    if MAMBA_AVAILABLE:
        try:
            return Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        except (TypeError, RuntimeError) as e:
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
        use_mamba3: bool = False,
        headdim: int | None = None,
        rope_fraction: float | None = None,
        chunk_size: int | None = None,
        is_mimo: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = _create_ssm(
            dim,
            d_state,
            d_conv,
            expand,
            dropout,
            use_mamba3=use_mamba3,
            headdim=headdim,
            rope_fraction=rope_fraction,
            chunk_size=chunk_size,
            is_mimo=is_mimo,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, D] -> [B, L, D]"""
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = torch.clamp(x, -100, 100)  # Prevent SSM output explosion
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
        use_mamba3: bool = False,
        headdim: int | None = None,
        rope_fraction: float | None = None,
        chunk_size: int | None = None,
        is_mimo: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(
                dim,
                d_state,
                d_conv,
                expand,
                dropout,
                use_mamba3=use_mamba3,
                headdim=headdim,
                rope_fraction=rope_fraction,
                chunk_size=chunk_size,
                is_mimo=is_mimo,
            )
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
        use_mamba3: bool = False,
        headdim: int | None = None,
        rope_fraction: float | None = None,
        chunk_size: int | None = None,
        is_mimo: bool = False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        ssm_kw = dict(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            use_mamba3=use_mamba3,
            headdim=headdim,
            rope_fraction=rope_fraction,
            chunk_size=chunk_size,
            is_mimo=is_mimo,
        )
        self.forward_ssm = _create_ssm(**ssm_kw)
        self.backward_ssm = _create_ssm(**ssm_kw)
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
        use_mamba3: bool = False,
        headdim: int | None = None,
        rope_fraction: float | None = None,
        chunk_size: int | None = None,
        is_mimo: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            BiMambaBlock(
                dim,
                d_state,
                d_conv,
                expand,
                dropout,
                use_mamba3=use_mamba3,
                headdim=headdim,
                rope_fraction=rope_fraction,
                chunk_size=chunk_size,
                is_mimo=is_mimo,
            )
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
    """Cross-scan Mamba for 3D volumes.

    Scans along 3 different spatial axis orderings (forward-only):
      - D-H-W (depth-first)
      - H-W-D (height-first)
      - W-D-H (width-first)
    Total: 3 scan directions, merged with learned projection.

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
        use_mamba3: bool = False,
        headdim: int | None = None,
        rope_fraction: float | None = None,
        chunk_size: int | None = None,
        is_mimo: bool = False,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims  # (D, H, W) at this stage
        self.norm = nn.LayerNorm(dim)

        # 3D depthwise conv for local spatial inductive bias (UlikeMamba)
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

        # One SSM per axis ordering
        ssm_kw = dict(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            use_mamba3=use_mamba3,
            headdim=headdim,
            rope_fraction=rope_fraction,
            chunk_size=chunk_size,
            is_mimo=is_mimo,
        )
        self.dhw_fwd = _create_ssm(**ssm_kw)
        self.hwd_fwd = _create_ssm(**ssm_kw)
        self.wdh_fwd = _create_ssm(**ssm_kw)

        # Merge 3 directions -> dim
        self.merge = nn.Linear(dim * 3, dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Uncertainty-aware feature gating (UD-Mamba inspired)
        # At init: bias=0 => 2*sigmoid(0)=1.0 => identity-preserving
        self.uncertainty_head = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )
        nn.init.zeros_(self.uncertainty_head[-1].weight)
        nn.init.zeros_(self.uncertainty_head[-1].bias)

        dhw_indices = torch.arange(
            spatial_dims[0] * spatial_dims[1] * spatial_dims[2],
            dtype=torch.long,
        ).reshape(*spatial_dims)
        dhw_to_hwd_idx = dhw_indices.permute(1, 2, 0).reshape(-1)
        dhw_to_wdh_idx = dhw_indices.permute(2, 0, 1).reshape(-1)
        self.register_buffer('dhw_to_hwd_idx', dhw_to_hwd_idx, persistent=False)
        self.register_buffer('hwd_to_dhw_idx', torch.argsort(dhw_to_hwd_idx), persistent=False)
        self.register_buffer('dhw_to_wdh_idx', dhw_to_wdh_idx, persistent=False)
        self.register_buffer('wdh_to_dhw_idx', torch.argsort(dhw_to_wdh_idx), persistent=False)

    def _reorder(self, x: torch.Tensor, src: str, dst: str) -> torch.Tensor:
        """Reorder flattened spatial tokens between different axis orderings."""
        if src == dst:
            return x

        if src == 'd h w' and dst == 'h w d':
            index = self.dhw_to_hwd_idx
        elif src == 'h w d' and dst == 'd h w':
            index = self.hwd_to_dhw_idx
        elif src == 'd h w' and dst == 'w d h':
            index = self.dhw_to_wdh_idx
        elif src == 'w d h' and dst == 'd h w':
            index = self.wdh_to_dhw_idx
        else:
            raise ValueError(f'Unsupported reorder: {src} -> {dst}')

        return torch.index_select(x, dim=1, index=index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, D*H*W, C] -> [B, D*H*W, C]"""
        residual = x
        x = self.norm(x)
        D, H, W = self.spatial_dims
        B, L, C = x.shape

        # DWConv: local 3x3x3 spatial features
        x_3d = x.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        x = x + self.dwconv(x_3d).permute(0, 2, 3, 4, 1).reshape(B, L, C)

        # Scan 1: D-H-W ordering (native)
        out_dhw_f = self.dhw_fwd(x)

        # Scan 2: H-W-D ordering
        x_hwd = self._reorder(x, 'd h w', 'h w d')
        out_hwd_f = self._reorder(self.hwd_fwd(x_hwd), 'h w d', 'd h w')

        # Scan 3: W-D-H ordering
        x_wdh = self._reorder(x, 'd h w', 'w d h')
        out_wdh_f = self._reorder(self.wdh_fwd(x_wdh), 'w d h', 'd h w')

        # Merge all 3 directions
        merged = torch.cat([out_dhw_f, out_hwd_f, out_wdh_f], dim=-1)
        x = self.merge(merged)
        x = self.gelu(x)
        x = torch.clamp(x, -100, 100)  # Prevent cross-scan output explosion

        # Uncertainty gating: 2*sigmoid range (0, 2), identity-preserving at init
        unc_gate = 2.0 * torch.sigmoid(self.uncertainty_head(x))  # [B, L, 1]
        x = x * unc_gate

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
        use_mamba3: bool = False,
        headdim: int | None = None,
        rope_fraction: float | None = None,
        chunk_size: int | None = None,
        is_mimo: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            CrossScanBiMamba3DBlock(
                dim,
                spatial_dims,
                d_state,
                d_conv,
                expand,
                dropout,
                use_mamba3=use_mamba3,
                headdim=headdim,
                rope_fraction=rope_fraction,
                chunk_size=chunk_size,
                is_mimo=is_mimo,
            )
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x
