# models/decoder_3d.py
import torch
import torch.nn as nn
from einops import rearrange
from .mamba_block import CrossScanBiMamba3DLayer
from .fusion import CrossScaleSkipAttention


class PatchExpanding3D(nn.Module):
    """Patch expanding for upsampling."""

    def __init__(self, dim: int, out_dim: int, spatial_dims: tuple):
        super().__init__()
        self.dim = dim
        self.spatial_dims = spatial_dims
        self.expand = nn.Linear(dim, 8 * out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        D, H, W = self.spatial_dims
        x = self.expand(x)
        x = rearrange(x, 'b (d h w) (p1 p2 p3 c) -> b (d p1 h p2 w p3) c',
                     d=D, h=H, w=W, p1=2, p2=2, p3=2)
        x = self.norm(x)
        return x


class MambaDecoder3D(nn.Module):
    """3D Mamba Decoder with skip connections.

    V4.6: Optional CrossScaleSkipAttention supplements (not replaces) the
    existing matched-level skip projection with cross-scale context.
    """

    def __init__(
        self,
        img_size: tuple = (96, 96, 96),
        patch_size: tuple = (4, 4, 4),
        out_channels: int = 4,
        embed_dim: int = 96,
        depths: list = [2, 2, 2, 2],
        d_state: int = 16,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
        deep_supervision: bool = False,
        use_cross_scale_skip: bool = False,
        use_mamba3: bool = False,
        headdim: int | None = None,
    ):
        super().__init__()
        self.num_stages = len(depths)
        self.deep_supervision = deep_supervision
        self.use_cross_scale_skip = use_cross_scale_skip

        d, h, w = img_size[0] // patch_size[0], \
                  img_size[1] // patch_size[1], \
                  img_size[2] // patch_size[2]

        self.stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_projs = nn.ModuleList()

        # V4.6: cross-scale attention modules (one per skip connection)
        self.cross_scale_attns = nn.ModuleList() if use_cross_scale_skip else None

        skip_count = 0

        for i in range(len(depths) - 1, -1, -1):
            dim = embed_dim * (2 ** i)
            spatial = (d // (2 ** i), h // (2 ** i), w // (2 ** i))

            stage = CrossScanBiMamba3DLayer(
                dim=dim,
                depth=depths[i],
                spatial_dims=spatial,
                d_state=d_state,
                dropout=dropout,
                use_checkpoint=use_checkpoint,
                use_mamba3=use_mamba3,
                headdim=headdim,
            )
            self.stages.append(stage)

            if i > 0:
                spatial = (d // (2 ** i), h // (2 ** i), w // (2 ** i))
                upsample = PatchExpanding3D(
                    dim=dim,
                    out_dim=dim // 2,
                    spatial_dims=spatial,
                )
                self.upsamples.append(upsample)

                # Original skip projection (ALWAYS present)
                skip_proj = nn.Linear(dim // 2, dim // 2)
                self.skip_projs.append(skip_proj)

                # V4.6: cross-scale attention (supplemental)
                if use_cross_scale_skip:
                    target_dim = dim // 2
                    skip_idx = len(depths) - 2 - skip_count
                    candidate_dims = [embed_dim * (2 ** j) for j in range(skip_idx + 1)]
                    self.cross_scale_attns.append(
                        CrossScaleSkipAttention(
                            target_dim=target_dim,
                            candidate_dims=candidate_dims,
                        )
                    )
                    skip_count += 1

        # Deep supervision
        if deep_supervision:
            self.aux_heads = nn.ModuleList()
            self.aux_spatials = []
            for idx, i in enumerate(range(len(depths) - 1, 0, -1)):
                dim = embed_dim * (2 ** i)
                spatial = (d // (2 ** i), h // (2 ** i), w // (2 ** i))
                self.aux_heads.append(nn.Conv3d(dim, out_channels, 1))
                self.aux_spatials.append(spatial)

        self.final_expand = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * (patch_size[0] ** 3)),
            nn.GELU(),
        )
        self.final_proj = nn.Conv3d(embed_dim, out_channels, 1)

        self.patch_size = patch_size
        self.base_spatial = (d, h, w)

    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: List of encoder features [stage0, stage1, ..., bottleneck]
        Returns:
            [B, out_channels, D, H, W]
        """
        self._aux_outputs = []
        x = features[-1]

        for i, stage in enumerate(self.stages):
            x = stage(x)

            if self.deep_supervision and self.training and i < len(self.aux_heads):
                B_aux, L_aux, C_aux = x.shape
                ds, hs, ws = self.aux_spatials[i]
                aux = rearrange(x, 'b (d h w) c -> b c d h w', d=ds, h=hs, w=ws)
                self._aux_outputs.append(self.aux_heads[i](aux))

            if i < len(self.upsamples):
                x = self.upsamples[i](x)

                # Skip connection
                skip_idx = len(features) - 2 - i
                if skip_idx >= 0:
                    # Original matched-level skip (always)
                    skip = self.skip_projs[i](features[skip_idx])
                    x = x + skip

                    # V4.6: cross-scale supplemental attention
                    if self.use_cross_scale_skip and self.cross_scale_attns is not None:
                        candidates = features[:skip_idx + 1]
                        x = x + self.cross_scale_attns[i](x, candidates)

        # Final expansion
        B, L, C = x.shape
        d, h, w = self.base_spatial
        p = self.patch_size[0]

        x = self.final_expand(x)
        x = rearrange(x, 'b (d h w) (p1 p2 p3 c) -> b c (d p1) (h p2) (w p3)',
                     d=d, h=h, w=w, p1=p, p2=p, p3=p, c=C)
        x = self.final_proj(x)

        return x
