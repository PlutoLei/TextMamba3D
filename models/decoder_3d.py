# models/decoder_3d.py
import torch
import torch.nn as nn
from einops import rearrange
from .mamba_block import CrossScanBiMamba3DLayer


class PatchExpanding3D(nn.Module):
    """Patch expanding for upsampling."""

    def __init__(self, dim: int, out_dim: int, spatial_dims: tuple):
        super().__init__()
        self.dim = dim
        self.spatial_dims = spatial_dims
        self.expand = nn.Linear(dim, 8 * out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D*H*W, C]
        Returns:
            [B, 8*D*H*W, C/2]
        """
        B, L, C = x.shape
        D, H, W = self.spatial_dims

        x = self.expand(x)  # [B, L, 8*out_dim]
        x = rearrange(x, 'b (d h w) (p1 p2 p3 c) -> b (d p1 h p2 w p3) c',
                     d=D, h=H, w=W, p1=2, p2=2, p3=2)
        x = self.norm(x)

        return x


class MambaDecoder3D(nn.Module):
    """3D Mamba Decoder with skip connections."""

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
    ):
        super().__init__()
        self.num_stages = len(depths)

        # Calculate spatial dimensions
        d, h, w = img_size[0] // patch_size[0], \
                  img_size[1] // patch_size[1], \
                  img_size[2] // patch_size[2]

        self.stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_projs = nn.ModuleList()

        # Build decoder stages (reverse order)
        for i in range(len(depths) - 1, -1, -1):
            dim = embed_dim * (2 ** i)
            spatial = (d // (2 ** i), h // (2 ** i), w // (2 ** i))

            # CrossScan BiMamba stage (6-direction 3D scanning)
            stage = CrossScanBiMamba3DLayer(
                dim=dim,
                depth=depths[i],
                spatial_dims=spatial,
                d_state=d_state,
                dropout=dropout,
                use_checkpoint=use_checkpoint,
            )
            self.stages.append(stage)

            # Upsample (except first decoder stage)
            if i > 0:
                spatial = (d // (2 ** i), h // (2 ** i), w // (2 ** i))
                upsample = PatchExpanding3D(
                    dim=dim,
                    out_dim=dim // 2,
                    spatial_dims=spatial,
                )
                self.upsamples.append(upsample)

                # Skip connection projection
                # After upsampling, we have dim//2, skip features also have dim//2
                skip_proj = nn.Linear(dim // 2, dim // 2)
                self.skip_projs.append(skip_proj)

        # Final projection to output
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
            features: List of encoder features [stage1, stage2, ..., bottleneck]
        Returns:
            [B, out_channels, D, H, W]
        """
        # Start from bottleneck
        x = features[-1]

        for i, stage in enumerate(self.stages):
            x = stage(x)

            if i < len(self.upsamples):
                x = self.upsamples[i](x)

                # Skip connection
                skip_idx = len(features) - 2 - i
                if skip_idx >= 0:
                    skip = self.skip_projs[i](features[skip_idx])
                    x = x + skip

        # Final expansion to original resolution
        B, L, C = x.shape
        d, h, w = self.base_spatial
        p = self.patch_size[0]

        x = self.final_expand(x)  # [B, L, C * p^3]
        x = rearrange(x, 'b (d h w) (p1 p2 p3 c) -> b c (d p1) (h p2) (w p3)',
                     d=d, h=h, w=w, p1=p, p2=p, p3=p, c=C)
        x = self.final_proj(x)

        return x
