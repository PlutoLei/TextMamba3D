# models/encoder_3d.py
import torch
import torch.nn as nn
from math import prod
from einops import rearrange
from .mamba_block import CrossScanBiMamba3DLayer


class PatchEmbed3D(nn.Module):
    """3D Image to Patch Embedding."""

    def __init__(
        self,
        img_size: tuple = (96, 96, 96),
        patch_size: tuple = (4, 4, 4),
        in_channels: int = 4,
        embed_dim: int = 96,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * \
                          (img_size[1] // patch_size[1]) * \
                          (img_size[2] // patch_size[2])

        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            [B, num_patches, embed_dim]
        """
        x = self.proj(x)  # [B, embed_dim, D', H', W']
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        x = self.norm(x)
        return x


class ModalityGroupPatchEmbed3D(nn.Module):
    """Physics-based modality grouping for BraTS 4-channel input.

    Splits 4 MRI channels into physics-based groups:
      - Group A: T1 + T1ce (anatomical structure, high SNR)
      - Group B: T2 + FLAIR (lesion signal, edema-sensitive)
    Each group gets independent patch embedding, then merged.
    Validated by CKD-TransBTS (physics-aware grouping > naive concatenation).
    """

    def __init__(
        self,
        img_size: tuple = (96, 96, 96),
        patch_size: tuple = (4, 4, 4),
        in_channels: int = 4,
        embed_dim: int = 96,
    ):
        super().__init__()
        assert in_channels == 4, f"ModalityGroupPatchEmbed3D requires 4 channels, got {in_channels}"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = prod(s // p for s, p in zip(img_size, patch_size))

        # Group A: T1 + T1ce (channels 0-1)
        self.group_a = nn.Conv3d(2, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Group B: T2 + FLAIR (channels 2-3)
        self.group_b = nn.Conv3d(2, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Merge two groups
        self.merge = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 4, D, H, W]
        Returns:
            [B, num_patches, embed_dim]
        """
        a = rearrange(self.group_a(x[:, :2]), 'b c d h w -> b (d h w) c')
        b = rearrange(self.group_b(x[:, 2:]), 'b c d h w -> b (d h w) c')
        x = self.merge(torch.cat([a, b], dim=-1))
        return self.norm(x)


class PatchMerging3D(nn.Module):
    """Patch merging for downsampling."""

    def __init__(self, dim: int, spatial_dims: tuple):
        super().__init__()
        self.dim = dim
        self.spatial_dims = spatial_dims
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(8 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D*H*W, C]
        Returns:
            [B, D*H*W/8, 2*C]
        """
        B, L, C = x.shape
        D, H, W = self.spatial_dims

        x = rearrange(x, 'b (d h w) c -> b d h w c', d=D, h=H, w=W)

        # Merge 2x2x2 patches
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)
        x = rearrange(x, 'b d h w c -> b (d h w) c')

        x = self.norm(x)
        x = self.reduction(x)
        return x


class MambaEncoder3D(nn.Module):
    """3D Mamba Encoder with hierarchical features."""

    def __init__(
        self,
        img_size: tuple = (96, 96, 96),
        in_channels: int = 4,
        embed_dim: int = 96,
        depths: list = [2, 2, 2, 2],
        patch_size: tuple = (4, 4, 4),
        d_state: int = 16,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.num_stages = len(depths)
        self.embed_dim = embed_dim

        # Patch embedding (modality grouping for BraTS 4-channel input)
        if in_channels == 4:
            self.patch_embed = ModalityGroupPatchEmbed3D(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed = PatchEmbed3D(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=embed_dim,
            )

        # Calculate spatial dimensions at each stage
        d, h, w = img_size[0] // patch_size[0], \
                  img_size[1] // patch_size[1], \
                  img_size[2] // patch_size[2]

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i, depth in enumerate(depths):
            # CrossScan BiMamba stage (6-direction 3D scanning)
            dim = embed_dim * (2 ** i)
            spatial = (d // (2 ** i), h // (2 ** i), w // (2 ** i))
            stage = CrossScanBiMamba3DLayer(
                dim=dim,
                depth=depth,
                spatial_dims=spatial,
                d_state=d_state,
                dropout=dropout,
                use_checkpoint=use_checkpoint,
            )
            self.stages.append(stage)

            # Downsample (except last stage)
            if i < len(depths) - 1:
                downsample = PatchMerging3D(
                    dim=dim,
                    spatial_dims=(d // (2 ** i), h // (2 ** i), w // (2 ** i)),
                )
                self.downsamples.append(downsample)

        self.spatial_dims = [
            (d // (2 ** i), h // (2 ** i), w // (2 ** i))
            for i in range(len(depths))
        ]

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            List of features at each stage
        """
        x = self.patch_embed(x)

        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)

            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        return features
