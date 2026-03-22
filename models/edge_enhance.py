# models/edge_enhance.py
"""Learnable Edge Enhancement module for decoder skip connections.

Applies lightweight depthwise convolution + sigmoid attention to enhance
boundary features in skip connections, improving small object segmentation.
"""
import torch
import torch.nn as nn
from einops import rearrange


class EdgeEnhance3D(nn.Module):
    """Lightweight boundary attention for 3D skip connection features.

    Reshapes sequence tokens to 3D spatial format, applies depthwise
    separable convolution to detect local edge patterns, produces a
    sigmoid attention map, and applies it with a residual connection.

    Args:
        channels: Feature dimension (C).
        spatial_dims: (D, H, W) spatial dimensions at this decoder stage.
    """

    def __init__(self, channels: int, spatial_dims: tuple[int, int, int]):
        super().__init__()
        self.spatial_dims = spatial_dims

        self.conv = nn.Sequential(
            # Depthwise: local spatial edge detection
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU(),
            # Pointwise: channel mixing -> attention logits
            nn.Conv3d(channels, channels, kernel_size=1, bias=True),
        )

        # Initialize sigmoid bias to -3 for near-identity at start
        # sigmoid(-3) ~ 0.047, so output ~ 0.047*x + x = 1.047*x
        nn.init.constant_(self.conv[-1].bias, -3.0)
        # Zero-init pointwise weights for even closer to identity
        nn.init.zeros_(self.conv[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] sequence features from skip connection
        Returns:
            [B, N, C] boundary-enhanced features
        """
        D, H, W = self.spatial_dims
        assert x.shape[1] == D * H * W, f"Expected N={D*H*W}, got {x.shape[1]}"
        x_3d = rearrange(x, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)

        attn = torch.sigmoid(self.conv(x_3d))
        out_3d = x_3d * attn + x_3d

        return rearrange(out_3d, 'b c d h w -> b (d h w) c')
