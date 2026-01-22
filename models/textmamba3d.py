# models/textmamba3d.py
import torch
import torch.nn as nn
from .encoder_3d import MambaEncoder3D
from .text_encoder import TextMambaEncoder
from .fusion import MambaFusion
from .decoder_3d import MambaDecoder3D


class TextMamba3D(nn.Module):
    """Text-guided 3D medical image segmentation with Mamba architecture."""

    def __init__(
        self,
        img_size: tuple = (96, 96, 96),
        in_channels: int = 4,
        out_channels: int = 4,
        embed_dim: int = 96,
        depths: list = [2, 2, 2, 2],
        patch_size: tuple = (4, 4, 4),
        text_embed_dim: int = 256,
        text_max_len: int = 128,
        text_depth: int = 4,
        fusion_depth: int = 2,
        d_state: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        # 3D Image Encoder
        self.img_encoder = MambaEncoder3D(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            patch_size=patch_size,
            d_state=d_state,
            dropout=dropout,
        )

        # Text Encoder
        self.text_encoder = TextMambaEncoder(
            embed_dim=text_embed_dim,
            max_len=text_max_len,
            depth=text_depth,
            d_state=d_state,
            dropout=dropout,
        )

        # Fusion at bottleneck
        bottleneck_dim = embed_dim * (2 ** (len(depths) - 1))
        self.fusion = MambaFusion(
            img_dim=bottleneck_dim,
            text_dim=text_embed_dim,
            hidden_dim=bottleneck_dim,
            depth=fusion_depth,
            d_state=d_state,
            dropout=dropout,
        )

        # Decoder
        self.decoder = MambaDecoder3D(
            img_size=img_size,
            patch_size=patch_size,
            out_channels=out_channels,
            embed_dim=embed_dim,
            depths=depths,
            d_state=d_state,
            dropout=dropout,
        )

        # Feature projection for contrastive loss
        self.img_proj = nn.Sequential(
            nn.Linear(bottleneck_dim, text_embed_dim),
            nn.LayerNorm(text_embed_dim),
        )

    def forward(
        self,
        img: torch.Tensor,
        text_ids: torch.Tensor,
        return_features: bool = False,
    ):
        """
        Args:
            img: [B, C, D, H, W] input image
            text_ids: [B, L] text token indices
            return_features: whether to return features for contrastive loss
        Returns:
            seg_output: [B, out_channels, D, H, W]
            img_feat: [B, text_embed_dim] (if return_features)
            text_feat: [B, text_embed_dim] (if return_features)
        """
        # Encode image
        img_features = self.img_encoder(img)

        # Encode text
        text_features = self.text_encoder(text_ids)

        # Fuse at bottleneck
        bottleneck = img_features[-1]
        fused_bottleneck = self.fusion(bottleneck, text_features)

        # Replace bottleneck with fused features
        decoder_features = img_features[:-1] + [fused_bottleneck]

        # Decode
        seg_output = self.decoder(decoder_features)

        if return_features:
            # Global features for contrastive loss
            img_global = self.img_proj(bottleneck.mean(dim=1))
            text_global = self.text_encoder.get_global_feature(text_features)
            return seg_output, img_global, text_global

        return seg_output
