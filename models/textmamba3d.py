# models/textmamba3d.py
"""Text-guided 3D medical image segmentation with Mamba architecture."""

from typing import Optional

import torch
import torch.nn as nn

from .decoder_3d import MambaDecoder3D
from .encoder_3d import MambaEncoder3D
from .fusion import MambaFusion, MultiScaleFiLM
from .text_encoder import TextMambaEncoder


class TextMamba3D(nn.Module):
    """Text-guided 3D medical image segmentation model using Mamba architecture."""

    def __init__(
        self,
        img_size: tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 4,
        out_channels: int = 4,
        embed_dim: int = 96,
        depths: list[int] = [2, 2, 2, 2],
        patch_size: tuple[int, int, int] = (4, 4, 4),
        text_embed_dim: int = 256,
        text_max_len: int = 128,
        text_depth: int = 4,
        fusion_depth: int = 2,
        d_state: int = 16,
        dropout: float = 0.0,
        use_pretrained_text: bool = True,
        unfreeze_text_layers: int = 0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.text_embed_dim = text_embed_dim
        self.text_max_len = text_max_len
        bottleneck_dim = embed_dim * (2 ** (len(depths) - 1))

        self.img_encoder = MambaEncoder3D(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            patch_size=patch_size,
            d_state=d_state,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
        )

        self.text_encoder = TextMambaEncoder(
            embed_dim=text_embed_dim,
            max_len=text_max_len,
            depth=text_depth,
            d_state=d_state,
            dropout=dropout,
            use_pretrained=use_pretrained_text,
            unfreeze_last_n=unfreeze_text_layers,
        )

        self.fusion = MambaFusion(
            img_dim=bottleneck_dim,
            text_dim=text_embed_dim,
            hidden_dim=bottleneck_dim,
            depth=fusion_depth,
            d_state=d_state,
            dropout=dropout,
        )

        self.decoder = MambaDecoder3D(
            img_size=img_size,
            patch_size=patch_size,
            out_channels=out_channels,
            embed_dim=embed_dim,
            depths=depths,
            d_state=d_state,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
        )

        # Multi-scale FiLM: text guides all encoder stages, not just bottleneck
        stage_dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.multi_scale_film = MultiScaleFiLM(
            stage_dims=stage_dims,
            text_dim=text_embed_dim,
        )

        self.img_proj = nn.Sequential(
            nn.Linear(bottleneck_dim, text_embed_dim),
            nn.LayerNorm(text_embed_dim),
        )

        self.default_text_embed = nn.Parameter(
            torch.randn(1, text_max_len, text_embed_dim) * 0.02
        )

    def forward(
        self,
        img: torch.Tensor,
        text_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
        use_text: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for text-guided 3D segmentation.

        Args:
            img: Input image tensor of shape [B, C, D, H, W]
            text_ids: Text token indices of shape [B, L], optional for inference
            attention_mask: [B, L] mask for text padding (1=valid, 0=pad)
            return_features: Whether to return features for contrastive loss
            use_text: Whether to use text guidance (False for text-free inference)

        Returns:
            Segmentation output [B, out_channels, D, H, W], and optionally
            (img_feat, text_feat) for contrastive loss
        """
        batch_size = img.shape[0]

        img_features = self.img_encoder(img)
        bottleneck = img_features[-1]

        has_text = use_text and text_ids is not None
        if has_text:
            text_features = self.text_encoder(text_ids, attention_mask)
        else:
            text_features = self.default_text_embed.expand(batch_size, -1, -1)

        # Text global feature for FiLM and contrastive loss
        text_global = (
            self.text_encoder.get_global_feature(text_features)
            if has_text
            else text_features.mean(dim=1)
        )

        # Multi-scale FiLM: modulate ALL encoder features with text
        filmed_features = self.multi_scale_film(img_features, text_global)

        # Deep bottleneck fusion (causal Mamba)
        fused_bottleneck = self.fusion(filmed_features[-1], text_features)

        # Decoder: FiLM-modulated skip connections + fused bottleneck
        decoder_features = filmed_features[:-1] + [fused_bottleneck]
        seg_output = self.decoder(decoder_features)

        if not return_features:
            return seg_output

        img_global = self.img_proj(bottleneck.mean(dim=1))
        return seg_output, img_global, text_global

    def forward_without_text(self, img: torch.Tensor) -> torch.Tensor:
        """Convenience method for inference without text guidance."""
        return self.forward(img, text_ids=None, use_text=False)
