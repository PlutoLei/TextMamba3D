# models/textmamba3d.py
"""Text-guided 3D medical image segmentation with Mamba architecture."""

from typing import Optional

import torch
import torch.nn as nn

from .decoder_3d import MambaDecoder3D
from .encoder_3d import MambaEncoder3D
from .fusion import MultiScalePixelTextAttention
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
        text_max_len: int = 256,
        text_depth: int = 4,
        d_state: int = 16,
        dropout: float = 0.0,
        use_pretrained_text: bool = True,
        unfreeze_text_layers: int = 0,
        use_checkpoint: bool = False,
        text_model_path: str | None = None,
        deep_supervision: bool = False,
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
            model_path=text_model_path,
        )

        # Multi-scale cross-attention: text guides skip connections at stages 1,2,3
        # Stage 0 excluded (32K tokens too expensive for cross-attention)
        stage_dims = [embed_dim * (2 ** i) for i in range(1, len(depths))]
        self.multi_scale_attn = MultiScalePixelTextAttention(
            stage_dims=stage_dims,     # [96, 192, 384] for embed_dim=48
            text_dim=text_embed_dim,   # 256
            num_heads=4,               # head_dim varies per stage
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
            deep_supervision=deep_supervision,
        )

        self.img_proj = nn.Sequential(
            nn.Linear(bottleneck_dim, text_embed_dim),
            nn.LayerNorm(text_embed_dim),
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
        img_features = self.img_encoder(img)

        has_text = use_text and text_ids is not None
        if has_text:
            text_features = self.text_encoder(text_ids, attention_mask)
            # Multi-scale fusion: stages 1,2,3 get text cross-attention
            # Stage 0 stays raw (32K tokens, too expensive for cross-attention)
            fused = self.multi_scale_attn(
                img_features[1:], text_features, attention_mask
            )
            decoder_features = [img_features[0]] + fused
        else:
            # Bypass fusion entirely for text-free path
            decoder_features = img_features

        seg_output = self.decoder(decoder_features)

        if not return_features:
            return seg_output

        if has_text:
            # Contrastive: project fused bottleneck (last in decoder_features) for alignment
            img_global = self.img_proj(decoder_features[-1].mean(dim=1))
            text_global = self.text_encoder.get_global_feature(text_features)
            return seg_output, img_global, text_global
        else:
            # No contrastive on text-free batches: return None features
            return seg_output, None, None

    def forward_without_text(self, img: torch.Tensor) -> torch.Tensor:
        """Convenience method for inference without text guidance."""
        return self.forward(img, text_ids=None, use_text=False)
