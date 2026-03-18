# models/textmamba3d.py
"""Text-guided 3D medical image segmentation with Mamba architecture."""

from typing import Optional

import torch
import torch.nn as nn

from .decoder_3d import MambaDecoder3D
from .encoder_3d import MambaEncoder3D
from .fusion import MultiScalePixelTextAttention, MultiScaleTextGate
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
        # V4.6 new parameters
        use_text_gate: bool = False,
        use_cross_scale_skip: bool = False,
        text_gate_init_bias: float = 2.0,
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

        # Multi-scale cross-attention: stages 1,2,3 (stage 0 excluded)
        stage_dims = [embed_dim * (2 ** i) for i in range(1, len(depths))]
        self.multi_scale_attn = MultiScalePixelTextAttention(
            stage_dims=stage_dims,
            text_dim=text_embed_dim,
            num_heads=4,
        )

        # V4.6 Direction B: Text Scale Gate
        if use_text_gate:
            self.text_gate = MultiScaleTextGate(
                stage_dims=stage_dims,
                init_bias=text_gate_init_bias,
            )
        else:
            self.text_gate = None

        # V4.6 Direction A: pass use_cross_scale_skip to decoder
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
            use_cross_scale_skip=use_cross_scale_skip,
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
    ) -> torch.Tensor | tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Forward pass for text-guided 3D segmentation."""
        img_features = self.img_encoder(img)

        has_text = use_text and text_ids is not None
        if has_text:
            text_features = self.text_encoder(text_ids, attention_mask)
            fused = self.multi_scale_attn(
                img_features[1:], text_features, attention_mask
            )
            # V4.6 Direction B: gate text contribution per scale
            if self.text_gate is not None:
                fused = self.text_gate(img_features[1:], fused)
            decoder_features = [img_features[0]] + fused
        else:
            decoder_features = img_features

        seg_output = self.decoder(decoder_features)

        if not return_features:
            return seg_output

        if has_text:
            pixel_feat = decoder_features[-1]
            img_global = self.img_proj(pixel_feat.mean(dim=1))
            text_global = self.text_encoder.get_global_feature(text_features)
            return seg_output, img_global, text_global, pixel_feat
        else:
            return seg_output, None, None, None

    def forward_without_text(self, img: torch.Tensor) -> torch.Tensor:
        """Convenience method for inference without text guidance."""
        return self.forward(img, text_ids=None, use_text=False)
