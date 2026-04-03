"""MFuser-style concatenate+scan fusion for Mamba backbones.

Instead of cross-attention (which doesn't work with Mamba features),
compress text into K prompt tokens, concatenate with image tokens,
and process through a Mamba block. The selective scan implicitly
performs cross-modal interaction.

Reference: MFuser (CVPR 2025 Highlight)
"""

import torch
import torch.nn as nn
from .mamba_block import _create_ssm


class TextPromptCompressor(nn.Module):
    """Compress variable-length text sequence into K fixed prompt tokens."""

    def __init__(self, text_dim: int, feat_dim: int, num_prompts: int = 8):
        super().__init__()
        self.num_prompts = num_prompts
        self.queries = nn.Parameter(torch.randn(1, num_prompts, feat_dim) * 0.02)
        self.kv_proj = nn.Linear(text_dim, feat_dim)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, text_feat: torch.Tensor) -> torch.Tensor:
        B = text_feat.size(0)
        kv = self.kv_proj(text_feat)
        q = self.queries.expand(B, -1, -1)
        attn = (q @ kv.transpose(-1, -2)) * (q.size(-1) ** -0.5)
        attn = attn.softmax(dim=-1)
        prompts = attn @ kv
        return self.norm(prompts)


class MambaConcatFusion(nn.Module):
    """Concatenate text prompts with image tokens, scan with Mamba.

    1. Compress text -> K prompt tokens
    2. Concat [prompts, image_tokens] -> [B, K+N, D]
    3. Mamba scan (selective scan does implicit cross-modal interaction)
    4. Extract image portion [B, N, D]
    5. Residual connection
    """

    def __init__(
        self,
        feat_dim: int,
        text_dim: int,
        num_prompts: int = 8,
        d_state: int = 16,
        expand: int = 2,
        use_mamba3: bool = True,
    ):
        super().__init__()
        self.compressor = TextPromptCompressor(text_dim, feat_dim, num_prompts)
        self.num_prompts = num_prompts
        self.norm = nn.LayerNorm(feat_dim)
        self.ssm = _create_ssm(
            dim=feat_dim, d_state=d_state, d_conv=4, expand=expand,
            use_mamba3=use_mamba3,
        )
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        # Small init — non-zero but near-identity
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.weight.data *= 0.01
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        text_feat: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        K = self.num_prompts
        prompts = self.compressor(text_feat)
        x_norm = self.norm(x)
        concat = torch.cat([prompts, x_norm], dim=1)
        scanned = self.ssm(concat)
        img_out = scanned[:, K:, :]
        return residual + self.out_proj(img_out)


class MultiScaleConcatFusion(nn.Module):
    """Apply MambaConcatFusion at multiple encoder scales."""

    def __init__(
        self,
        stage_dims: list[int],
        text_dim: int,
        num_prompts: int = 8,
        d_state: int = 16,
        use_mamba3: bool = True,
        **kwargs,  # accept num_heads etc for drop-in compatibility
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaConcatFusion(dim, text_dim, num_prompts, d_state,
                              use_mamba3=use_mamba3)
            for dim in stage_dims
        ])

    def forward(
        self,
        features: list[torch.Tensor],
        text_feat: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        return [
            layer(feat, text_feat, text_mask)
            for layer, feat in zip(self.layers, features)
        ]
