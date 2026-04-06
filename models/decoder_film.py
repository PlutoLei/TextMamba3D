# models/decoder_film.py
"""Decoder-side FiLM conditioning: text modulates features INSIDE the decoder,
after skip connection merge. This forces the decoder to route through text,
unlike encoder-side fusion which the decoder can ignore.

Reference: Diffusion model insight — conditioning works when there's an
information deficit. Feature noise creates the deficit, FiLM resolves it.
"""

import torch
import torch.nn as nn


class DecoderFiLM(nn.Module):
    """FiLM layer for decoder skip connections.

    Applied AFTER skip + upsample merge, so the decoder cannot bypass it.
    Uses (1 + gamma) * x + beta formulation with zero-init for safe start.
    """

    def __init__(self, feat_dim: int, text_dim: int):
        super().__init__()
        self.proj = nn.Linear(text_dim, feat_dim * 2)
        # Zero-init: gamma=0, beta=0 → output = (1+0)*x + 0 = x (identity)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, text_global: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] decoder features (after skip merge)
            text_global: [B, D_text] text CLS token
        Returns:
            [B, L, D] text-conditioned features
        """
        params = self.proj(text_global).unsqueeze(1)  # [B, 1, 2*D]
        gamma, beta = params.chunk(2, dim=-1)         # each [B, 1, D]
        return (1 + gamma) * x + beta


class FeatureNoiseInjector(nn.Module):
    """Inject structured noise into backbone features during training.

    Creates an information deficit that text must fill.
    Inspired by the diffusion model paradigm: noise creates the NEED for conditioning.

    At inference, noise_rate=0 (no noise), but the text pathway has learned
    to provide refinement that works even without noise.
    """

    def __init__(self, noise_rate: float = 0.3):
        super().__init__()
        self.noise_rate = noise_rate

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply channel dropout to each feature tensor.

        Args:
            features: list of [B, L, D] encoder features
        Returns:
            list of [B, L, D] with random channels zeroed out
        """
        if not self.training or self.noise_rate <= 0:
            return features

        noised = []
        for feat in features:
            B, L, D = feat.shape
            # Per-channel binary mask (same across spatial dims)
            mask = torch.ones(B, 1, D, device=feat.device)
            mask = torch.bernoulli(mask * (1 - self.noise_rate))
            # Scale to preserve expected magnitude
            noised.append(feat * mask / (1 - self.noise_rate))
        return noised
