# models/text_encoder.py
import torch
import torch.nn as nn
from .mamba_block import MambaLayer


class TextMambaEncoder(nn.Module):
    """Text encoder using Mamba architecture."""

    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 256,
        max_len: int = 128,
        depth: int = 4,
        d_state: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Mamba layers
        self.mamba_layers = MambaLayer(
            dim=embed_dim,
            depth=depth,
            d_state=d_state,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] token indices
        Returns:
            [B, L, embed_dim] text features
        """
        B, L = input_ids.shape

        # Embed tokens
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :L, :]
        x = self.dropout(x)

        # Mamba encoding
        x = self.mamba_layers(x)
        x = self.norm(x)

        return x

    def get_global_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global feature via mean pooling.

        Args:
            x: [B, L, D] sequence features
        Returns:
            [B, D] global feature
        """
        return x.mean(dim=1)
