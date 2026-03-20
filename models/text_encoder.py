# models/text_encoder.py
"""Text encoder with optional PubMedBERT pretrained backbone."""

import torch
import torch.nn as nn
from .mamba_block import MambaLayer

try:
    from transformers import AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TextMambaEncoder(nn.Module):
    """Text encoder: PubMedBERT (frozen) + projection + Mamba adaptation.

    When transformers is available, uses PubMedBERT as a pretrained backbone
    with frozen weights, projecting 768-dim BERT features to embed_dim.
    Falls back to random nn.Embedding when transformers is not installed.
    """

    # PubMedBERT model name on HuggingFace
    PUBMEDBERT_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    BERT_HIDDEN_DIM = 768

    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 256,
        max_len: int = 128,
        depth: int = 4,
        d_state: int = 16,
        dropout: float = 0.1,
        use_pretrained: bool = True,
        unfreeze_last_n: int = 0,
        model_path: str | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_pretrained = use_pretrained and TRANSFORMERS_AVAILABLE
        self.model_path = model_path

        if self.use_pretrained:
            self._init_pretrained(embed_dim, depth, d_state, dropout, unfreeze_last_n)
        else:
            self._init_random(vocab_size, embed_dim, max_len, depth, d_state, dropout)

    def _init_pretrained(self, embed_dim, depth, d_state, dropout, unfreeze_last_n):
        """Initialize with frozen PubMedBERT + projection + Mamba."""
        load_path = self.model_path or self.PUBMEDBERT_NAME
        self.bert = AutoModel.from_pretrained(load_path)

        # Freeze all BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Optionally unfreeze last N encoder layers for fine-tuning
        if unfreeze_last_n > 0:
            for layer in self.bert.encoder.layer[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Cache frozen state to avoid per-forward parameter scan
        self._bert_frozen = not any(p.requires_grad for p in self.bert.parameters())

        # Project BERT 768 -> embed_dim
        self.proj = nn.Sequential(
            nn.Linear(self.BERT_HIDDEN_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Lightweight Mamba for task-specific adaptation
        self.mamba_layers = MambaLayer(
            dim=embed_dim,
            depth=max(1, depth // 2),  # Half the depth since BERT already processes
            d_state=d_state,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def _init_random(self, vocab_size, embed_dim, max_len, depth, d_state, dropout):
        """Initialize with random embeddings (fallback)."""
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.embed_dropout = nn.Dropout(dropout)

        self.mamba_layers = MambaLayer(
            dim=embed_dim,
            depth=depth,
            d_state=d_state,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] token indices
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
        Returns:
            [B, L, embed_dim] text features
        """
        if self.use_pretrained:
            return self._forward_pretrained(input_ids, attention_mask)
        return self._forward_random(input_ids)

    def _forward_pretrained(self, input_ids, attention_mask):
        """Forward with PubMedBERT backbone."""
        with torch.no_grad() if self._bert_frozen else torch.enable_grad():
            bert_out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden = bert_out.last_hidden_state  # [B, L, 768]

        # Cast BERT fp32 output to match proj dtype (pure bf16 mode: BERT fp32, proj bf16)
        hidden = hidden.to(self.proj[0].weight.dtype)

        # Project to embed_dim
        x = self.proj(hidden)  # [B, L, embed_dim]

        # Task-specific Mamba adaptation
        x = self.mamba_layers(x)
        x = self.norm(x)
        return x

    def _forward_random(self, input_ids):
        """Forward with random embeddings (fallback)."""
        B, L = input_ids.shape
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :L, :]
        x = self.embed_dropout(x)
        x = self.mamba_layers(x)
        x = self.norm(x)
        return x

    def get_global_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global feature — CLS token (index 0) for pretrained, mean pooling for random.

        Args:
            x: [B, L, D] sequence features
        Returns:
            [B, D] global feature
        """
        if self.use_pretrained:
            return x[:, 0, :]  # CLS token
        return x.mean(dim=1)
