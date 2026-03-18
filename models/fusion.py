# models/fusion.py
import torch
import torch.nn as nn
from .mamba_block import MambaLayer


# ---------------------------------------------------------------------------
# FiLM: Feature-wise Linear Modulation for multi-scale text guidance
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: output = gamma * input + beta.

    Predicts per-channel scale and shift from a conditioning vector,
    then applies them to every spatial token.
    """

    def __init__(self, feat_dim: int, cond_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, feat_dim)
        self.beta_proj = nn.Linear(cond_dim, feat_dim)

        # Bounded gating: gamma = 2*sigmoid(raw), so gamma in (0, 2).
        # Warm init: small random weights give a weak text modulation signal
        # from the start, accelerating text-guided learning.
        # gamma ≈ 2*sigmoid(~0) ≈ 1.0 ± 0.005, beta ≈ ± 0.005.
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.normal_(self.gamma_proj.weight, std=0.01)
        nn.init.zeros_(self.beta_proj.bias)
        nn.init.normal_(self.beta_proj.weight, std=0.01)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] spatial feature tokens
            cond: [B, D_cond] conditioning vector (text global feature)
        Returns:
            [B, L, D] modulated features
        """
        raw_gamma = self.gamma_proj(cond).unsqueeze(1)   # [B, 1, D]
        gamma = 2.0 * torch.sigmoid(raw_gamma)            # bounded (0, 2)
        beta = self.beta_proj(cond).unsqueeze(1)           # [B, 1, D]
        return gamma * x + beta


class MultiScaleFiLM(nn.Module):
    """Apply FiLM modulation at multiple encoder scales.

    One FiLMLayer per encoder stage, conditioned on text global feature.
    Modulates skip connections before they enter the decoder.
    """

    def __init__(self, stage_dims: list[int], text_dim: int):
        """
        Args:
            stage_dims: Channel dimensions for each encoder stage, e.g. [96, 192, 384, 768]
            text_dim: Dimension of text global feature
        """
        super().__init__()
        self.film_layers = nn.ModuleList([
            FiLMLayer(feat_dim=dim, cond_dim=text_dim)
            for dim in stage_dims
        ])

    def forward(
        self,
        features: list[torch.Tensor],
        text_global: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Args:
            features: List of encoder features [stage0, stage1, ..., stageN]
            text_global: [B, D_text] text global feature
        Returns:
            List of FiLM-modulated features (same shapes)
        """
        return [
            film(feat, text_global)
            for film, feat in zip(self.film_layers, features)
        ]


# ---------------------------------------------------------------------------
# Pixel-Level Text Cross-Attention (DenseCLIP-inspired)
# ---------------------------------------------------------------------------

class PixelTextCrossAttention(nn.Module):
    """Pixel-level text guidance via cross-attention.

    Q = image spatial tokens, K/V = text tokens.
    Zero-init out_proj for identity-preserving start.
    """

    def __init__(self, feat_dim: int, text_dim: int, num_heads: int = 4):
        super().__init__()
        assert feat_dim % num_heads == 0, \
            f"feat_dim ({feat_dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(text_dim, feat_dim)
        self.v_proj = nn.Linear(text_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        self.norm = nn.LayerNorm(feat_dim)

        # Zero-init output projection for identity start
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        text_feat: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] spatial feature tokens
            text_feat: [B, M, D_text] text token features
            text_mask: [B, M] optional mask (1=valid, 0=pad)
        Returns:
            [B, N, D] text-attended features
        """
        B, N, D = x.shape
        H, hd = self.num_heads, self.head_dim
        residual = x
        x = self.norm(x)

        q = self.q_proj(x).reshape(B, N, H, hd).transpose(1, 2)
        k = self.k_proj(text_feat).reshape(B, -1, H, hd).transpose(1, 2)
        v = self.v_proj(text_feat).reshape(B, -1, H, hd).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, M]

        if text_mask is not None:
            mask = text_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, M]
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = torch.nan_to_num(attn)  # guard against all-masked rows
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)

        return residual + self.out_proj(out)


class MultiScalePixelTextAttention(nn.Module):
    """Apply pixel-text cross-attention at multiple encoder scales."""

    def __init__(self, stage_dims: list[int], text_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn_layers = nn.ModuleList([
            PixelTextCrossAttention(dim, text_dim, num_heads=num_heads)
            for dim in stage_dims
        ])

    def forward(
        self,
        features: list[torch.Tensor],
        text_feat: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        return [
            attn(feat, text_feat, text_mask)
            for attn, feat in zip(self.attn_layers, features)
        ]


# ---------------------------------------------------------------------------
# Sequential Cross-Attention (TextBraTS-inspired, MICCAI 2025)
# ---------------------------------------------------------------------------

class SequentialCrossAttention(nn.Module):
    """TextBraTS-style Sequential Cross-Attention for text-guided segmentation.

    Two-step cross-attention that reverses the Q/KV direction:
      Step 1 (T2I): Text=Q, Image=KV -> refined features (text-length)
      Step 2 (I2T): Image=Q, Refined=KV -> joint features (image-length)

    This ensures text actively "asks" the image where its descriptions are
    relevant, rather than the image passively querying text tokens.
    """

    def __init__(self, feat_dim: int, text_dim: int, num_heads: int = 4):
        super().__init__()
        assert feat_dim % num_heads == 0, \
            f"feat_dim ({feat_dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Project text to image feature dimension
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, feat_dim),
            nn.LayerNorm(feat_dim),
        )

        # Step 1: Text queries Image (T2I)
        self.t2i_norm_q = nn.LayerNorm(feat_dim)
        self.t2i_norm_kv = nn.LayerNorm(feat_dim)
        self.t2i_q = nn.Linear(feat_dim, feat_dim)
        self.t2i_k = nn.Linear(feat_dim, feat_dim)
        self.t2i_v = nn.Linear(feat_dim, feat_dim)
        self.t2i_out = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
        )

        # Step 2: Image queries Refined (I2T)
        self.i2t_norm_q = nn.LayerNorm(feat_dim)
        self.i2t_norm_kv = nn.LayerNorm(feat_dim)
        self.i2t_q = nn.Linear(feat_dim, feat_dim)
        self.i2t_k = nn.Linear(feat_dim, feat_dim)
        self.i2t_v = nn.Linear(feat_dim, feat_dim)
        self.i2t_out = nn.Linear(feat_dim, feat_dim)

        # Zero-init Step 2 output for identity-preserving start
        nn.init.zeros_(self.i2t_out.weight)
        nn.init.zeros_(self.i2t_out.bias)

    def _multi_head_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multi-head attention computation.

        Args:
            q: [B, Nq, D] queries
            k: [B, Nk, D] keys
            v: [B, Nk, D] values
            mask: [B, Nk] optional mask (1=valid, 0=pad), applied on K dimension
        Returns:
            [B, Nq, D] attention output
        """
        B, Nq, D = q.shape
        H, hd = self.num_heads, self.head_dim

        q = q.reshape(B, Nq, H, hd).transpose(1, 2)   # [B, H, Nq, hd]
        k = k.reshape(B, -1, H, hd).transpose(1, 2)    # [B, H, Nk, hd]
        v = v.reshape(B, -1, H, hd).transpose(1, 2)    # [B, H, Nk, hd]

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, H, Nq, Nk]

        if mask is not None:
            attn = attn.masked_fill(
                mask.unsqueeze(1).unsqueeze(2) == 0,     # [B, 1, 1, Nk]
                float('-inf'),
            )

        attn = attn.softmax(dim=-1)
        attn = torch.nan_to_num(attn)  # guard against all-masked rows

        out = (attn @ v).transpose(1, 2).reshape(B, Nq, D)
        return out

    def forward(
        self,
        x: torch.Tensor,
        text_feat: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] spatial feature tokens (image)
            text_feat: [B, M, D_text] text token features
            text_mask: [B, M] optional mask (1=valid, 0=pad)
        Returns:
            [B, N, D] text-guided features (same shape as input)
        """
        residual = x

        # Project text to image feature space
        text_proj = self.text_proj(text_feat)  # [B, M, D]

        # Step 1: Text=Q, Image=KV
        # Text asks: "where in the image are my descriptions relevant?"
        q1 = self.t2i_q(self.t2i_norm_q(text_proj))    # [B, M, D]
        k1 = self.t2i_k(self.t2i_norm_kv(x))           # [B, N, D]
        v1 = self.t2i_v(self.t2i_norm_kv(x))            # [B, N, D]
        refined = self.t2i_out(self._multi_head_attn(q1, k1, v1))
        # refined: [B, M, D] — text-length, contains image info selected by text

        # Step 2: Image=Q, Refined=KV
        # Image enhances itself using text-selected visual information
        q2 = self.i2t_q(self.i2t_norm_q(x))             # [B, N, D]
        k2 = self.i2t_k(self.i2t_norm_kv(refined))      # [B, M, D]
        v2 = self.i2t_v(self.i2t_norm_kv(refined))       # [B, M, D]
        joint = self.i2t_out(self._multi_head_attn(q2, k2, v2, text_mask))
        # joint: [B, N, D] — image-length

        return residual + joint


class MultiScaleSeqCA(nn.Module):
    """Apply Sequential Cross-Attention at multiple encoder scales."""

    def __init__(self, stage_dims: list[int], text_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn_layers = nn.ModuleList([
            SequentialCrossAttention(dim, text_dim, num_heads=num_heads)
            for dim in stage_dims
        ])

    def forward(
        self,
        features: list[torch.Tensor],
        text_feat: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        return [
            attn(feat, text_feat, text_mask)
            for attn, feat in zip(self.attn_layers, features)
        ]


# ---------------------------------------------------------------------------
# MambaFusion: Deep bottleneck fusion via causal Mamba
# ---------------------------------------------------------------------------

class MambaFusion(nn.Module):
    """Fuse image and text features using Mamba.

    Strategy: Concatenate [text, image] tokens, process with Mamba,
    then extract image portion. Text at the front guides image features
    through Mamba's causal nature.
    """

    def __init__(
        self,
        img_dim: int,
        text_dim: int,
        hidden_dim: int,
        depth: int = 2,
        d_state: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Project to common dimension
        self.img_proj = nn.Sequential(nn.Linear(img_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.text_proj = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim))

        # Mamba fusion layers
        self.mamba_fusion = MambaLayer(
            dim=hidden_dim,
            depth=depth,
            d_state=d_state,
            dropout=dropout,
        )

        # Project back to image dimension
        self.out_proj = nn.Linear(hidden_dim, img_dim)
        self.norm = nn.LayerNorm(img_dim)

    def forward(
        self,
        img_feat: torch.Tensor,
        text_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            img_feat: [B, N, D_img] image features
            text_feat: [B, M, D_text] text features
        Returns:
            [B, N, D_img] fused image features
        """
        B, N, _ = img_feat.shape
        M = text_feat.shape[1]

        # Project to common space
        img_h = self.img_proj(img_feat)    # [B, N, hidden_dim]
        text_h = self.text_proj(text_feat)  # [B, M, hidden_dim]

        # Concatenate: [text, image] - text guides image
        concat = torch.cat([text_h, img_h], dim=1)  # [B, M+N, hidden_dim]

        # Mamba fusion
        fused = self.mamba_fusion(concat)  # [B, M+N, hidden_dim]

        # Extract image portion
        img_fused = fused[:, M:, :]  # [B, N, hidden_dim]

        # Project back and residual
        out = self.out_proj(img_fused)
        out = self.norm(out + img_feat)

        return out


# ---------------------------------------------------------------------------
# Cross-Scale Skip Attention (AttnRes-inspired, V4.6)
# ---------------------------------------------------------------------------

# Custom RMSNorm for compatibility with PyTorch < 2.4 (which lacks nn.RMSNorm)
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (AttnRes best practice for keys)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale


class CrossScaleSkipAttention(nn.Module):
    """AttnRes-inspired cross-scale skip attention for U-Net decoders.

    SUPPLEMENTS (not replaces) the original matched-level skip connection.
    Uses a learned pseudo-query to attend over global-average-pooled representations
    of ALL available encoder features, producing a cross-scale context vector
    that is broadcast to all spatial positions.

    The original skip provides spatially-detailed per-voxel information.
    This module adds coarse multi-resolution context on top.

    AttnRes best practices: zero-init pseudo-query, RMSNorm on keys, zero-init out_proj.
    """

    def __init__(self, target_dim: int, candidate_dims: list):
        """
        Args:
            target_dim: Channel dim of decoder target (after upsample)
            candidate_dims: Channel dims of each candidate encoder feature
        """
        super().__init__()
        if len(candidate_dims) == 0:
            raise ValueError("candidate_dims must be non-empty")
        self.target_dim = target_dim
        self.num_candidates = len(candidate_dims)

        self.key_projs = nn.ModuleList([
            nn.Linear(cd, target_dim) for cd in candidate_dims
        ])
        self.key_norms = nn.ModuleList([
            RMSNorm(target_dim) for _ in candidate_dims
        ])
        self.val_projs = nn.ModuleList([
            nn.Linear(cd, target_dim) for cd in candidate_dims
        ])

        # Pseudo-query: one learnable vector per module instance (zero-init)
        self.pseudo_query = nn.Parameter(torch.zeros(1, 1, target_dim))

        # Output projection (zero-init for identity start)
        self.out_proj = nn.Linear(target_dim, target_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.register_buffer('scale', torch.tensor(target_dim ** -0.5))

    def forward(
        self,
        target: torch.Tensor,
        candidates: list,
    ) -> torch.Tensor:
        """
        Args:
            target: [B, L_target, D_target] decoder features after upsample
            candidates: list of [B, L_i, D_i] encoder features (variable lengths)
        Returns:
            [B, L_target, D_target] cross-scale skip contribution (additive)
        """
        assert len(candidates) == self.num_candidates, \
            f"Expected {self.num_candidates} candidates, got {len(candidates)}"
        B, L, _ = target.shape

        keys = []
        values = []
        for i, cand in enumerate(candidates):
            pooled = cand.mean(dim=1, keepdim=True)  # [B, 1, D_i]
            k = self.key_norms[i](self.key_projs[i](pooled))  # [B, 1, D_target]
            v = self.val_projs[i](pooled)                       # [B, 1, D_target]
            keys.append(k)
            values.append(v)

        keys = torch.cat(keys, dim=1)      # [B, S, D_target]
        values = torch.cat(values, dim=1)   # [B, S, D_target]

        q = self.pseudo_query.expand(B, -1, -1)  # [B, 1, D_target]
        attn = (q * self.scale) @ keys.transpose(-2, -1)  # [B, 1, S]
        attn = attn.softmax(dim=-1)

        agg = (attn @ values)  # [B, 1, D_target]
        out = self.out_proj(agg).expand(-1, L, -1).contiguous()  # [B, L, D_target]

        return out
