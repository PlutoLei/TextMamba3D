# TextMamba3D V4.6: AttnRes-Inspired Cross-Scale Skip Attention + Text Scale Gate

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve TextMamba3D mean Dice from 83.48% to ≥84% by augmenting decoder skip connections with learned cross-scale attention (Direction A) and adding adaptive text fusion gating per scale (Direction B), inspired by Moonshot AI's Attention Residuals paper.

**Architecture:** Direction A adds a `CrossScaleSkipAttention` module that supplements (NOT replaces) the existing `x = x + skip_proj(skip)` with a cross-scale attention term. The original matched-level skip preserves spatial detail; the new module adds a learned cross-scale contribution on top. Direction B adds a `TextScaleGate` between text fusion and decoder, allowing adaptive mixing of raw vs text-fused features per scale — addressing the v4.5 TC regression (-1.16%).

**Tech Stack:** PyTorch, Mamba SSM, PubMedBERT, einops, YAML config, Colab A100 40GB

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `models/fusion.py` | Modify | Add `RMSNorm`, `CrossScaleSkipAttention`, `TextScaleGate`, `MultiScaleTextGate` |
| `models/decoder_3d.py` | Modify | Add `CrossScaleSkipAttention` alongside existing skip connections |
| `models/textmamba3d.py` | Modify | Add `TextScaleGate` + forward new params to decoder |
| `models/__init__.py` | Modify | Export 4 new classes |
| `train.py` | Modify | Forward 3 new config fields to model constructor |
| `evaluate_full.py` | Modify | Forward new config fields |
| `inference.py` | Modify | Forward new config fields |
| `configs/textbrats_v8.yaml` | Create | V4.6 config |
| `tests/test_v46_modules.py` | Create | Unit + integration tests |
| `TextMamba3D_A100_V4.6.ipynb` | Create (later, via crossfire) | Colab training notebook |

### Architecture Diagram

```
Encoder:  f0[B,4096,48]  f1[B,512,96]  f2[B,64,192]  f3[B,8,384]
              |              |              |              |
              |         SeqCA(text)    SeqCA(text)    SeqCA(text)
              |              |              |              |
              |         TextScaleGate  TextScaleGate  TextScaleGate  ← Dir B
              |           (g1)           (g2)           (g3)
              |              |              |              |
              v              v              v              v
            d_f0           d_f1           d_f2           d_f3
                                                          |
                                                    decoder_stage_0
                                                          |  upsample
                                                    skip_proj(d_f2)     ← original spatial skip (KEPT)
                                                  + CrossScaleSkipAttn  ← Dir A (supplemental)
                                                      (attend: d_f0, d_f1, d_f2)
                                                          |
                                                    decoder_stage_1
                                                          |  upsample
                                                    skip_proj(d_f1)     ← original spatial skip (KEPT)
                                                  + CrossScaleSkipAttn
                                                      (attend: d_f0, d_f1)
                                                          |
                                                    decoder_stage_2
                                                          |  upsample
                                                    skip_proj(d_f0)     ← original spatial skip (KEPT)
                                                  + CrossScaleSkipAttn
                                                      (attend: d_f0)
                                                          |
                                                    decoder_stage_3
                                                          |
                                                    final_proj → segmentation
```

### Key Design Decisions

**D1: Cross-Scale Skip Attention SUPPLEMENTS the original matched-level skip, not replaces it.**
The original `x = x + skip_proj(features[matched_level])` is critical for spatial detail recovery. CrossScaleSkipAttention adds a learned cross-scale term on top: `x = x + skip_proj(features[matched]) + cross_scale_attn(x, all_candidates)`. This preserves all v4.5 spatial expressiveness while adding the ability to pull information from other scales.

**D2: Cross-scale attention uses global-average-pooled scale representations.**
Encoder features at different scales (4096/512/64/8 tokens) are pooled to single vectors for scale-level attention. The attention answers "how much from each scale?" (S=1..3 candidates), producing a spatially-uniform cross-scale supplement. Fine-grained spatial detail comes from the original skip; cross-scale attention adds coarse multi-resolution context. Cost: O(S) not O(N).

**D3: TextScaleGate uses sigmoid gating with zero-init weight + bias=+2 (starts near text ON).**
Gate formula: `output = gate * fused + (1 - gate) * raw`, where `gate = sigmoid(linear(concat(raw, fused)))`. Zero-init weight + bias=+2 → sigmoid(2)≈0.88 → text fusion ON by default. The gate learns to suppress text at scales where it hurts (hypothesized cause of TC regression).

**D4: Backward compatibility preserved.**
All new features default to off. When `use_text=False`, TextScaleGate is bypassed. CrossScaleSkipAttention uses zero-init pseudo-queries + zero-init out_proj → zero contribution at init → identical to v4.5 at start of training.

**D5: Memory budget.**
- CrossScaleSkipAttention: 3 pseudo-query vectors + 3 key projectors + 3 val projectors + 3 RMSNorm + 3 out_proj ≈ ~50K params. Negligible.
- TextScaleGate: 3 gates, each `Linear(2*dim, 1)` ≈ ~1.5K params. Negligible.
- VRAM: attention is over 1-3 scale slots → FLOPs negligible. batch_size=4 on A100 40GB safe.

---

## Chunk 1: New Modules in fusion.py

### Task 1: CrossScaleSkipAttention Module

**Files:**
- Modify: `models/fusion.py` (append after line 392)
- Create: `tests/test_v46_modules.py`

- [ ] **Step 1: Write the failing test for CrossScaleSkipAttention**

```python
# tests/test_v46_modules.py
import torch
import pytest


def test_cross_scale_skip_attention_output_shape():
    """CrossScaleSkipAttention should output [B, L_target, D_target]."""
    from models.fusion import CrossScaleSkipAttention

    B = 2
    # Simulate decoder stage 1: target is f2-resolution after upsample
    # Candidates: f0=[B,4096,48], f1=[B,512,96], f2=[B,64,192]
    target = torch.randn(B, 64, 192)
    candidates = [
        torch.randn(B, 4096, 48),
        torch.randn(B, 512, 96),
        torch.randn(B, 64, 192),
    ]
    candidate_dims = [48, 96, 192]

    module = CrossScaleSkipAttention(
        target_dim=192,
        candidate_dims=candidate_dims,
    )

    out = module(target, candidates)
    assert out.shape == (B, 64, 192), f"Expected (2,64,192), got {out.shape}"


def test_cross_scale_skip_attention_zero_init():
    """At initialization, output should be near-zero (identity-preserving)."""
    from models.fusion import CrossScaleSkipAttention

    B = 2
    target = torch.randn(B, 64, 192)
    candidates = [torch.randn(B, 512, 96), torch.randn(B, 64, 192)]

    module = CrossScaleSkipAttention(
        target_dim=192,
        candidate_dims=[96, 192],
    )

    out = module(target, candidates)
    # Zero-init pseudo-query → softmax(zeros) = uniform → but out_proj is zero-init
    # So output should be near zero
    assert out.abs().max() < 1e-5, f"Expected near-zero output at init, got max={out.abs().max()}"


def test_cross_scale_skip_attention_single_candidate():
    """Should work with a single candidate (last decoder stage)."""
    from models.fusion import CrossScaleSkipAttention

    B = 2
    target = torch.randn(B, 4096, 48)
    candidates = [torch.randn(B, 4096, 48)]

    module = CrossScaleSkipAttention(target_dim=48, candidate_dims=[48])
    out = module(target, candidates)
    assert out.shape == (B, 4096, 48)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd TextMamba3D && python -m pytest tests/test_v46_modules.py -v -k "cross_scale"`
Expected: FAIL with `ImportError: cannot import name 'CrossScaleSkipAttention'`

- [ ] **Step 3: Implement CrossScaleSkipAttention**

Append to `models/fusion.py` after line 392:

```python
# ---------------------------------------------------------------------------
# Cross-Scale Skip Attention (AttnRes-inspired, V4.6)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (AttnRes best practice for keys)."""

    def __init__(self, dim: int, eps: float = 1e-8):
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

    def __init__(self, target_dim: int, candidate_dims: list[int]):
        """
        Args:
            target_dim: Channel dim of decoder target (after upsample)
            candidate_dims: Channel dims of each candidate encoder feature
        """
        super().__init__()
        self.target_dim = target_dim
        self.num_candidates = len(candidate_dims)

        # Per-candidate: project to target_dim, then RMSNorm
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

        self.scale = target_dim ** -0.5

    def forward(
        self,
        target: torch.Tensor,
        candidates: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            target: [B, L_target, D_target] decoder features after upsample
            candidates: list of [B, L_i, D_i] encoder features (variable lengths)
        Returns:
            [B, L_target, D_target] cross-scale skip contribution (additive)
        """
        B, L, _ = target.shape

        # Pool each candidate to a single representative vector
        keys = []
        values = []
        for i, cand in enumerate(candidates):
            pooled = cand.mean(dim=1, keepdim=True)  # [B, 1, D_i]
            k = self.key_norms[i](self.key_projs[i](pooled))  # [B, 1, D_target]
            v = self.val_projs[i](pooled)                       # [B, 1, D_target]
            keys.append(k)
            values.append(v)

        keys = torch.cat(keys, dim=1)     # [B, S, D_target]
        values = torch.cat(values, dim=1)  # [B, S, D_target]

        # Pseudo-query attention over scale candidates
        q = self.pseudo_query.expand(B, -1, -1)  # [B, 1, D_target]
        attn = (q * self.scale) @ keys.transpose(-2, -1)  # [B, 1, S]
        attn = attn.softmax(dim=-1)

        # Weighted combination → broadcast to all spatial positions → project
        agg = (attn @ values)  # [B, 1, D_target]
        out = self.out_proj(agg.expand(-1, L, -1))  # [B, L, D_target]

        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd TextMamba3D && python -m pytest tests/test_v46_modules.py -v -k "cross_scale"`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add models/fusion.py tests/test_v46_modules.py
git commit -m "feat(v4.6): add CrossScaleSkipAttention + RMSNorm modules"
```

---

### Task 2: TextScaleGate Module

**Files:**
- Modify: `models/fusion.py` (append after CrossScaleSkipAttention)
- Modify: `tests/test_v46_modules.py`

- [ ] **Step 1: Write the failing test for TextScaleGate**

Append to `tests/test_v46_modules.py`:

```python
def test_text_scale_gate_output_shape():
    """TextScaleGate should output same shape as input."""
    from models.fusion import TextScaleGate

    B, L, D = 2, 512, 96
    raw = torch.randn(B, L, D)
    fused = torch.randn(B, L, D)

    gate = TextScaleGate(feat_dim=D)
    out = gate(raw, fused)
    assert out.shape == (B, L, D), f"Expected ({B},{L},{D}), got {out.shape}"


def test_text_scale_gate_init_near_fused():
    """At init, gate ≈ 0.88, so output ≈ 0.88*fused + 0.12*raw."""
    from models.fusion import TextScaleGate

    B, L, D = 2, 64, 192
    raw = torch.zeros(B, L, D)
    fused = torch.ones(B, L, D)

    gate = TextScaleGate(feat_dim=D)
    out = gate(raw, fused)
    # sigmoid(2) ≈ 0.8808
    assert 0.85 < out.mean().item() < 0.92, f"Expected ~0.88, got {out.mean():.4f}"


def test_text_scale_gate_bypass_when_same():
    """When raw == fused, output should equal both (gate*x + (1-gate)*x = x)."""
    from models.fusion import TextScaleGate

    B, L, D = 2, 64, 192
    x = torch.randn(B, L, D)

    gate = TextScaleGate(feat_dim=D)
    out = gate(x, x)
    assert torch.allclose(out, x, atol=1e-5), "When raw==fused, output should equal input"


def test_multi_scale_text_gate():
    """MultiScaleTextGate should apply gates at each scale independently."""
    from models.fusion import MultiScaleTextGate

    B = 2
    stage_dims = [96, 192, 384]
    raw_features = [torch.randn(B, 512, 96), torch.randn(B, 64, 192), torch.randn(B, 8, 384)]
    fused_features = [torch.randn(B, 512, 96), torch.randn(B, 64, 192), torch.randn(B, 8, 384)]

    gate = MultiScaleTextGate(stage_dims=stage_dims)
    outputs = gate(raw_features, fused_features)

    assert len(outputs) == 3
    for out, raw in zip(outputs, raw_features):
        assert out.shape == raw.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd TextMamba3D && python -m pytest tests/test_v46_modules.py -v -k "text_scale_gate or multi_scale_text"`
Expected: FAIL with `ImportError: cannot import name 'TextScaleGate'`

- [ ] **Step 3: Implement TextScaleGate and MultiScaleTextGate**

Append to `models/fusion.py`:

```python
# ---------------------------------------------------------------------------
# Text Scale Gate (AttnRes-inspired adaptive text fusion, V4.6)
# ---------------------------------------------------------------------------

class TextScaleGate(nn.Module):
    """Learned gate controlling text fusion contribution at each scale.

    Adaptively mixes raw encoder features with text-fused features:
        output = gate * fused + (1 - gate) * raw

    Init: zero weights + bias=2.0 → sigmoid(2)≈0.88 → text fusion ON by default.
    """

    def __init__(self, feat_dim: int, init_bias: float = 2.0):
        super().__init__()
        self.gate_proj = nn.Linear(2 * feat_dim, 1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_bias)

    def forward(self, raw: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw: [B, L, D] raw encoder features (before text fusion)
            fused: [B, L, D] text-fused features (after SeqCA)
        Returns:
            [B, L, D] gated combination
        """
        gate_input = torch.cat([raw, fused], dim=-1)  # [B, L, 2D]
        gate = torch.sigmoid(self.gate_proj(gate_input))  # [B, L, 1]
        return gate * fused + (1 - gate) * raw


class MultiScaleTextGate(nn.Module):
    """Apply TextScaleGate at multiple encoder scales."""

    def __init__(self, stage_dims: list[int], init_bias: float = 2.0):
        super().__init__()
        self.gates = nn.ModuleList([
            TextScaleGate(feat_dim=dim, init_bias=init_bias)
            for dim in stage_dims
        ])

    def forward(
        self,
        raw_features: list[torch.Tensor],
        fused_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return [
            gate(raw, fused)
            for gate, raw, fused in zip(self.gates, raw_features, fused_features)
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd TextMamba3D && python -m pytest tests/test_v46_modules.py -v -k "text_scale_gate or multi_scale_text"`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add models/fusion.py tests/test_v46_modules.py
git commit -m "feat(v4.6): add TextScaleGate and MultiScaleTextGate modules"
```

---

## Chunk 2: Integration into Decoder and Model

### Task 3: Integrate CrossScaleSkipAttention into MambaDecoder3D

**Files:**
- Modify: `models/decoder_3d.py`
- Modify: `tests/test_v46_modules.py`

The decoder currently has 3 fixed skip connections (lines 139-146). We ADD `CrossScaleSkipAttention` alongside them, keeping the original skip_proj intact.

**Key change:** New parameter `use_cross_scale_skip: bool = False`. When True, each skip connection becomes: `x = x + skip_proj(matched_level) + cross_scale_attn(x, all_candidates)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_v46_modules.py`:

```python
def test_decoder_with_cross_scale_skip():
    """Decoder with cross-scale skip should produce same output shape."""
    from models.decoder_3d import MambaDecoder3D

    B = 1
    decoder = MambaDecoder3D(
        img_size=(128, 128, 128),
        patch_size=(4, 4, 4),
        out_channels=4,
        embed_dim=48,
        depths=[2, 2, 2, 2],
        use_cross_scale_skip=True,
    )

    features = [
        torch.randn(B, 4096, 48),
        torch.randn(B, 512, 96),
        torch.randn(B, 64, 192),
        torch.randn(B, 8, 384),
    ]

    out = decoder(features)
    assert out.shape == (B, 4, 128, 128, 128), f"Expected (1,4,128,128,128), got {out.shape}"


def test_decoder_backward_compat():
    """Decoder without cross-scale skip should work identically to v4.5."""
    from models.decoder_3d import MambaDecoder3D

    B = 1
    decoder = MambaDecoder3D(
        img_size=(128, 128, 128),
        patch_size=(4, 4, 4),
        out_channels=4,
        embed_dim=48,
        depths=[2, 2, 2, 2],
        use_cross_scale_skip=False,
    )

    features = [
        torch.randn(B, 4096, 48),
        torch.randn(B, 512, 96),
        torch.randn(B, 64, 192),
        torch.randn(B, 8, 384),
    ]

    out = decoder(features)
    assert out.shape == (B, 4, 128, 128, 128)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd TextMamba3D && python -m pytest tests/test_v46_modules.py -v -k "decoder"`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'use_cross_scale_skip'`

- [ ] **Step 3: Modify MambaDecoder3D**

Complete replacement of `models/decoder_3d.py`:

```python
# models/decoder_3d.py
import torch
import torch.nn as nn
from einops import rearrange
from .mamba_block import CrossScanBiMamba3DLayer
from .fusion import CrossScaleSkipAttention


class PatchExpanding3D(nn.Module):
    """Patch expanding for upsampling."""

    def __init__(self, dim: int, out_dim: int, spatial_dims: tuple):
        super().__init__()
        self.dim = dim
        self.spatial_dims = spatial_dims
        self.expand = nn.Linear(dim, 8 * out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        D, H, W = self.spatial_dims
        x = self.expand(x)
        x = rearrange(x, 'b (d h w) (p1 p2 p3 c) -> b (d p1 h p2 w p3) c',
                     d=D, h=H, w=W, p1=2, p2=2, p3=2)
        x = self.norm(x)
        return x


class MambaDecoder3D(nn.Module):
    """3D Mamba Decoder with skip connections.

    V4.6: Optional CrossScaleSkipAttention supplements (not replaces) the
    existing matched-level skip projection with cross-scale context.
    """

    def __init__(
        self,
        img_size: tuple = (96, 96, 96),
        patch_size: tuple = (4, 4, 4),
        out_channels: int = 4,
        embed_dim: int = 96,
        depths: list = [2, 2, 2, 2],
        d_state: int = 16,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
        deep_supervision: bool = False,
        use_cross_scale_skip: bool = False,
    ):
        super().__init__()
        self.num_stages = len(depths)
        self.deep_supervision = deep_supervision
        self.use_cross_scale_skip = use_cross_scale_skip

        d, h, w = img_size[0] // patch_size[0], \
                  img_size[1] // patch_size[1], \
                  img_size[2] // patch_size[2]

        self.stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_projs = nn.ModuleList()

        # V4.6: cross-scale attention modules (one per skip connection)
        self.cross_scale_attns = nn.ModuleList() if use_cross_scale_skip else None

        # Track skip connection index for cross-scale attention construction
        skip_count = 0

        for i in range(len(depths) - 1, -1, -1):
            dim = embed_dim * (2 ** i)
            spatial = (d // (2 ** i), h // (2 ** i), w // (2 ** i))

            stage = CrossScanBiMamba3DLayer(
                dim=dim,
                depth=depths[i],
                spatial_dims=spatial,
                d_state=d_state,
                dropout=dropout,
                use_checkpoint=use_checkpoint,
            )
            self.stages.append(stage)

            if i > 0:
                spatial = (d // (2 ** i), h // (2 ** i), w // (2 ** i))
                upsample = PatchExpanding3D(
                    dim=dim,
                    out_dim=dim // 2,
                    spatial_dims=spatial,
                )
                self.upsamples.append(upsample)

                # Original skip projection (ALWAYS present)
                skip_proj = nn.Linear(dim // 2, dim // 2)
                self.skip_projs.append(skip_proj)

                # V4.6: cross-scale attention (supplemental)
                if use_cross_scale_skip:
                    target_dim = dim // 2
                    skip_idx = len(depths) - 2 - skip_count
                    candidate_dims = [embed_dim * (2 ** j) for j in range(skip_idx + 1)]
                    self.cross_scale_attns.append(
                        CrossScaleSkipAttention(
                            target_dim=target_dim,
                            candidate_dims=candidate_dims,
                        )
                    )
                    skip_count += 1

        # Deep supervision
        if deep_supervision:
            self.aux_heads = nn.ModuleList()
            self.aux_spatials = []
            for idx, i in enumerate(range(len(depths) - 1, 0, -1)):
                dim = embed_dim * (2 ** i)
                spatial = (d // (2 ** i), h // (2 ** i), w // (2 ** i))
                self.aux_heads.append(nn.Conv3d(dim, out_channels, 1))
                self.aux_spatials.append(spatial)

        self.final_expand = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * (patch_size[0] ** 3)),
            nn.GELU(),
        )
        self.final_proj = nn.Conv3d(embed_dim, out_channels, 1)

        self.patch_size = patch_size
        self.base_spatial = (d, h, w)

    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: List of encoder features [stage0, stage1, ..., bottleneck]
        Returns:
            [B, out_channels, D, H, W]
        """
        self._aux_outputs = []
        x = features[-1]

        for i, stage in enumerate(self.stages):
            x = stage(x)

            if self.deep_supervision and self.training and i < len(self.aux_heads):
                B_aux, L_aux, C_aux = x.shape
                ds, hs, ws = self.aux_spatials[i]
                aux = rearrange(x, 'b (d h w) c -> b c d h w', d=ds, h=hs, w=ws)
                self._aux_outputs.append(self.aux_heads[i](aux))

            if i < len(self.upsamples):
                x = self.upsamples[i](x)

                # Skip connection
                skip_idx = len(features) - 2 - i
                if skip_idx >= 0:
                    # Original matched-level skip (always)
                    skip = self.skip_projs[i](features[skip_idx])
                    x = x + skip

                    # V4.6: cross-scale supplemental attention
                    if self.use_cross_scale_skip and self.cross_scale_attns is not None:
                        candidates = features[:skip_idx + 1]
                        x = x + self.cross_scale_attns[i](x, candidates)

        # Final expansion
        B, L, C = x.shape
        d, h, w = self.base_spatial
        p = self.patch_size[0]

        x = self.final_expand(x)
        x = rearrange(x, 'b (d h w) (p1 p2 p3 c) -> b c (d p1) (h p2) (w p3)',
                     d=d, h=h, w=w, p1=p, p2=p, p3=p, c=C)
        x = self.final_proj(x)

        return x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd TextMamba3D && python -m pytest tests/test_v46_modules.py -v -k "decoder"`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add models/decoder_3d.py tests/test_v46_modules.py
git commit -m "feat(v4.6): integrate CrossScaleSkipAttention into MambaDecoder3D (supplemental)"
```

---

### Task 4: Integrate TextScaleGate into TextMamba3D Forward Pass

**Files:**
- Modify: `models/textmamba3d.py`
- Modify: `tests/test_v46_modules.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_v46_modules.py`:

```python
def test_textmamba3d_with_text_gate():
    """TextMamba3D with use_text_gate=True should produce same output shape."""
    from models.textmamba3d import TextMamba3D

    model = TextMamba3D(
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=4,
        embed_dim=48,
        depths=[2, 2, 2, 2],
        text_embed_dim=256,
        text_max_len=192,
        use_pretrained_text=False,
        use_text_gate=True,
        use_cross_scale_skip=True,
    )

    img = torch.randn(1, 4, 128, 128, 128)
    out = model(img, use_text=False)
    assert out.shape == (1, 4, 128, 128, 128)


def test_textmamba3d_v46_backward_compat():
    """TextMamba3D without new features should have no text_gate."""
    from models.textmamba3d import TextMamba3D

    model = TextMamba3D(
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=4,
        embed_dim=48,
        depths=[2, 2, 2, 2],
        text_embed_dim=256,
        text_max_len=192,
        use_pretrained_text=False,
        use_text_gate=False,
        use_cross_scale_skip=False,
    )

    assert model.text_gate is None
    assert not model.decoder.use_cross_scale_skip
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd TextMamba3D && python -m pytest tests/test_v46_modules.py -v -k "textmamba3d"`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'use_text_gate'`

- [ ] **Step 3: Modify TextMamba3D**

Complete replacement of `models/textmamba3d.py`:

```python
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
        """
        Forward pass for text-guided 3D segmentation.

        Args:
            img: [B, C, D, H, W]
            text_ids: [B, L] optional
            attention_mask: [B, L] (1=valid, 0=pad)
            return_features: return features for contrastive loss
            use_text: whether to use text guidance
        """
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd TextMamba3D && python -m pytest tests/test_v46_modules.py -v -k "textmamba3d"`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add models/textmamba3d.py tests/test_v46_modules.py
git commit -m "feat(v4.6): integrate TextScaleGate into TextMamba3D forward pass"
```

---

### Task 5: Update models/__init__.py Exports

**Files:**
- Modify: `models/__init__.py`

- [ ] **Step 1: Add new exports**

Change line 8 of `models/__init__.py` from:
```python
from .fusion import PixelTextCrossAttention, MultiScalePixelTextAttention, FiLMLayer, MultiScaleFiLM, MambaFusion
```
to:
```python
from .fusion import (
    PixelTextCrossAttention, MultiScalePixelTextAttention,
    FiLMLayer, MultiScaleFiLM, MambaFusion,
    RMSNorm, CrossScaleSkipAttention, TextScaleGate, MultiScaleTextGate,
)
```

- [ ] **Step 2: Run import smoke test**

Run: `cd TextMamba3D && python -c "from models import CrossScaleSkipAttention, TextScaleGate, MultiScaleTextGate, RMSNorm; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add models/__init__.py
git commit -m "chore: export V4.6 modules from models package"
```

---

## Chunk 3: Config and Script Updates

### Task 6: Forward New Config Fields in train.py, evaluate_full.py, inference.py

**Files:**
- Modify: `train.py:353-367` — model construction
- Modify: `evaluate_full.py:238-248` — model construction
- Modify: `inference.py:153-162` — model construction

All three scripts manually enumerate kwargs. We must add the 3 new config fields.

- [ ] **Step 1: Modify train.py**

At `train.py:353`, after the existing `TextMamba3D(` kwargs, add before `.to(device)`:

```python
        # V4.6 features (default off for backward compat)
        use_text_gate=config['model'].get('use_text_gate', False),
        use_cross_scale_skip=config['model'].get('use_cross_scale_skip', False),
        text_gate_init_bias=config['model'].get('text_gate_init_bias', 2.0),
```

- [ ] **Step 2: Modify evaluate_full.py**

At `evaluate_full.py:238`, add after `use_checkpoint=...`:

```python
        use_text_gate=model_cfg.get('use_text_gate', False),
        use_cross_scale_skip=model_cfg.get('use_cross_scale_skip', False),
        text_gate_init_bias=model_cfg.get('text_gate_init_bias', 2.0),
```

- [ ] **Step 3: Modify inference.py**

At `inference.py:153`, add after `use_pretrained_text=...`:

```python
            use_text_gate=self.config['model'].get('use_text_gate', False),
            use_cross_scale_skip=self.config['model'].get('use_cross_scale_skip', False),
            text_gate_init_bias=self.config['model'].get('text_gate_init_bias', 2.0),
```

- [ ] **Step 4: Verify old configs still work**

Run: `cd TextMamba3D && python -c "
import yaml
from models.textmamba3d import TextMamba3D
cfg = yaml.safe_load(open('configs/textbrats_v7.yaml'))
m = cfg['model']
# Simulate train.py construction with v7 config (no v4.6 fields)
print('use_text_gate:', m.get('use_text_gate', False))
print('use_cross_scale_skip:', m.get('use_cross_scale_skip', False))
print('Both default to False — v4.5 behavior preserved')
"`
Expected: Both False

- [ ] **Step 5: Commit**

```bash
git add train.py evaluate_full.py inference.py
git commit -m "feat(v4.6): forward new config fields in train/eval/inference scripts"
```

---

### Task 7: Create textbrats_v8.yaml Config

**Files:**
- Create: `configs/textbrats_v8.yaml`

- [ ] **Step 1: Write the config**

```yaml
# configs/textbrats_v8.yaml
# V4.6: AttnRes-inspired Cross-Scale Skip Attention + Text Scale Gate
# Based on textbrats_v7.yaml (V4.5), adds Direction A + B

data:
  data_dir: "./data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
  dataset_type: "textbrats"
  patch_size: [128, 128, 128]
  batch_size: 4
  num_workers: 4                  # Colab Linux; use 0 on Windows
  train_ratio: 0.596
  val_ratio: 0.149
  et_enriched: true
  enriched_prob: 0.5

model:
  img_size: [128, 128, 128]
  in_channels: 4
  out_channels: 4
  embed_dim: 48
  depths: [2, 2, 2, 2]
  dropout: 0.1
  text_embed_dim: 256
  text_max_len: 192
  use_pretrained_text: true
  unfreeze_text_layers: 2
  text_model_path: null
  # V4.6 new features
  use_cross_scale_skip: true      # Direction A
  use_text_gate: true             # Direction B
  text_gate_init_bias: 2.0        # sigmoid(2)≈0.88, text ON by default

loss:
  dice_weight: 1.0
  ce_weight: 1.0
  edge_weight: 1.0
  contrastive_weight: 0.05
  temperature: 0.07
  class_weights: [0.25, 3.0, 1.0, 4.0]

augmentation:
  use_elastic: true
  use_modality_dropout: true

training:
  epochs: 200
  lr: 0.0001
  weight_decay: 0.01
  warmup_epochs: 10
  contrastive_warmup_epochs: 30
  patience: 40
  gradient_accumulation: 1
  gradient_checkpointing: true
  deep_supervision: true
  ds_weights: [0.2, 0.1, 0.05]
  use_amp: true
  no_text_ratio: 0.15
  gradient_clip_norm: 1.0

eval:
  metrics: ["dice", "hd95"]
  sliding_window: true
  sw_overlap: 0.5
  sw_batch_size: 2

experiment:
  name: "TextMamba3D_A100_v4.6_attnres"
  description: "V4.6: SeqCA + CrossScaleSkipAttention (Dir A) + TextScaleGate (Dir B) + ET-Enriched"
```

- [ ] **Step 2: Validate YAML**

Run: `cd TextMamba3D && python -c "import yaml; cfg = yaml.safe_load(open('configs/textbrats_v8.yaml')); print('OK:', cfg['experiment']['name'])"`
Expected: `OK: TextMamba3D_A100_v4.6_attnres`

- [ ] **Step 3: Smoke test model construction from config**

Run:
```python
cd TextMamba3D && python -c "
import yaml
from models.textmamba3d import TextMamba3D
cfg = yaml.safe_load(open('configs/textbrats_v8.yaml'))
m = cfg['model']
model = TextMamba3D(
    img_size=tuple(m['img_size']), in_channels=m['in_channels'],
    out_channels=m['out_channels'], embed_dim=m['embed_dim'],
    depths=m['depths'], text_embed_dim=m['text_embed_dim'],
    text_max_len=m['text_max_len'], dropout=m.get('dropout', 0.0),
    use_pretrained_text=False,  # skip download
    use_text_gate=m.get('use_text_gate', False),
    use_cross_scale_skip=m.get('use_cross_scale_skip', False),
    text_gate_init_bias=m.get('text_gate_init_bias', 2.0),
)
print(f'text_gate: {model.text_gate is not None}')
print(f'cross_scale_skip: {model.decoder.use_cross_scale_skip}')
print(f'params: {sum(p.numel() for p in model.parameters()):,}')
"
```
Expected: text_gate: True, cross_scale_skip: True

- [ ] **Step 4: Commit**

```bash
git add configs/textbrats_v8.yaml
git commit -m "feat(v4.6): add textbrats_v8.yaml config"
```

---

## Chunk 4: Integration Tests

### Task 8: End-to-End Integration Tests

**Files:**
- Modify: `tests/test_v46_modules.py`

- [ ] **Step 1: Write integration tests**

Append to `tests/test_v46_modules.py`:

```python
@pytest.mark.slow
def test_v46_full_forward_pass():
    """Full V4.6 model forward pass with text input."""
    from models.textmamba3d import TextMamba3D

    model = TextMamba3D(
        img_size=(32, 32, 32),
        in_channels=4,
        out_channels=4,
        embed_dim=48,
        depths=[1, 1, 1, 1],
        text_embed_dim=64,
        text_max_len=32,
        text_depth=1,
        use_text_gate=True,
        use_cross_scale_skip=True,
        use_pretrained_text=False,
    )
    model.eval()

    img = torch.randn(1, 4, 32, 32, 32)
    text_ids = torch.randint(0, 1000, (1, 32))
    mask = torch.ones(1, 32)

    with torch.no_grad():
        out_text = model(img, text_ids, mask, use_text=True)
        assert out_text.shape == (1, 4, 32, 32, 32)

        out_no_text = model(img, use_text=False)
        assert out_no_text.shape == (1, 4, 32, 32, 32)

    assert not torch.allclose(out_text, out_no_text, atol=1e-3), \
        "Text and no-text outputs should differ"


@pytest.mark.slow
def test_v46_gradient_flow():
    """Verify gradients flow through both new modules."""
    from models.textmamba3d import TextMamba3D

    model = TextMamba3D(
        img_size=(32, 32, 32),
        in_channels=4,
        out_channels=4,
        embed_dim=48,
        depths=[1, 1, 1, 1],
        text_embed_dim=64,
        text_max_len=16,
        text_depth=1,
        use_text_gate=True,
        use_cross_scale_skip=True,
        use_pretrained_text=False,
    )

    img = torch.randn(1, 4, 32, 32, 32)
    text_ids = torch.randint(0, 1000, (1, 16))
    mask = torch.ones(1, 16)

    out = model(img, text_ids, mask, use_text=True)
    loss = out.sum()
    loss.backward()

    # TextScaleGate gradients
    for i, gate in enumerate(model.text_gate.gates):
        assert gate.gate_proj.weight.grad is not None, f"No gradient for text gate {i}"
        assert gate.gate_proj.weight.grad.abs().sum() > 0, f"Zero gradient for text gate {i}"

    # CrossScaleSkipAttention gradients
    for i, attn in enumerate(model.decoder.cross_scale_attns):
        assert attn.pseudo_query.grad is not None, f"No gradient for pseudo_query {i}"


@pytest.mark.slow
def test_v46_with_contrastive_features():
    """V4.6 model should return features for contrastive loss."""
    from models.textmamba3d import TextMamba3D

    model = TextMamba3D(
        img_size=(32, 32, 32),
        in_channels=4,
        out_channels=4,
        embed_dim=48,
        depths=[1, 1, 1, 1],
        text_embed_dim=64,
        text_max_len=16,
        text_depth=1,
        use_text_gate=True,
        use_cross_scale_skip=True,
        use_pretrained_text=False,
    )

    img = torch.randn(1, 4, 32, 32, 32)
    text_ids = torch.randint(0, 1000, (1, 16))
    mask = torch.ones(1, 16)

    seg, img_g, text_g, pix = model(img, text_ids, mask, return_features=True)
    assert seg.shape == (1, 4, 32, 32, 32)
    assert img_g is not None
    assert text_g is not None
    assert pix is not None
```

- [ ] **Step 2: Run integration tests**

Run: `cd TextMamba3D && python -m pytest tests/test_v46_modules.py -v -k "v46_full or v46_gradient or v46_with_contrastive" -m slow`
Expected: 3 tests PASS

- [ ] **Step 3: Run ALL v4.6 tests**

Run: `cd TextMamba3D && python -m pytest tests/test_v46_modules.py -v`
Expected: All 12 tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_v46_modules.py
git commit -m "test(v4.6): add end-to-end integration and gradient flow tests"
```

---

## Chunk 5: Notebook (Deferred to Crossfire)

### Task 9: Create V4.6 Training Notebook

**Files:**
- Create: `TextMamba3D_A100_V4.6.ipynb`

This task is best executed via **crossfire** pipeline. The notebook mirrors V4.5 with these additions:

1. **Patch cells** — Inline patches for all V4.6 modules (RMSNorm, CrossScaleSkipAttention, TextScaleGate, MultiScaleTextGate) + modified decoder + modified TextMamba3D

2. **Verification cell:**
```python
model = TextMamba3D(**model_cfg)
assert model.text_gate is not None, "TextScaleGate not initialized"
assert model.decoder.use_cross_scale_skip, "CrossScaleSkipAttention not enabled"
for attn in model.decoder.cross_scale_attns:
    assert attn.pseudo_query.abs().max() < 1e-8, "Pseudo-query not zero-init"
print("V4.6 patches verified")
```

3. **Gate monitoring in training loop:**
```python
if model.text_gate is not None:
    for i, gate_module in enumerate(model.text_gate.gates):
        bias = gate_module.gate_proj.bias.item()
        print(f"  Gate {i}: bias={bias:.4f}, sigmoid={torch.sigmoid(torch.tensor(bias)):.4f}")
```

- [ ] **Step 1: Generate notebook via crossfire**
- [ ] **Step 2: Verify notebook loads without errors**
- [ ] **Step 3: Commit**

---

### Task 10: V4.6 Evaluation Notebook (Post-Training)

Deferred until v4.6 training produces a checkpoint. Mirrors V4.5 eval notebook with:

1. **Extended ablation:** v4.5 baseline vs v4.6 full vs v4.6 Dir A only vs v4.6 Dir B only
2. **Gate analysis:** Visualize learned gate biases per scale
3. **Cross-scale attention weights:** Which scales each decoder stage attends to

- [ ] **Step 1: Train v4.6 on Colab A100**
- [ ] **Step 2: Create eval notebook**
- [ ] **Step 3: Run evaluation**
- [ ] **Step 4: Compare results**

---

## Summary

| Component | V4.5 | V4.6 |
|-----------|------|------|
| Skip connections | `x + skip_proj(matched_level)` | `x + skip_proj(matched) + cross_scale_attn(all_levels)` |
| Text fusion | Unconditional replacement | Gated via TextScaleGate per scale |
| New params | — | ~50K (CrossScale) + ~1.5K (Gates) |
| VRAM impact | — | Negligible |
| Config | textbrats_v7.yaml | textbrats_v8.yaml |
| Backward compat | — | Both features default to off |
| Scripts updated | — | train.py, evaluate_full.py, inference.py |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| New modules hurt | Zero-init → training starts from v4.5-equivalent behavior |
| VRAM overflow | Cross-scale is O(S) not O(N), <100K new params |
| TC regression persists | TextScaleGate learns to suppress text at harmful scales |
| Spatial detail loss | Original skip_proj KEPT; cross-scale is supplemental only |
| Checkpoint compat | `use_cross_scale_skip=False, use_text_gate=False` loads v4.5 |
| Config fields ignored | All 3 scripts explicitly forward new fields with .get() defaults |
