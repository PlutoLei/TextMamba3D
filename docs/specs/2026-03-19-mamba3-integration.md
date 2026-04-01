# Mamba-3 SSM Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Mamba-3 (complex-valued SSM + exponential-trapezoidal discretization) as an opt-in SSM backend for TextMamba3D, controlled by a single `use_mamba3` config flag, enabling A/B comparison against current Mamba-1 baseline.

**Architecture:** Feature-flag approach — `use_mamba3=False` (default) preserves all existing behavior. When `True`, every `_create_ssm()` call instantiates `mamba_ssm.Mamba3` instead of `mamba_ssm.Mamba`. The 3D CrossScan structure (3-axis scanning + DWConv + Uncertainty Gating) is unchanged; only the inner SSM is swapped. Text encoder's lightweight MambaLayer stays on Mamba-1 (not worth upgrading a 2-layer adapter). A new config `textbrats_a100_v5.yaml` enables Mamba3 with `d_state=64, headdim=48`.

**Tech Stack:** PyTorch, `mamba-ssm>=2.3.1` (Mamba3), Triton (A100 SM_80), pytest

**Motivation (from Mamba-3 paper, arXiv 2603.15569):**
- Complex-valued SSM encodes rotational dynamics → may fix TC Dice regression in cross-scan
- Half the state size achieves Mamba-2 parity → memory efficient
- Exponential-trapezoidal discretization implicitly replaces short causal convolution
- MIMO formulation increases hardware utilization during decoding (future Phase 2)

**Key constraint:** Mamba3 internal structure completely different from Mamba-1 — **no weight migration possible, must train from scratch.**

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `models/mamba_block.py` | Modify | Add `Mamba3` import, update `_create_ssm()`, propagate `use_mamba3` flag through all block/layer classes |
| `models/encoder_3d.py` | Modify | Accept and forward `use_mamba3` to `CrossScanBiMamba3DLayer` |
| `models/decoder_3d.py` | Modify | Accept and forward `use_mamba3` to `CrossScanBiMamba3DLayer` |
| `models/textmamba3d.py` | Modify | Add `use_mamba3` constructor parameter, pass to encoder + decoder |
| `train.py` | Modify | Read `use_mamba3` from config, pass to model constructor |
| `configs/textbrats_a100_v5.yaml` | Create | Mamba3-specific config (d_state=64, headdim=48) |
| `tests/test_mamba3.py` | Create | Mamba3 integration tests (shape, forward, gradient, config toggle) |

**Not modified:**
- `models/text_encoder.py` — MambaLayer for text stays Mamba-1 (lightweight 2-layer adapter post-BERT)
- `models/fusion.py` — MambaFusion uses MambaLayer, not in active architecture path
- `losses/` — No changes needed
- `data/` — No changes needed

---

## Chunk 1: Core SSM Swap + Tests

### Task 1: Add Mamba3 availability detection to mamba_block.py

**Files:**
- Modify: `models/mamba_block.py:1-13`

- [ ] **Step 1: Add Mamba3 import alongside existing Mamba import**

In `models/mamba_block.py`, after the existing Mamba import block (lines 8-13), add Mamba3 detection:

```python
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    Mamba = None
    MAMBA_AVAILABLE = False

try:
    from mamba_ssm import Mamba3
    MAMBA3_AVAILABLE = True
except ImportError:
    Mamba3 = None
    MAMBA3_AVAILABLE = False
```

- [ ] **Step 2: Verify import doesn't break existing code**

Run: `cd E:/VSCode_Project/TextMamba3D && python -c "from models.mamba_block import MAMBA_AVAILABLE; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add models/mamba_block.py
git commit -m "feat: add Mamba3 availability detection"
```

---

### Task 2: Update `_create_ssm()` to support Mamba3 backend

**Files:**
- Modify: `models/mamba_block.py:16-28` (the `_create_ssm` function)

- [ ] **Step 1: Write the failing test**

Create `tests/test_mamba3.py`:

```python
# tests/test_mamba3.py
"""Tests for Mamba-3 SSM integration."""
import pytest
import torch


def _has_mamba3():
    try:
        from mamba_ssm import Mamba3
        return True
    except ImportError:
        return False


requires_mamba3 = pytest.mark.skipif(
    not _has_mamba3(), reason="mamba_ssm with Mamba3 not installed"
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


class TestCreateSSM:
    """Test _create_ssm factory with use_mamba3 flag."""

    def test_create_ssm_mamba1_default(self):
        """Default _create_ssm returns Mamba (or fallback MLP)."""
        from models.mamba_block import _create_ssm
        ssm = _create_ssm(dim=96, d_state=16, d_conv=4, expand=2)
        assert ssm is not None

    def test_create_ssm_mamba3_flag_false(self):
        """use_mamba3=False returns Mamba-1 SSM."""
        from models.mamba_block import _create_ssm
        ssm = _create_ssm(dim=96, d_state=16, d_conv=4, expand=2, use_mamba3=False)
        assert ssm is not None

    @requires_mamba3
    @requires_cuda
    def test_create_ssm_mamba3_returns_mamba3(self):
        """use_mamba3=True returns Mamba3 instance."""
        from mamba_ssm import Mamba3
        from models.mamba_block import _create_ssm
        ssm = _create_ssm(dim=48, d_state=64, d_conv=4, expand=2,
                          use_mamba3=True, headdim=48)
        assert isinstance(ssm, Mamba3)

    @requires_mamba3
    @requires_cuda
    def test_create_ssm_mamba3_forward_shape(self):
        """Mamba3 SSM preserves (B, L, D) shape."""
        from models.mamba_block import _create_ssm
        ssm = _create_ssm(dim=48, d_state=64, d_conv=4, expand=2,
                          use_mamba3=True, headdim=48).cuda()
        x = torch.randn(2, 100, 48).cuda()
        out = ssm(x)
        assert out.shape == (2, 100, 48)

    @requires_mamba3
    @requires_cuda
    def test_create_ssm_mamba3_auto_headdim(self):
        """When headdim not specified, _create_ssm auto-selects valid headdim."""
        from models.mamba_block import _create_ssm
        # dim=96, expand=2 → d_inner=192, must be divisible by headdim
        ssm = _create_ssm(dim=96, d_state=64, d_conv=4, expand=2,
                          use_mamba3=True)
        assert ssm is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd E:/VSCode_Project/TextMamba3D && python -m pytest tests/test_mamba3.py::TestCreateSSM::test_create_ssm_mamba3_flag_false -v`
Expected: FAIL — `_create_ssm() got an unexpected keyword argument 'use_mamba3'`

- [ ] **Step 3: Update `_create_ssm()` to accept Mamba3 options**

Replace the entire `_create_ssm` function in `models/mamba_block.py`:

```python
def _auto_headdim(d_inner: int) -> int:
    """Pick largest valid headdim from preferred list."""
    for hd in [64, 48, 32, 24, 16]:
        if d_inner % hd == 0:
            return hd
    return 1


def _create_ssm(
    dim: int,
    d_state: int,
    d_conv: int,
    expand: int,
    dropout: float = 0.0,
    use_mamba3: bool = False,
    headdim: int | None = None,
):
    """Create a Mamba SSM (v1 or v3) or MLP fallback.

    Args:
        dim: Model dimension (d_model).
        d_state: SSM state expansion factor.
        d_conv: Local convolution width (Mamba-1 only; ignored by Mamba3).
        expand: Block expansion factor.
        dropout: Dropout rate (absorbed by Mamba3 as kwarg).
        use_mamba3: If True, use Mamba3 complex-valued SSM.
        headdim: Head dimension for Mamba3. Auto-selected if None.
    """
    if use_mamba3 and MAMBA3_AVAILABLE:
        d_inner = int(dim * expand)
        hd = headdim or _auto_headdim(d_inner)
        try:
            return Mamba3(
                d_model=dim,
                d_state=d_state,
                expand=expand,
                headdim=hd,
                dropout=dropout,
            )
        except Exception as e:
            print(f"Warning: Mamba3 init failed ({e}), falling back to Mamba-1")

    if MAMBA_AVAILABLE:
        try:
            return Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        except Exception as e:
            print(f"Warning: Mamba init failed ({e}), using fallback MLP")

    return nn.Sequential(
        nn.Linear(dim, dim * expand),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * expand, dim),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd E:/VSCode_Project/TextMamba3D && python -m pytest tests/test_mamba3.py::TestCreateSSM -v`
Expected: PASS (Mamba3-specific tests may skip if not on CUDA/no mamba3)

- [ ] **Step 5: Verify existing tests still pass**

Run: `cd E:/VSCode_Project/TextMamba3D && python -m pytest tests/test_models.py::TestMambaBlock -v`
Expected: All PASS (backward compatible — `use_mamba3` defaults to `False`)

- [ ] **Step 6: Commit**

```bash
git add models/mamba_block.py tests/test_mamba3.py
git commit -m "feat: _create_ssm() supports Mamba3 backend via use_mamba3 flag"
```

---

### Task 3: Propagate `use_mamba3` through block and layer classes

**Files:**
- Modify: `models/mamba_block.py:31-277` (MambaBlock, MambaLayer, BiMambaBlock, BiMambaLayer, CrossScanBiMamba3DBlock, CrossScanBiMamba3DLayer)

The propagation pattern is identical for every class: add `use_mamba3=False` and `headdim=None` to `__init__`, forward them to `_create_ssm()`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mamba3.py`:

```python
class TestCrossScanMamba3D:
    """Test CrossScanBiMamba3DBlock/Layer with Mamba3."""

    @requires_mamba3
    @requires_cuda
    def test_crossscan_block_mamba3_shape(self):
        """CrossScanBiMamba3DBlock with Mamba3 preserves shape."""
        from models.mamba_block import CrossScanBiMamba3DBlock
        block = CrossScanBiMamba3DBlock(
            dim=48, spatial_dims=(8, 8, 8), d_state=64,
            use_mamba3=True, headdim=48,
        ).cuda()
        x = torch.randn(1, 512, 48).cuda()  # 8*8*8=512
        out = block(x)
        assert out.shape == (1, 512, 48)

    @requires_mamba3
    @requires_cuda
    def test_crossscan_layer_mamba3_shape(self):
        """CrossScanBiMamba3DLayer with Mamba3 preserves shape."""
        from models.mamba_block import CrossScanBiMamba3DLayer
        layer = CrossScanBiMamba3DLayer(
            dim=48, depth=2, spatial_dims=(8, 8, 8), d_state=64,
            use_mamba3=True, headdim=48,
        ).cuda()
        x = torch.randn(1, 512, 48).cuda()
        out = layer(x)
        assert out.shape == (1, 512, 48)

    @requires_mamba3
    @requires_cuda
    def test_crossscan_block_mamba3_gradient(self):
        """Mamba3 CrossScan block produces valid gradients."""
        from models.mamba_block import CrossScanBiMamba3DBlock
        block = CrossScanBiMamba3DBlock(
            dim=48, spatial_dims=(4, 4, 4), d_state=64,
            use_mamba3=True, headdim=48,
        ).cuda()
        x = torch.randn(1, 64, 48, requires_grad=True).cuda()
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_crossscan_block_default_no_mamba3(self):
        """Default CrossScanBiMamba3DBlock uses Mamba-1 (backward compat)."""
        from models.mamba_block import CrossScanBiMamba3DBlock
        # Should work without CUDA (falls back to MLP if no mamba_ssm)
        block = CrossScanBiMamba3DBlock(dim=48, spatial_dims=(4, 4, 4), d_state=16)
        x = torch.randn(1, 64, 48)
        out = block(x)
        assert out.shape == (1, 64, 48)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd E:/VSCode_Project/TextMamba3D && python -m pytest tests/test_mamba3.py::TestCrossScanMamba3D::test_crossscan_block_default_no_mamba3 -v`
Expected: FAIL — `__init__() got an unexpected keyword argument 'use_mamba3'`

- [ ] **Step 3: Add `use_mamba3` and `headdim` to all block/layer classes**

**MambaBlock** (line ~31): Add `use_mamba3=False, headdim=None` to `__init__`, pass to `_create_ssm()`.

```python
class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.0,
                 use_mamba3=False, headdim=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = _create_ssm(dim, d_state, d_conv, expand, dropout,
                                 use_mamba3=use_mamba3, headdim=headdim)
        self.dropout = nn.Dropout(dropout)
```

**MambaLayer** (line ~58): Add flags, forward to MambaBlock.

```python
class MambaLayer(nn.Module):
    def __init__(self, dim, depth, d_state=16, d_conv=4, expand=2, dropout=0.0,
                 use_mamba3=False, headdim=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(dim, d_state, d_conv, expand, dropout,
                       use_mamba3=use_mamba3, headdim=headdim)
            for _ in range(depth)
        ])
```

**BiMambaBlock** (line ~86): Add flags, forward to both `_create_ssm()` calls.

```python
class BiMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.0,
                 use_mamba3=False, headdim=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.forward_ssm = _create_ssm(dim, d_state, d_conv, expand, dropout,
                                       use_mamba3=use_mamba3, headdim=headdim)
        self.backward_ssm = _create_ssm(dim, d_state, d_conv, expand, dropout,
                                        use_mamba3=use_mamba3, headdim=headdim)
        self.merge = nn.Linear(dim * 2, dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
```

**BiMambaLayer** (line ~128): Forward to BiMambaBlock.

```python
class BiMambaLayer(nn.Module):
    def __init__(self, dim, depth, d_state=16, d_conv=4, expand=2, dropout=0.0,
                 use_mamba3=False, headdim=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            BiMambaBlock(dim, d_state, d_conv, expand, dropout,
                         use_mamba3=use_mamba3, headdim=headdim)
            for _ in range(depth)
        ])
```

**CrossScanBiMamba3DBlock** (line ~156): Add flags, forward to all 3 `_create_ssm()` calls.

```python
class CrossScanBiMamba3DBlock(nn.Module):
    def __init__(self, dim, spatial_dims, d_state=16, d_conv=4, expand=2,
                 dropout=0.0, use_mamba3=False, headdim=None):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

        ssm_kwargs = dict(dim=dim, d_state=d_state, d_conv=d_conv, expand=expand,
                          dropout=dropout, use_mamba3=use_mamba3, headdim=headdim)
        self.dhw_fwd = _create_ssm(**ssm_kwargs)
        self.hwd_fwd = _create_ssm(**ssm_kwargs)
        self.wdh_fwd = _create_ssm(**ssm_kwargs)
        # ... rest unchanged (merge, gelu, dropout, uncertainty_head)
```

**CrossScanBiMamba3DLayer** (line ~249): Forward to CrossScanBiMamba3DBlock.

```python
class CrossScanBiMamba3DLayer(nn.Module):
    def __init__(self, dim, depth, spatial_dims, d_state=16, d_conv=4, expand=2,
                 dropout=0.0, use_checkpoint=False, use_mamba3=False, headdim=None):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            CrossScanBiMamba3DBlock(dim, spatial_dims, d_state, d_conv, expand,
                                    dropout, use_mamba3=use_mamba3, headdim=headdim)
            for _ in range(depth)
        ])
```

- [ ] **Step 4: Run all tests**

Run: `cd E:/VSCode_Project/TextMamba3D && python -m pytest tests/test_mamba3.py tests/test_models.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add models/mamba_block.py tests/test_mamba3.py
git commit -m "feat: propagate use_mamba3 flag through all Mamba block/layer classes"
```

---

## Chunk 2: Model + Config Integration

### Task 4: Propagate `use_mamba3` through encoder and decoder

**Files:**
- Modify: `models/encoder_3d.py:130-216` (MambaEncoder3D)
- Modify: `models/decoder_3d.py:29-170` (MambaDecoder3D)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mamba3.py`:

```python
class TestEncoderDecoderMamba3:
    """Test encoder/decoder accept use_mamba3 flag."""

    @requires_mamba3
    @requires_cuda
    def test_encoder_mamba3_forward(self):
        """MambaEncoder3D with Mamba3 produces correct number of features."""
        from models.encoder_3d import MambaEncoder3D
        enc = MambaEncoder3D(
            img_size=(32, 32, 32), in_channels=4, embed_dim=48,
            depths=[1, 1, 1, 1], patch_size=(4, 4, 4), d_state=64,
            use_mamba3=True, headdim=48,
        ).cuda()
        x = torch.randn(1, 4, 32, 32, 32).cuda()
        features = enc(x)
        assert len(features) == 4
        # Stage dims: 48, 96, 192, 384
        assert features[0].shape[-1] == 48
        assert features[3].shape[-1] == 384

    @requires_mamba3
    @requires_cuda
    def test_decoder_mamba3_forward(self):
        """MambaDecoder3D with Mamba3 produces correct output shape."""
        from models.decoder_3d import MambaDecoder3D
        dec = MambaDecoder3D(
            img_size=(32, 32, 32), patch_size=(4, 4, 4), out_channels=4,
            embed_dim=48, depths=[1, 1, 1, 1], d_state=64,
            use_mamba3=True, headdim=48,
        ).cuda()
        # Simulate encoder features (4 stages)
        features = [
            torch.randn(1, 512, 48).cuda(),   # 8*8*8, dim=48
            torch.randn(1, 64, 96).cuda(),     # 4*4*4, dim=96
            torch.randn(1, 8, 192).cuda(),     # 2*2*2, dim=192
            torch.randn(1, 1, 384).cuda(),     # 1*1*1, dim=384
        ]
        out = dec(features)
        assert out.shape == (1, 4, 32, 32, 32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd E:/VSCode_Project/TextMamba3D && python -m pytest tests/test_mamba3.py::TestEncoderDecoderMamba3::test_encoder_mamba3_forward -v`
Expected: FAIL — `MambaEncoder3D.__init__() got an unexpected keyword argument 'use_mamba3'`

- [ ] **Step 3: Add `use_mamba3` and `headdim` to MambaEncoder3D**

In `models/encoder_3d.py`, `MambaEncoder3D.__init__()`:

```python
class MambaEncoder3D(nn.Module):
    def __init__(
        self,
        img_size=(96, 96, 96), in_channels=4, embed_dim=96,
        depths=[2, 2, 2, 2], patch_size=(4, 4, 4), d_state=16,
        dropout=0.0, use_checkpoint=False,
        use_mamba3=False, headdim=None,  # NEW
    ):
        # ... (patch_embed, spatial dims unchanged) ...

        for i, depth in enumerate(depths):
            dim = embed_dim * (2 ** i)
            spatial = (d // (2 ** i), h // (2 ** i), w // (2 ** i))
            stage = CrossScanBiMamba3DLayer(
                dim=dim, depth=depth, spatial_dims=spatial,
                d_state=d_state, dropout=dropout,
                use_checkpoint=use_checkpoint,
                use_mamba3=use_mamba3, headdim=headdim,  # NEW
            )
            self.stages.append(stage)
```

- [ ] **Step 4: Add `use_mamba3` and `headdim` to MambaDecoder3D**

In `models/decoder_3d.py`, `MambaDecoder3D.__init__()`:

```python
class MambaDecoder3D(nn.Module):
    def __init__(
        self,
        img_size=(96, 96, 96), patch_size=(4, 4, 4), out_channels=4,
        embed_dim=96, depths=[2, 2, 2, 2], d_state=16,
        dropout=0.0, use_checkpoint=False, deep_supervision=False,
        use_cross_scale_skip=False,
        use_mamba3=False, headdim=None,  # NEW
    ):
        # ... in the loop ...
            stage = CrossScanBiMamba3DLayer(
                dim=dim, depth=depths[i], spatial_dims=spatial,
                d_state=d_state, dropout=dropout,
                use_checkpoint=use_checkpoint,
                use_mamba3=use_mamba3, headdim=headdim,  # NEW
            )
```

- [ ] **Step 5: Run tests**

Run: `cd E:/VSCode_Project/TextMamba3D && python -m pytest tests/test_mamba3.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add models/encoder_3d.py models/decoder_3d.py tests/test_mamba3.py
git commit -m "feat: propagate use_mamba3 through encoder and decoder"
```

---

### Task 5: Add `use_mamba3` to top-level TextMamba3D model

**Files:**
- Modify: `models/textmamba3d.py:15-98`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mamba3.py`:

```python
class TestTextMamba3DWithMamba3:
    """End-to-end TextMamba3D with Mamba3 backend."""

    @requires_mamba3
    @requires_cuda
    def test_full_model_mamba3_forward(self):
        """Full TextMamba3D forward pass with Mamba3."""
        from models.textmamba3d import TextMamba3D
        model = TextMamba3D(
            img_size=(32, 32, 32), in_channels=4, out_channels=4,
            embed_dim=48, depths=[1, 1, 1, 1], patch_size=(4, 4, 4),
            d_state=64, use_pretrained_text=False,
            use_mamba3=True, headdim=48,
        ).cuda()
        img = torch.randn(1, 4, 32, 32, 32).cuda()
        text = torch.randint(0, 100, (1, 16)).cuda()
        out = model(img, text)
        assert out.shape == (1, 4, 32, 32, 32)

    @requires_mamba3
    @requires_cuda
    def test_full_model_mamba3_no_text(self):
        """Forward without text still works with Mamba3."""
        from models.textmamba3d import TextMamba3D
        model = TextMamba3D(
            img_size=(32, 32, 32), in_channels=4, out_channels=4,
            embed_dim=48, depths=[1, 1, 1, 1], patch_size=(4, 4, 4),
            d_state=64, use_pretrained_text=False,
            use_mamba3=True, headdim=48,
        ).cuda()
        img = torch.randn(1, 4, 32, 32, 32).cuda()
        out = model(img, use_text=False)
        assert out.shape == (1, 4, 32, 32, 32)

    @requires_mamba3
    @requires_cuda
    def test_full_model_mamba3_gradient_flow(self):
        """Gradients flow through entire Mamba3 model."""
        from models.textmamba3d import TextMamba3D
        model = TextMamba3D(
            img_size=(32, 32, 32), in_channels=4, out_channels=4,
            embed_dim=48, depths=[1, 1, 1, 1], patch_size=(4, 4, 4),
            d_state=64, use_pretrained_text=False,
            use_mamba3=True, headdim=48,
        ).cuda()
        img = torch.randn(1, 4, 32, 32, 32).cuda()
        text = torch.randint(0, 100, (1, 16)).cuda()
        out = model(img, text)
        loss = out.sum()
        loss.backward()
        # Check encoder SSM has gradients
        enc_block = model.img_encoder.stages[0].blocks[0]
        ssm_params = list(enc_block.dhw_fwd.parameters())
        assert len(ssm_params) > 0
        assert ssm_params[0].grad is not None

    @requires_mamba3
    @requires_cuda
    def test_param_count_mamba3_vs_mamba1(self):
        """Mamba3 model has reasonable param count relative to Mamba1."""
        from models.textmamba3d import TextMamba3D
        kwargs = dict(
            img_size=(32, 32, 32), in_channels=4, out_channels=4,
            embed_dim=48, depths=[1, 1, 1, 1], patch_size=(4, 4, 4),
            use_pretrained_text=False,
        )
        m1 = TextMamba3D(d_state=16, use_mamba3=False, **kwargs)
        m3 = TextMamba3D(d_state=64, use_mamba3=True, headdim=48, **kwargs)
        p1 = sum(p.numel() for p in m1.parameters())
        p3 = sum(p.numel() for p in m3.parameters())
        # Mamba3 should not be more than 5x larger
        assert p3 < p1 * 5, f"Mamba3 params ({p3:,}) > 5x Mamba1 ({p1:,})"
        print(f"Mamba1: {p1:,} params, Mamba3: {p3:,} params, ratio: {p3/p1:.2f}x")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd E:/VSCode_Project/TextMamba3D && python -m pytest tests/test_mamba3.py::TestTextMamba3DWithMamba3::test_full_model_mamba3_forward -v`
Expected: FAIL — `TextMamba3D.__init__() got an unexpected keyword argument 'use_mamba3'`

- [ ] **Step 3: Add `use_mamba3` and `headdim` to TextMamba3D**

In `models/textmamba3d.py`, add to `__init__` signature and forward to encoder + decoder:

```python
class TextMamba3D(nn.Module):
    def __init__(
        self,
        # ... existing params ...
        text_gate_init_bias: float = 2.0,
        # V5.0 Mamba-3 parameters
        use_mamba3: bool = False,
        headdim: int | None = None,
    ) -> None:
        super().__init__()
        # ... existing code ...

        self.img_encoder = MambaEncoder3D(
            img_size=img_size, in_channels=in_channels,
            embed_dim=embed_dim, depths=depths, patch_size=patch_size,
            d_state=d_state, dropout=dropout, use_checkpoint=use_checkpoint,
            use_mamba3=use_mamba3, headdim=headdim,  # NEW
        )

        # ... text_encoder unchanged (stays Mamba-1) ...

        self.decoder = MambaDecoder3D(
            img_size=img_size, patch_size=patch_size,
            out_channels=out_channels, embed_dim=embed_dim,
            depths=depths, d_state=d_state, dropout=dropout,
            use_checkpoint=use_checkpoint, deep_supervision=deep_supervision,
            use_cross_scale_skip=use_cross_scale_skip,
            use_mamba3=use_mamba3, headdim=headdim,  # NEW
        )
```

- [ ] **Step 4: Run all tests**

Run: `cd E:/VSCode_Project/TextMamba3D && python -m pytest tests/test_mamba3.py tests/test_models.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add models/textmamba3d.py tests/test_mamba3.py
git commit -m "feat: TextMamba3D accepts use_mamba3 flag for V5.0 experiments"
```

---

### Task 6: Config and training script integration

**Files:**
- Create: `configs/textbrats_a100_v5.yaml`
- Modify: `train.py:372-389`

- [ ] **Step 1: Create V5.0 config file**

```yaml
# configs/textbrats_a100_v5.yaml
# A100 40GB — Mamba-3 SSM backbone (V5.0 experiment)
# Key change: mamba_ssm.Mamba → mamba_ssm.Mamba3 (complex-valued SSM)
# Must train from scratch — Mamba3 weights incompatible with Mamba-1

data:
  data_dir: "./data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
  dataset_type: "textbrats"
  patch_size: [128, 128, 128]
  batch_size: 2                  # 4→2: Mamba3 d_state=64 uses more memory
  num_workers: 4
  train_ratio: 0.596
  val_ratio: 0.149

model:
  img_size: [128, 128, 128]
  in_channels: 4
  out_channels: 4
  embed_dim: 48
  depths: [2, 2, 2, 2]
  dropout: 0.1
  text_embed_dim: 256
  text_max_len: 256
  use_pretrained_text: true
  unfreeze_text_layers: 2
  text_model_path: null
  # V5.0 Mamba-3 settings
  use_mamba3: true
  headdim: 48                    # 48 divides all stage dims [48, 96, 192, 384]
  d_state: 64                   # Paper: d_state=64 matches Mamba-2 d_state=128

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
  gradient_accumulation: 2       # batch_size=2 * grad_accum=2 = effective batch 4
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
  name: "TextMamba3D_A100_v5.0_mamba3"
  description: "V5.0: Mamba3 complex-valued SSM backbone, d_state=64, headdim=48"
```

- [ ] **Step 2: Update train.py to read Mamba3 config**

In `train.py`, update the model construction block (around line 372):

```python
    model = TextMamba3D(
        img_size=tuple(config['model']['img_size']),
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        text_embed_dim=config['model']['text_embed_dim'],
        text_max_len=max_text_len,
        use_pretrained_text=use_pretrained_text,
        unfreeze_text_layers=config['model'].get('unfreeze_text_layers', 0),
        use_checkpoint=use_checkpoint,
        text_model_path=text_model_path,
        deep_supervision=deep_supervision,
        dropout=config['model'].get('dropout', 0.0),
        use_text_gate=config['model'].get('use_text_gate', False),
        use_cross_scale_skip=config['model'].get('use_cross_scale_skip', False),
        text_gate_init_bias=config['model'].get('text_gate_init_bias', 2.0),
        # V5.0 Mamba-3
        use_mamba3=config['model'].get('use_mamba3', False),
        headdim=config['model'].get('headdim', None),
    ).to(device)
```

Also add a print line after model creation:

```python
    if config['model'].get('use_mamba3', False):
        print(f'SSM Backend: Mamba-3 (d_state={config["model"].get("d_state", 64)}, '
              f'headdim={config["model"].get("headdim", "auto")})')
    else:
        print('SSM Backend: Mamba-1 (default)')
```

- [ ] **Step 3: Also pass d_state from config (currently hardcoded)**

In `train.py`, the `d_state` is not being passed from config. Fix by adding it to model construction. Check if it's already set — looking at the existing `TextMamba3D` constructor call, `d_state` is NOT passed (uses default 16). Add it:

```python
        d_state=config['model'].get('d_state', 16),
```

This line goes inside the `TextMamba3D()` constructor call alongside the other model params.

- [ ] **Step 4: Run existing config to verify backward compatibility**

Run: `cd E:/VSCode_Project/TextMamba3D && python -c "
import yaml
from models import TextMamba3D
cfg = yaml.safe_load(open('configs/textbrats_a100.yaml'))
m = cfg['model']
model = TextMamba3D(
    img_size=tuple(m['img_size']), in_channels=m['in_channels'],
    out_channels=m['out_channels'], embed_dim=m['embed_dim'],
    depths=m['depths'], use_pretrained_text=False,
    use_mamba3=m.get('use_mamba3', False),
)
print(f'OK: {sum(p.numel() for p in model.parameters()):,} params')
"`
Expected: `OK: <N> params` (no error, use_mamba3 defaults to False)

- [ ] **Step 5: Commit**

```bash
git add configs/textbrats_a100_v5.yaml train.py tests/test_mamba3.py
git commit -m "feat: V5.0 config + train.py reads use_mamba3 from config"
```

---

## Chunk 3: Notebook + Memory Update

### Task 7: Create V5.0 training notebook cell header

**Files:**
- This is documentation/notebook prep only — the actual notebook will be created in a Colab session.

- [ ] **Step 1: Verify the full test suite passes**

Run: `cd E:/VSCode_Project/TextMamba3D && python -m pytest tests/ -v --tb=short`
Expected: All existing + new tests PASS

- [ ] **Step 2: Update task file**

Update `.claude/tasks/bs6222_textmamba3d_proposal.md` to reflect V5.0 plan:

Add under `## 待完成事项`:
```markdown
- [ ] V5.0: Mamba-3 SSM 集成（代码完成，待 A100 训练验证）
  - 实施计划: `TextMamba3D/docs/specs/2026-03-19-mamba3-integration.md`
  - 配置: `configs/textbrats_a100_v5.yaml`
  - 测试: `tests/test_mamba3.py`
```

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat(v5.0): Mamba-3 SSM integration - code ready for A100 training"
```

---

## Summary

| Task | What | Files | Est. Lines Changed |
|------|------|-------|-------------------|
| 1 | Mamba3 import detection | `mamba_block.py` | +7 |
| 2 | `_create_ssm()` Mamba3 backend | `mamba_block.py`, `test_mamba3.py` | +50 |
| 3 | Propagate through block/layer | `mamba_block.py`, `test_mamba3.py` | +30 |
| 4 | Encoder + Decoder flag | `encoder_3d.py`, `decoder_3d.py`, `test_mamba3.py` | +20 |
| 5 | TextMamba3D flag | `textmamba3d.py`, `test_mamba3.py` | +20 |
| 6 | Config + train.py | `textbrats_a100_v5.yaml`, `train.py` | +70 |
| 7 | Task file + final validation | task file | +5 |

**Total: ~200 lines of changes across 7 files, ~100 lines of new tests.**

**Risk mitigations built in:**
- `use_mamba3=False` default → existing behavior 100% preserved
- Mamba3 unavailable → graceful fallback to Mamba-1
- Mamba3 init failure → graceful fallback to Mamba-1
- `batch_size=2` + `grad_accum=2` in V5 config → memory safety on A100 40GB
- All existing tests must pass → backward compatibility enforced

**Post-implementation (Colab A100):**
1. `pip install mamba-ssm>=2.3.1 --no-build-isolation`
2. `python train.py --config configs/textbrats_a100_v5.yaml`
3. Compare V5.0 Dice against V4.x baseline (~88%)
4. Monitor TC Dice specifically (the metric Mamba3 complex-valued SSM targets)
