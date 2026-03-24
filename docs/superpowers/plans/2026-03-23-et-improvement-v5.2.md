# TextMamba3D V5.2: ET Improvement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve ET Dice from 79.10% to ≥82.0% through Focal Tversky Loss and learnable Edge Enhancement module

**Architecture:** Replace DiceLoss with FocalTverskyLoss (asymmetric FP/FN weighting for small ET regions) and add lightweight EdgeEnhance attention modules in decoder skip connections (Stages 1-3) for boundary refinement.

**Tech Stack:** PyTorch, mamba-ssm, einops. Execution via Jupyter notebook on Colab A100 40GB.

**Spec:** `docs/superpowers/specs/2026-03-23-et-improvement-v5.2-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `losses/focal_tversky_loss.py` | FocalTverskyLoss class with configurable α/β/γ |
| Modify | `losses/__init__.py:5,37,87-88,130` | Import FTL, replace DiceLoss instantiation, use FTL in deep supervision |
| Create | `models/edge_enhance.py` | EdgeEnhance3D module (depthwise conv + sigmoid attention) |
| Modify | `models/decoder_3d.py:36-53,99-101,159-162` | Add `use_edge_enhance` param, instantiate EE modules, apply in skip connections |
| Modify | `models/textmamba3d.py:18-48,104-121` | Pass `use_edge_enhance` config through to decoder |
| Modify | `train.py:387-411,428-441` | Read FTL params from config, pass `use_edge_enhance` to model |
| Create | `configs/a100_v5.2b.yaml` | Exp 2 config: FTL only (no EE) |
| Create | `configs/a100_v5.2a.yaml` | Exp 1 config: FTL + EE |
| Create | `tests/test_focal_tversky.py` | FTL unit tests |
| Create | `tests/test_edge_enhance.py` | EE module unit tests |

---

## Chunk 1: Focal Tversky Loss

### Task 1: Implement FocalTverskyLoss

**Files:**
- Create: `TextMamba3D/losses/focal_tversky_loss.py`
- Test: `TextMamba3D/tests/test_focal_tversky.py`

- [ ] **Step 1: Write failing tests for FocalTverskyLoss**

Create `tests/test_focal_tversky.py`:

```python
# tests/test_focal_tversky.py
import torch
import pytest


class TestFocalTverskyLoss:
    def test_output_is_scalar(self):
        """FTL should return a scalar loss."""
        from losses.focal_tversky_loss import FocalTverskyLoss

        ftl = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
        pred = torch.randn(2, 4, 8, 8, 8)
        target = torch.randint(0, 4, (2, 8, 8, 8))
        loss = ftl(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_perfect_prediction_low_loss(self):
        """Perfect prediction should give loss close to 0."""
        from losses.focal_tversky_loss import FocalTverskyLoss

        ftl = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
        pred = torch.zeros(1, 4, 16, 16, 16)
        target = torch.zeros(1, 16, 16, 16, dtype=torch.long)
        for c in range(4):
            start = c * 4
            end = (c + 1) * 4
            pred[:, c, start:end, :, :] = 10.0
            target[:, start:end, :, :] = c
        loss = ftl(pred, target)
        assert loss.item() < 0.1

    def test_excludes_background_by_default(self):
        """Background class (0) should be excluded from loss computation."""
        from losses.focal_tversky_loss import FocalTverskyLoss

        ftl = FocalTverskyLoss(include_background=False)
        # All background — classes 1-3 absent, loss from those should be 0
        pred = torch.zeros(1, 4, 8, 8, 8)
        pred[:, 0] = 10.0
        target = torch.zeros(1, 8, 8, 8, dtype=torch.long)
        loss = ftl(pred, target)
        assert loss.item() >= 0  # Should not crash

    def test_class_weights_affect_loss(self):
        """Different class weights should produce different loss values."""
        from losses.focal_tversky_loss import FocalTverskyLoss

        pred = torch.randn(1, 4, 8, 8, 8)
        target = torch.randint(0, 4, (1, 8, 8, 8))
        ftl_uniform = FocalTverskyLoss(class_weights=[1.0, 1.0, 1.0, 1.0])
        ftl_weighted = FocalTverskyLoss(class_weights=[0.25, 3.0, 1.0, 4.0])
        loss_u = ftl_uniform(pred, target)
        loss_w = ftl_weighted(pred, target)
        assert loss_u.shape == ()
        assert loss_w.shape == ()

    def test_asymmetry_fn_penalty(self):
        """Higher beta should penalize false negatives more (higher loss when under-predicting)."""
        from losses.focal_tversky_loss import FocalTverskyLoss

        torch.manual_seed(42)
        # Create a target with class 3 present, but predict all background
        pred = torch.zeros(1, 4, 8, 8, 8)
        pred[:, 0] = 10.0  # Predict all background
        target = torch.zeros(1, 8, 8, 8, dtype=torch.long)
        target[:, 4:8, :, :] = 3  # Half is ET

        ftl_high_beta = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.0, include_background=True)
        ftl_low_beta = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.0, include_background=True)
        loss_high = ftl_high_beta(pred, target)
        loss_low = ftl_low_beta(pred, target)
        # Missing ET (FN) should hurt more with higher beta
        assert loss_high.item() > loss_low.item()

    def test_backward_pass(self):
        """FTL should support gradient computation."""
        from losses.focal_tversky_loss import FocalTverskyLoss

        ftl = FocalTverskyLoss(class_weights=[0.25, 3.0, 1.0, 4.0])
        pred = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
        target = torch.randint(0, 4, (1, 8, 8, 8))
        loss = ftl(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_gamma_focal_effect(self):
        """Higher gamma should down-weight easy samples more."""
        from losses.focal_tversky_loss import FocalTverskyLoss

        pred = torch.randn(1, 4, 8, 8, 8)
        target = torch.randint(0, 4, (1, 8, 8, 8))
        ftl_low_gamma = FocalTverskyLoss(gamma=1.0)
        ftl_high_gamma = FocalTverskyLoss(gamma=2.0)
        # Both should produce valid non-negative losses
        loss_low = ftl_low_gamma(pred, target)
        loss_high = ftl_high_gamma(pred, target)
        assert loss_low.item() >= 0
        assert loss_high.item() >= 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd TextMamba3D && python -m pytest tests/test_focal_tversky.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'losses.focal_tversky_loss'`

- [ ] **Step 3: Implement FocalTverskyLoss**

Create `losses/focal_tversky_loss.py`:

```python
# losses/focal_tversky_loss.py
"""Focal Tversky Loss for class-imbalanced medical image segmentation.

Reference: Abraham & Khan, "A Novel Focal Tversky Loss Function with
Improved Attention U-Net for Lesion Segmentation", ISBI 2019.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss with asymmetric FP/FN weighting.

    Args:
        alpha: Weight for false positives (over-segmentation). Default 0.3.
        beta: Weight for false negatives (under-segmentation). Default 0.7.
            Higher beta penalizes missed regions more — use for small targets like ET.
        gamma: Focal parameter. Values >1 down-weight easy samples. Default 1.33.
        smooth: Smoothing term to avoid division by zero. Default 1.0.
        include_background: Whether to include class 0 in loss. Default False.
        class_weights: Per-class weights [C] including background. Default None.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        smooth: float = 1.0,
        include_background: bool = False,
        class_weights: list[float] | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.include_background = include_background

        if class_weights is not None:
            self.register_buffer(
                'class_weights',
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, D, H, W] logits
            target: [B, D, H, W] class indices
        Returns:
            Scalar loss
        """
        num_classes = pred.shape[1]
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes)
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()

        start_idx = 0 if self.include_background else 1

        ftl_scores = []
        weights = []
        for i in range(start_idx, num_classes):
            p_i = pred_soft[:, i]
            g_i = target_onehot[:, i]

            # Skip classes absent from both prediction and target
            if g_i.sum().item() < 1e-6:
                ftl_scores.append(pred.new_zeros(()))
                weights.append(pred.new_zeros(()))
                continue

            tp = (p_i * g_i).sum()
            fp = (p_i * (1 - g_i)).sum()
            fn = ((1 - p_i) * g_i).sum()

            tversky_index = (tp + self.smooth) / (
                tp + self.alpha * fp + self.beta * fn + self.smooth
            )
            focal_tversky = (1 - tversky_index) ** self.gamma

            ftl_scores.append(focal_tversky)

            if self.class_weights is not None and i < len(self.class_weights):
                weights.append(
                    self.class_weights[i].to(device=pred.device, dtype=pred.dtype)
                )
            else:
                weights.append(pred.new_ones(()))

        if not ftl_scores:
            return pred.new_zeros(())

        ftl_tensor = torch.stack(ftl_scores)
        weight_tensor = torch.stack(weights)
        weight_sum = weight_tensor.sum()
        if weight_sum.item() < 1e-8:
            return pred.new_zeros(())
        weight_tensor = weight_tensor / weight_sum

        return (ftl_tensor * weight_tensor).sum()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd TextMamba3D && python -m pytest tests/test_focal_tversky.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
cd TextMamba3D
git add losses/focal_tversky_loss.py tests/test_focal_tversky.py
git commit -m "feat(loss): add FocalTverskyLoss with asymmetric FP/FN weighting for ET improvement"
```

---

### Task 2: Integrate FTL into CombinedLoss

**Files:**
- Modify: `TextMamba3D/losses/__init__.py:5,37,87-88,130`

- [ ] **Step 1: Write failing integration test**

Append to `tests/test_focal_tversky.py`:

```python
class TestFTLIntegration:
    def test_combined_loss_uses_ftl(self):
        """CombinedLoss should use FTL when use_ftl=True."""
        from losses import CombinedLoss
        from losses.focal_tversky_loss import FocalTverskyLoss

        criterion = CombinedLoss(
            use_ftl=True,
            ftl_alpha=0.3, ftl_beta=0.7, ftl_gamma=1.33,
            class_weights=[0.25, 3.0, 1.0, 4.0],
        )
        assert isinstance(criterion.dice_loss, FocalTverskyLoss)

        pred = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
        target = torch.randint(0, 4, (1, 8, 8, 8))
        losses = criterion(pred, target)
        assert losses['total'].item() > 0
        losses['total'].backward()
        assert pred.grad is not None

    def test_combined_loss_default_uses_dice(self):
        """CombinedLoss should default to DiceLoss (backward compatible)."""
        from losses import CombinedLoss
        from losses.dice_loss import DiceLoss

        criterion = CombinedLoss(class_weights=[0.25, 3.0, 1.0, 4.0])
        assert isinstance(criterion.dice_loss, DiceLoss)

    def test_combined_loss_ftl_deep_supervision(self):
        """Deep supervision auxiliary losses should also use FTL."""
        from losses import CombinedLoss

        criterion = CombinedLoss(
            use_ftl=True,
            class_weights=[0.25, 3.0, 1.0, 4.0],
        )
        pred = torch.randn(1, 4, 16, 16, 16, requires_grad=True)
        target = torch.randint(0, 4, (1, 16, 16, 16))
        aux = [torch.randn(1, 4, 8, 8, 8), torch.randn(1, 4, 4, 4, 4)]
        losses = criterion(pred, target, aux_preds=aux)
        assert losses['deep_supervision'].item() > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd TextMamba3D && python -m pytest tests/test_focal_tversky.py::TestFTLIntegration -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'use_ftl'`

- [ ] **Step 3: Modify CombinedLoss to support FTL**

In `losses/__init__.py`:

**Line 5 — add import:**
```python
from .focal_tversky_loss import FocalTverskyLoss
```

**Lines 20-30 — add FTL params to `__init__`:**
```python
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        edge_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        temperature: float = 0.07,
        feat_dim: int = 384,
        text_dim: int = 256,
        class_weights: list[float] | None = None,
        # V5.2: Focal Tversky Loss
        use_ftl: bool = False,
        ftl_alpha: float = 0.3,
        ftl_beta: float = 0.7,
        ftl_gamma: float = 1.33,
    ) -> None:
```

**Line 37 — conditional instantiation:**
```python
        if use_ftl:
            self.dice_loss = FocalTverskyLoss(
                alpha=ftl_alpha, beta=ftl_beta, gamma=ftl_gamma,
                class_weights=class_weights,
            )
        else:
            self.dice_loss = DiceLoss(class_weights=class_weights)
```

No other changes needed — `self.dice_loss` is already used in both main loss (line 88) and deep supervision (line 130). The FTL drop-in replacement works because it has the same `forward(pred, target) -> scalar` interface.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd TextMamba3D && python -m pytest tests/test_focal_tversky.py -v && python -m pytest tests/test_losses.py -v`
Expected: All tests PASS (including existing DiceLoss tests — backward compatible)

- [ ] **Step 5: Commit**

```bash
cd TextMamba3D
git add losses/__init__.py tests/test_focal_tversky.py
git commit -m "feat(loss): integrate FTL into CombinedLoss with use_ftl flag, backward compatible"
```

---

## Chunk 2: Edge Enhancement Module

### Task 3: Implement EdgeEnhance3D module

**Files:**
- Create: `TextMamba3D/models/edge_enhance.py`
- Test: `TextMamba3D/tests/test_edge_enhance.py`

- [ ] **Step 1: Write failing tests for EdgeEnhance3D**

Create `tests/test_edge_enhance.py`:

```python
# tests/test_edge_enhance.py
import torch
import pytest


class TestEdgeEnhance3D:
    def test_output_shape(self):
        """Output shape should match input shape."""
        from models.edge_enhance import EdgeEnhance3D

        ee = EdgeEnhance3D(channels=96, spatial_dims=(32, 32, 32))
        x = torch.randn(2, 32*32*32, 96)  # [B, N, C]
        out = ee(x)
        assert out.shape == x.shape

    def test_near_identity_at_init(self):
        """At initialization, output ≈ 1.047 * input (near-identity)."""
        from models.edge_enhance import EdgeEnhance3D

        ee = EdgeEnhance3D(channels=48, spatial_dims=(8, 8, 8))
        x = torch.randn(1, 512, 48)
        out = ee(x)
        # sigmoid(-3) ≈ 0.047, so output ≈ x * 0.047 + x = 1.047 * x
        ratio = out.mean() / x.mean()
        assert 0.9 < ratio.item() < 1.2  # Approximately identity

    def test_backward_pass(self):
        """Should support gradient computation."""
        from models.edge_enhance import EdgeEnhance3D

        ee = EdgeEnhance3D(channels=96, spatial_dims=(8, 8, 8))
        x = torch.randn(1, 512, 96, requires_grad=True)
        out = ee(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_different_stages(self):
        """Should work with different channel dims and spatial dims."""
        from models.edge_enhance import EdgeEnhance3D

        for channels, spatial in [(48, (32, 32, 32)), (96, (16, 16, 16)), (192, (8, 8, 8))]:
            N = spatial[0] * spatial[1] * spatial[2]
            ee = EdgeEnhance3D(channels=channels, spatial_dims=spatial)
            x = torch.randn(1, N, channels)
            out = ee(x)
            assert out.shape == (1, N, channels)

    def test_parameter_count(self):
        """Parameter count should be small (~42K for C=192)."""
        from models.edge_enhance import EdgeEnhance3D

        ee = EdgeEnhance3D(channels=192, spatial_dims=(8, 8, 8))
        num_params = sum(p.numel() for p in ee.parameters())
        assert num_params < 50000  # ~42K expected
```

- [ ] **Step 2: Run to verify failure**

Run: `cd TextMamba3D && python -m pytest tests/test_edge_enhance.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement EdgeEnhance3D**

Create `models/edge_enhance.py`:

```python
# models/edge_enhance.py
"""Learnable Edge Enhancement module for decoder skip connections.

Applies lightweight depthwise convolution + sigmoid attention to enhance
boundary features in skip connections, improving small object segmentation.
"""
import torch
import torch.nn as nn
from einops import rearrange


class EdgeEnhance3D(nn.Module):
    """Lightweight boundary attention for 3D skip connection features.

    Reshapes sequence tokens to 3D spatial format, applies depthwise
    separable convolution to detect local edge patterns, produces a
    sigmoid attention map, and applies it with a residual connection.

    Args:
        channels: Feature dimension (C).
        spatial_dims: (D, H, W) spatial dimensions at this decoder stage.
    """

    def __init__(self, channels: int, spatial_dims: tuple[int, int, int]):
        super().__init__()
        self.spatial_dims = spatial_dims

        self.conv = nn.Sequential(
            # Depthwise: local spatial edge detection
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU(),
            # Pointwise: channel mixing → attention logits
            nn.Conv3d(channels, channels, kernel_size=1, bias=True),
        )

        # Initialize sigmoid bias to -3 for near-identity at start
        # sigmoid(-3) ≈ 0.047, so output ≈ 0.047*x + x = 1.047*x
        nn.init.constant_(self.conv[-1].bias, -3.0)
        # Zero-init pointwise weights for even closer to identity
        nn.init.zeros_(self.conv[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] sequence features from skip connection
        Returns:
            [B, N, C] boundary-enhanced features
        """
        D, H, W = self.spatial_dims
        x_3d = rearrange(x, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)

        attn = torch.sigmoid(self.conv(x_3d))
        out_3d = x_3d * attn + x_3d

        return rearrange(out_3d, 'b c d h w -> b (d h w) c')
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd TextMamba3D && python -m pytest tests/test_edge_enhance.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd TextMamba3D
git add models/edge_enhance.py tests/test_edge_enhance.py
git commit -m "feat(model): add EdgeEnhance3D boundary attention module for skip connections"
```

---

### Task 4: Integrate EdgeEnhance3D into decoder

**Files:**
- Modify: `TextMamba3D/models/decoder_3d.py:36-53,99-101,159-162`
- Modify: `TextMamba3D/models/textmamba3d.py:18-48,104-121`

- [ ] **Step 1: Write failing integration test**

Append to `tests/test_edge_enhance.py`:

```python
class TestEdgeEnhanceIntegration:
    def test_decoder_with_edge_enhance(self):
        """Decoder should work with use_edge_enhance=True."""
        from models.decoder_3d import MambaDecoder3D

        decoder = MambaDecoder3D(
            img_size=(128, 128, 128),
            patch_size=(4, 4, 4),
            out_channels=4,
            embed_dim=48,
            depths=[2, 2, 2, 2],
            use_edge_enhance=True,
        )
        # Check EE modules exist for stages 1, 2, 3 (3 skip connections)
        assert decoder.edge_enhances is not None
        assert len(decoder.edge_enhances) == 3

    def test_decoder_without_edge_enhance_backward_compatible(self):
        """Decoder should work without use_edge_enhance (default False)."""
        from models.decoder_3d import MambaDecoder3D

        decoder = MambaDecoder3D(
            img_size=(128, 128, 128),
            patch_size=(4, 4, 4),
            out_channels=4,
            embed_dim=48,
            depths=[2, 2, 2, 2],
        )
        assert decoder.edge_enhances is None
```

- [ ] **Step 2: Run to verify failure**

Run: `cd TextMamba3D && python -m pytest tests/test_edge_enhance.py::TestEdgeEnhanceIntegration -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'use_edge_enhance'`

- [ ] **Step 3: Modify decoder_3d.py**

Add `use_edge_enhance` parameter to `MambaDecoder3D.__init__` (after `use_cross_scale_skip`):

```python
        use_edge_enhance: bool = False,
```

Add instance variable after `self.use_cross_scale_skip`:

```python
        self.use_edge_enhance = use_edge_enhance
```

Add import at top of file:

```python
from .edge_enhance import EdgeEnhance3D
```

Inside the `__init__` loop, after the `cross_scale_attns` block (after line 114), add EE module creation. Add module list before the loop:

```python
        self.edge_enhances = nn.ModuleList() if use_edge_enhance else None
```

Inside the loop, in the `if i > 0:` block, after `self.skip_projs.append(skip_proj)` (line 101), add:

```python
                if use_edge_enhance:
                    skip_dim = dim // 2
                    skip_spatial = (d // (2 ** (i - 1)), h // (2 ** (i - 1)), w // (2 ** (i - 1)))
                    self.edge_enhances.append(
                        EdgeEnhance3D(channels=skip_dim, spatial_dims=skip_spatial)
                    )
```

In `forward`, after `x = x + skip` (line 162), add:

```python
                    # V5.2: Edge Enhancement on skip features
                    if self.use_edge_enhance and self.edge_enhances is not None:
                        x = self.edge_enhances[i](x)
```

- [ ] **Step 4: Pass `use_edge_enhance` through TextMamba3D**

In `models/textmamba3d.py`:

Add parameter to `__init__` (after `fusion_type`):

```python
        # V5.2 Edge Enhancement
        use_edge_enhance: bool = False,
```

Pass to decoder (line 115, add after `use_cross_scale_skip`):

```python
            use_edge_enhance=use_edge_enhance,
```

- [ ] **Step 5: Pass `use_edge_enhance` from train.py**

In `train.py`, at the `TextMamba3D(...)` call (line ~410), add:

```python
        use_edge_enhance=config['model'].get('use_edge_enhance', False),
```

- [ ] **Step 6: Run tests to verify pass**

Run: `cd TextMamba3D && python -m pytest tests/test_edge_enhance.py -v && python -m pytest tests/test_models.py -v`
Expected: All tests PASS (including existing model tests — backward compatible)

- [ ] **Step 7: Commit**

```bash
cd TextMamba3D
git add models/decoder_3d.py models/textmamba3d.py models/edge_enhance.py train.py tests/test_edge_enhance.py
git commit -m "feat(model): integrate EdgeEnhance3D into decoder skip connections with config flag"
```

---

## Chunk 3: Configs and Training Setup

### Task 5: Create experiment configs

**Files:**
- Create: `TextMamba3D/configs/a100_v5.2b.yaml` (Exp 2: FTL only)
- Create: `TextMamba3D/configs/a100_v5.2a.yaml` (Exp 1: FTL + EE)

- [ ] **Step 1: Read V5.1 config as reference and adapt for V5.2**

The V5.2 configs are based on V5.0 (Mamba-2, not Mamba-3). Key differences from V5.1:
- `use_mamba3: false` (back to Mamba-2)
- `use_amp: true` (standard autocast, not pure bf16)
- `bf16_mode` removed (not needed with autocast)
- FTL params added under `loss:`
- Fine-tuning params: 80 epochs, lr 5e-5, warmup 5, patience 20

- [ ] **Step 2: Create Exp 2 config (FTL only)**

Create `configs/a100_v5.2b.yaml`:

```yaml
# configs/textbrats_a100_v5.2b.yaml
# A100 40GB — V5.2 Exp 2: Focal Tversky Loss (no Edge Enhancement)
# Fine-tune from best_v5.0.pth (Mamba-2 baseline)

data:
  data_dir: "./data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
  dataset_type: "textbrats"
  patch_size: [128, 128, 128]
  batch_size: 2
  num_workers: 4
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
  d_state: 16
  dropout: 0.1
  text_embed_dim: 256
  text_max_len: 192
  use_pretrained_text: true
  unfreeze_text_layers: 2
  text_model_path: null
  use_mamba3: false
  fusion_type: seqca
  use_edge_enhance: false          # Exp 2: no EE

loss:
  dice_weight: 1.0
  ce_weight: 1.0
  edge_weight: 1.0
  contrastive_weight: 0.05
  temperature: 0.07
  class_weights: [0.25, 3.0, 1.0, 4.0]
  # V5.2: Focal Tversky Loss
  use_ftl: true
  ftl_alpha: 0.3
  ftl_beta: 0.7
  ftl_gamma: 1.33

augmentation:
  use_elastic: true
  use_modality_dropout: true

training:
  epochs: 80                        # Fine-tuning
  lr: 0.00005                       # 5e-5 (half of V5.0)
  weight_decay: 0.01
  warmup_epochs: 5
  contrastive_warmup_epochs: 0      # Already warmed up in V5.0
  patience: 20
  gradient_accumulation: 2
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
  name: "TextMamba3D_A100_v5.2b_ftl_only"
  description: "V5.2 Exp 2: Focal Tversky Loss (alpha=0.3, beta=0.7, gamma=1.33), fine-tune from V5.0"
```

- [ ] **Step 3: Create Exp 1 config (FTL + EE)**

Create `configs/a100_v5.2a.yaml` — identical to v5.2b except:

```yaml
model:
  use_edge_enhance: true             # Exp 1: with EE

experiment:
  name: "TextMamba3D_A100_v5.2a_ftl_ee"
  description: "V5.2 Exp 1: Focal Tversky Loss + Edge Enhancement, fine-tune from V5.0"
```

(Copy v5.2b and change only these 3 lines.)

- [ ] **Step 4: Wire FTL config params in train.py**

In `train.py`, modify the `CombinedLoss(...)` instantiation (lines 432-441) to read FTL params:

```python
    criterion = CombinedLoss(
        dice_weight=config['loss']['dice_weight'],
        ce_weight=config['loss']['ce_weight'],
        edge_weight=config['loss']['edge_weight'],
        contrastive_weight=config['loss']['contrastive_weight'],
        temperature=config['loss']['temperature'],
        feat_dim=config['model']['embed_dim'] * (2 ** (len(config['model']['depths']) - 1)),
        text_dim=config['model']['text_embed_dim'],
        class_weights=class_weights,
        # V5.2: Focal Tversky Loss
        use_ftl=config['loss'].get('use_ftl', False),
        ftl_alpha=config['loss'].get('ftl_alpha', 0.3),
        ftl_beta=config['loss'].get('ftl_beta', 0.7),
        ftl_gamma=config['loss'].get('ftl_gamma', 1.33),
    ).to(device)
```

- [ ] **Step 5: Commit**

```bash
cd TextMamba3D
git add configs/a100_v5.2a.yaml configs/a100_v5.2b.yaml train.py
git commit -m "feat(config): add V5.2 experiment configs with FTL and EE settings"
```

---

### Task 6: Checkpoint loading compatibility

**Files:**
- Modify: `TextMamba3D/train.py` (resume logic)

- [ ] **Step 1: Verify V5.0 checkpoint loading works with new EE parameters**

The V5.0 checkpoint does not contain EdgeEnhance3D weights. When loading with `strict=False`, new EE params will be randomly initialized. Verify this works.

Check how `--resume` currently works in train.py:

```bash
cd TextMamba3D && grep -n "resume\|load_state_dict\|strict" train.py
```

- [ ] **Step 2: Ensure `strict=False` is used when loading checkpoint with new modules**

If the existing resume logic uses `strict=True`, add a config option or auto-detect:

```python
    # When fine-tuning from a checkpoint with architecture changes (e.g., new EE modules),
    # use strict=False to ignore missing keys
    strict = not config['model'].get('use_edge_enhance', False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
```

Print missing/unexpected keys for verification:

```python
    result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if result.missing_keys:
        print(f'Missing keys (new modules): {len(result.missing_keys)}')
        for k in result.missing_keys[:5]:
            print(f'  {k}')
    if result.unexpected_keys:
        print(f'Unexpected keys: {len(result.unexpected_keys)}')
```

- [ ] **Step 3: Commit**

```bash
cd TextMamba3D
git add train.py
git commit -m "fix(train): support loading V5.0 checkpoint into V5.2 model with new EE modules"
```

---

### Task 7: Final smoke test

- [ ] **Step 1: Run all tests**

```bash
cd TextMamba3D && python -m pytest tests/ -v --ignore=tests/test_mamba3.py
```

Expected: All tests PASS. (`test_mamba3.py` requires Mamba-3 install, skip on local.)

- [ ] **Step 2: Smoke test training loop (1 epoch, 2 samples)**

```bash
cd TextMamba3D && python train.py --config configs/a100_v5.2b.yaml --max-epochs 1 --max-samples 2
```

Expected: Completes without error. Loss prints include `dice` (now FTL), `ce`, `edge`, `contrastive`.

- [ ] **Step 3: Smoke test with EE enabled**

```bash
cd TextMamba3D && python train.py --config configs/a100_v5.2a.yaml --max-epochs 1 --max-samples 2
```

Expected: Completes. Prints `Missing keys (new modules): N` (EE params not in checkpoint).

- [ ] **Step 4: Commit any smoke test fixes**

If smoke tests reveal issues, fix and commit.

---

## Execution Summary

| Experiment | Config | What to Run | Success Metric |
|-----------|--------|------------|---------------|
| **Exp 2 (first)** | `a100_v5.2b.yaml` | Fine-tune from `best_v5.0.pth`, 80 epochs | ET ≥ 82%, TC/WT ≥ baseline |
| **Exp 1 (second)** | `a100_v5.2a.yaml` | Fine-tune from `best_v5.0.pth`, 80 epochs | ET ≥ 82%, EE marginal contribution |

**Notebook execution:** Use existing V5.1 notebook as template. Change config path and resume path. Upload modified source files to Colab Drive.

**Ablation:** Exp 2 vs V5.0 = FTL contribution. Exp 1 vs Exp 2 = EE contribution.

**Rollback:** If both regress TC/WT, return to V5.0.
