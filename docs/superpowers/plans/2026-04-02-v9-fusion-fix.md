# V9.0 Fusion Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make text guidance effective on TextMamba3D's Mamba2 backbone by solving two root causes: modality laziness (strong backbone ignores text gradients) and cross-attention/Mamba feature incompatibility.

**Architecture:** Incremental 5-task approach. Tasks 1-4 are low-cost fixes applied to the existing SeqCA fusion on train.py and textmamba3d.py. Task 5 replaces SeqCA with concatenate+scan (MFuser-style) only if Tasks 1-4 are insufficient. Each task is independently testable via a V9.x training run measuring text delta.

**Tech Stack:** PyTorch, mamba-ssm (Mamba2), PubMedBERT, BraTS2020+TextBraTS dataset

**Key Metrics:**
- V8.0 baseline: Mean=0.8753, text delta=0.00%
- V5.0 reference: Mean=0.8479, text delta=+0.55%
- TextBraTS SOTA: Mean=0.853, text delta=+1.5% (SwinUNETR)
- Success: text delta > +0.5% on V8.0 backbone

**Red Lines (from 5 prior failed iterations):**
1. No causal SSM fusion (text→image only)
2. Avoid shallow layer injection (Stage 0-1)
3. No pseudo text embeddings
4. Multi-scale fusion required
5. Additive > Multiplicative
6. One variable per experiment
7. Text=Query > Image=Query

---

### Task 1: Vision Backbone Freeze Warmup

**Problem:** Strong pretrained backbone dominates gradient flow from epoch 0. Text branch never receives enough gradient signal to become useful (DGL, ICCV 2025).

**Solution:** Freeze `img_encoder` for the first N epochs of Stage 2, letting only text encoder + fusion layers + decoder train. This gives text a head start.

**Files:**
- Modify: `train.py:576-598` (training loop)
- Create: `configs/autoresearch/V9.0_freeze_warmup.yaml` (copy of V8.0_stage2 + new params)

- [ ] **Step 1: Write failing test**

Create `tests/test_freeze_warmup.py`:

```python
import torch
from models.textmamba3d import TextMamba3D

def test_freeze_unfreeze_img_encoder():
    model = TextMamba3D(img_size=(32,32,32), embed_dim=48, depths=[2,2,2,2],
                        text_embed_dim=256, use_pretrained_text=False)
    # Freeze
    for p in model.img_encoder.parameters():
        p.requires_grad = False
    frozen_count = sum(1 for p in model.img_encoder.parameters() if p.requires_grad)
    assert frozen_count == 0

    # Unfreeze
    for p in model.img_encoder.parameters():
        p.requires_grad = True
    unfrozen_count = sum(1 for p in model.img_encoder.parameters() if p.requires_grad)
    assert unfrozen_count > 0

    # Fusion + decoder should always be trainable
    fusion_trainable = sum(1 for p in model.multi_scale_attn.parameters() if p.requires_grad)
    assert fusion_trainable > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/leiyuxuan/Documents/PyCharm_Project/TextMamba3D && python -m pytest tests/test_freeze_warmup.py -v`
Expected: PASS (this is a sanity test for the freeze mechanism, should pass with existing code)

- [ ] **Step 3: Add freeze warmup logic to train.py**

In `train.py`, add CLI arg and freeze logic:

```python
# After line 43 (existing argparse):
parser.add_argument('--freeze-vision-epochs', type=int, default=0,
                    help='Freeze img_encoder for first N epochs (text warmup)')
```

```python
# At optimizer creation (replace the existing AdamW call around line 515):
# Separate param groups: img_encoder (frozen initially) vs rest
if args.freeze_vision_epochs > 0:
    img_params = list(model.img_encoder.parameters())
    img_param_ids = {id(p) for p in img_params}
    other_params = [p for p in model.parameters() if id(p) not in img_param_ids]
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': base_lr},
        {'params': img_params, 'lr': 0.0},  # frozen: LR=0
    ], weight_decay=config['training']['weight_decay'])
    print(f'img_encoder frozen for first {args.freeze_vision_epochs} epochs (LR=0)')
else:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr,
        weight_decay=config['training']['weight_decay'],
    )
```

```python
# In training loop, before train_epoch call (around line 597):
if args.freeze_vision_epochs > 0 and epoch == args.freeze_vision_epochs:
    optimizer.param_groups[1]['lr'] = current_lr * 0.1
    print(f'Unfroze img_encoder at epoch {epoch} (lr={current_lr*0.1:.2e})')
```

- [ ] **Step 4: Create V9.0 config**

Copy `configs/autoresearch/V8.0_stage2_finetune.yaml` to `configs/autoresearch/V9.0_freeze_warmup.yaml`. Update `experiment.name` to `V9.0_freeze_warmup`.

- [ ] **Step 5: Run test to verify**

Run: `python -m pytest tests/test_freeze_warmup.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add train.py tests/test_freeze_warmup.py configs/autoresearch/V9.0_freeze_warmup.yaml
git commit -m "feat(v9): add --freeze-vision-epochs for text warmup"
```

---

### Task 2: Non-Zero Fusion Initialization

**Problem:** `SequentialCrossAttention.i2t_out` uses zero-init (fusion.py:219-220). With a strong backbone, the fusion path starts contributing exactly zero, and gradient pressure is insufficient to change this.

**Solution:** Replace zero-init with small-magnitude Xavier init, so the fusion path has non-zero gradient flow from the start.

**Files:**
- Modify: `models/fusion.py:218-220` (SequentialCrossAttention.__init__)
- Test: `tests/test_fusion_init.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_fusion_init.py`:

```python
import torch
from models.fusion import SequentialCrossAttention

def test_seqca_nonzero_init():
    seqca = SequentialCrossAttention(feat_dim=96, text_dim=256, num_heads=4)
    # i2t_out should NOT be all zeros
    w = seqca.i2t_out.weight
    assert w.abs().sum() > 0, "i2t_out.weight should not be zero-initialized"
    # But should be small magnitude (< 0.1 per element on average)
    assert w.abs().mean() < 0.1, "i2t_out.weight should be small magnitude"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fusion_init.py::test_seqca_nonzero_init -v`
Expected: FAIL — `i2t_out.weight` is currently all zeros

- [ ] **Step 3: Replace zero-init with small Xavier**

In `models/fusion.py`, replace lines 218-220:

```python
        # Small-magnitude init: non-zero gradient flow but near-identity start
        # Scale factor 0.01 keeps initial fusion contribution small
        nn.init.xavier_uniform_(self.i2t_out.weight)
        self.i2t_out.weight.data *= 0.01
        nn.init.zeros_(self.i2t_out.bias)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fusion_init.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/fusion.py tests/test_fusion_init.py
git commit -m "feat(v9): non-zero init for SeqCA fusion output"
```

---

### Task 3: Modality Dropout (Vision Masking)

**Problem:** When vision features are always available, the model has no incentive to use text (MoBaNet, 2026; OGM-GE, CVPR 2022). The text path becomes a dead branch.

**Solution:** During training, randomly replace vision features with zeros for X% of samples, forcing the model to rely on text. At inference, vision is always available.

**Files:**
- Modify: `models/textmamba3d.py:144-162` (forward method)
- Modify: `train.py` (add CLI arg)
- Test: `tests/test_modality_dropout.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_modality_dropout.py`:

```python
import torch
from models.textmamba3d import TextMamba3D

def test_vision_dropout_changes_output():
    torch.manual_seed(42)
    model = TextMamba3D(img_size=(32,32,32), embed_dim=48, depths=[2,2,2,2],
                        text_embed_dim=256, use_pretrained_text=False)
    model.eval()
    img = torch.randn(1, 4, 32, 32, 32)
    text_ids = torch.randint(0, 1000, (1, 16))
    mask = torch.ones(1, 16)

    # Normal forward
    out_normal = model(img, text_ids, mask, use_text=True)

    # With vision dropout (all zeros)
    model.vision_dropout_rate = 1.0  # 100% dropout for test
    model.train()
    out_dropped = model(img, text_ids, mask, use_text=True)

    # Outputs should differ when vision is dropped
    assert not torch.allclose(out_normal, out_dropped, atol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_modality_dropout.py -v`
Expected: FAIL — `vision_dropout_rate` attribute doesn't exist yet

- [ ] **Step 3: Add vision dropout to TextMamba3D**

In `models/textmamba3d.py`, add attribute in `__init__`:

```python
        # V9.0: Vision modality dropout — force text utilization
        self.vision_dropout_rate = 0.0  # set via train.py
```

In `forward` method, after `img_features = self.img_encoder(img)` (line 145), add:

```python
        # V9.0: randomly zero out vision features to force text usage
        # Note: applies to entire batch (not per-sample) for simplicity.
        # With batch_size=2, 15% rate means ~15% of batches lose vision.
        if self.training and self.vision_dropout_rate > 0:
            if torch.rand(1).item() < self.vision_dropout_rate:
                img_features = [torch.zeros_like(f) for f in img_features]
```

In `train.py`, add CLI arg:

```python
parser.add_argument('--vision-dropout', type=float, default=0.0,
                    help='Probability of zeroing vision features (force text usage)')
```

And before the training loop:

```python
model.vision_dropout_rate = args.vision_dropout
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_modality_dropout.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/textmamba3d.py train.py tests/test_modality_dropout.py
git commit -m "feat(v9): add vision modality dropout to force text utilization"
```

---

### Task 4: Pre-Alignment Loss (InfoNCE)

**Problem:** PubMedBERT text features and Mamba2 image features live in different spaces. Cross-attention must bridge this gap, but with no explicit alignment signal.

**Solution:** Add InfoNCE contrastive loss between img_global and text_global to pull matching text-image pairs together. Infrastructure already exists — `return_features=True` outputs `img_global` [B,256] and `text_global` [B,256].

**Files:**
- Modify: `losses/__init__.py` (CombinedLoss — add alignment loss)
- Create: `losses/alignment_loss.py`
- Test: `tests/test_alignment_loss.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_alignment_loss.py`:

```python
import torch
from losses.alignment_loss import InfoNCELoss

def test_infonce_positive_pairs():
    loss_fn = InfoNCELoss(temperature=0.07)
    # Identical pairs should have low loss
    feats = torch.randn(4, 256)
    loss_same = loss_fn(feats, feats)
    # Random pairs should have higher loss
    loss_rand = loss_fn(feats, torch.randn(4, 256))
    assert loss_same < loss_rand

def test_infonce_gradient_flows():
    loss_fn = InfoNCELoss(temperature=0.07)
    img = torch.randn(4, 256, requires_grad=True)
    txt = torch.randn(4, 256, requires_grad=True)
    loss = loss_fn(img, txt)
    loss.backward()
    assert img.grad is not None
    assert txt.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_alignment_loss.py -v`
Expected: FAIL — `losses.alignment_loss` module doesn't exist

- [ ] **Step 3: Create alignment loss module**

Create `losses/alignment_loss.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for text-image alignment.

    Pulls matching text-image pairs together, pushes non-matching apart.
    Uses in-batch negatives (batch size = number of negatives).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, img_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_feat: [B, D] L2-normalized image global features
            text_feat: [B, D] L2-normalized text global features
        Returns:
            scalar loss
        """
        img_feat = F.normalize(img_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        # Similarity matrix [B, B]
        logits = img_feat @ text_feat.T / self.temperature

        # Positive pairs are on the diagonal
        labels = torch.arange(logits.size(0), device=logits.device)

        # Symmetric loss: img->text + text->img
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2
```

- [ ] **Step 4: Wire into CombinedLoss**

In `losses/__init__.py`, add to CombinedLoss:

```python
from .alignment_loss import InfoNCELoss
```

In `CombinedLoss.__init__`, add parameter `alignment_weight: float = 0.0` and:

```python
        self.alignment_loss = InfoNCELoss(temperature=temperature)
        self.alignment_weight = alignment_weight
```

In `CombinedLoss.forward`, add after existing contrastive loss block (note: use `img_feat`/`text_feat`, the existing parameter names in `forward()`):

```python
        # V9.0: Text-image alignment loss
        if self.alignment_weight > 0 and img_feat is not None and text_feat is not None:
            align_loss = self.alignment_loss(img_feat, text_feat)
            total += self.alignment_weight * align_loss
            losses['alignment'] = align_loss.item()
```

In `train.py`, wire `alignment_weight` from config to `CombinedLoss` (around line 493-512):

```python
    criterion = CombinedLoss(
        ...
        alignment_weight=config['loss'].get('alignment_weight', 0.0),
    ).to(device)
```

In V9.0 config YAML, add under `loss:`:

```yaml
  alignment_weight: 0.1
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_alignment_loss.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add losses/alignment_loss.py losses/__init__.py tests/test_alignment_loss.py
git commit -m "feat(v9): add InfoNCE alignment loss for text-image pre-alignment"
```

---

### Task 5: Replace SeqCA with Concatenate+Scan (MFuser-style)

**Condition:** Only implement if Tasks 1-4 do not achieve text delta > +0.5%.

**Problem:** Cross-attention is architecturally incompatible with Mamba features (Hidden Attention of Mamba, ACL 2025). No paper reports positive text delta from cross-attention on Mamba.

**Solution:** Replace `MultiScaleSeqCA` with `MambaConcatFusion` — concatenate compressed text prompt tokens with image tokens and process through a Mamba block. This is the fusion pattern used by MFuser (CVPR 2025 Highlight, +1.29 mIoU text delta).

**Files:**
- Create: `models/concat_fusion.py`
- Modify: `models/textmamba3d.py:88-95` (fusion selection)
- Test: `tests/test_concat_fusion.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_concat_fusion.py`:

```python
import torch
from models.concat_fusion import MambaConcatFusion, MultiScaleConcatFusion

def test_concat_fusion_shape():
    fusion = MambaConcatFusion(feat_dim=96, text_dim=256, num_prompts=8)
    img = torch.randn(2, 512, 96)   # [B, N, D]
    text = torch.randn(2, 32, 256)  # [B, M, D_text]
    out = fusion(img, text)
    assert out.shape == img.shape    # [B, N, D] — same as input

def test_concat_fusion_text_matters():
    torch.manual_seed(42)
    fusion = MambaConcatFusion(feat_dim=96, text_dim=256, num_prompts=8)
    img = torch.randn(2, 512, 96)
    text_a = torch.randn(2, 32, 256)
    text_b = torch.randn(2, 32, 256)
    out_a = fusion(img, text_a)
    out_b = fusion(img, text_b)
    # Different text should produce different outputs
    assert not torch.allclose(out_a, out_b, atol=1e-4)

def test_multiscale_concat_fusion():
    fusion = MultiScaleConcatFusion(
        stage_dims=[96, 192, 384], text_dim=256, num_prompts=8
    )
    features = [
        torch.randn(2, 4096, 96),
        torch.randn(2, 512, 192),
        torch.randn(2, 64, 384),
    ]
    text = torch.randn(2, 32, 256)
    out = fusion(features, text)
    assert len(out) == 3
    for o, f in zip(out, features):
        assert o.shape == f.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_concat_fusion.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement MambaConcatFusion**

Create `models/concat_fusion.py`:

```python
"""MFuser-style concatenate+scan fusion for Mamba backbones.

Instead of cross-attention (which doesn't work with Mamba features),
compress text into K prompt tokens, concatenate with image tokens,
and process through a Mamba block. The selective scan implicitly
performs cross-modal interaction.

Reference: MFuser (CVPR 2025 Highlight) — Mamba fusion beats
attention fusion: 68.20 vs 67.89 mIoU with 60% fewer params.
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
        """
        Args:
            text_feat: [B, M, D_text]
        Returns:
            [B, K, D] compressed prompt tokens
        """
        B = text_feat.size(0)
        kv = self.kv_proj(text_feat)  # [B, M, D]
        q = self.queries.expand(B, -1, -1)  # [B, K, D]
        # Simple attention to compress
        attn = (q @ kv.transpose(-1, -2)) * (q.size(-1) ** -0.5)
        attn = attn.softmax(dim=-1)
        prompts = attn @ kv  # [B, K, D]
        return self.norm(prompts)


class MambaConcatFusion(nn.Module):
    """Concatenate text prompts with image tokens, scan with Mamba.

    Architecture:
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
    ):
        super().__init__()
        self.compressor = TextPromptCompressor(text_dim, feat_dim, num_prompts)
        self.num_prompts = num_prompts
        self.norm = nn.LayerNorm(feat_dim)
        self.ssm = _create_ssm(
            dim=feat_dim, d_state=d_state, d_conv=4, expand=expand,
            use_mamba3=True,
        )
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        # Small init (not zero — learned from Task 2)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.weight.data *= 0.01
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        text_feat: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] image features
            text_feat: [B, M, D_text] text token features
        Returns:
            [B, N, D] text-enhanced image features
        """
        residual = x
        K = self.num_prompts

        prompts = self.compressor(text_feat)        # [B, K, D]
        x_norm = self.norm(x)
        concat = torch.cat([prompts, x_norm], dim=1)  # [B, K+N, D]
        scanned = self.ssm(concat)                     # [B, K+N, D]
        img_out = scanned[:, K:, :]                    # [B, N, D]

        return residual + self.out_proj(img_out)


class MultiScaleConcatFusion(nn.Module):
    """Apply MambaConcatFusion at multiple encoder scales."""

    def __init__(
        self,
        stage_dims: list[int],
        text_dim: int,
        num_prompts: int = 8,
        d_state: int = 16,
        **kwargs,  # accept and ignore num_heads etc for drop-in compatibility
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaConcatFusion(dim, text_dim, num_prompts, d_state)
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
```

- [ ] **Step 4: Wire into TextMamba3D**

In `models/textmamba3d.py`, add import:

```python
from .concat_fusion import MultiScaleConcatFusion
```

Modify fusion selection (line 90):

```python
        if fusion_type == "seqca":
            fusion_cls = MultiScaleSeqCA
        elif fusion_type == "concat_scan":
            fusion_cls = MultiScaleConcatFusion
        else:
            fusion_cls = MultiScalePixelTextAttention
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_concat_fusion.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add models/concat_fusion.py models/textmamba3d.py tests/test_concat_fusion.py
git commit -m "feat(v9): add MambaConcatFusion (MFuser-style concatenate+scan)"
```

---

## Experiment Plan

Each task is tested independently with a training run:

| Experiment | Config Changes | CLI Flags | Expected Outcome |
|------------|---------------|-----------|-----------------|
| V9.1 | V9.0 config | `--freeze-vision-epochs 10` | text delta > 0% |
| V9.2 | V9.1 + non-zero init | (code change, no flag) | text delta > V9.1 |
| V9.3 | V9.2 + vision dropout | `--vision-dropout 0.15` | text delta > V9.2 |
| V9.4 | V9.3 + alignment loss | `alignment_weight: 0.1` in config | text delta > V9.3 |
| V9.5 | V9.4 + concat_scan | `fusion_type: concat_scan` | text delta > +0.5% |

**Notebook update:** After each task is implemented, update `TextMamba3D_V8.0.ipynb` (or create V9.0 notebook) with the new config and CLI flags.

## Rollback Strategy

Each task is independently revertable:
- Task 1: `--freeze-vision-epochs 0` (default)
- Task 2: Revert fusion.py init (one line)
- Task 3: `--vision-dropout 0.0` (default)
- Task 4: `alignment_weight: 0.0` in config
- Task 5: `fusion_type: seqca` in config
