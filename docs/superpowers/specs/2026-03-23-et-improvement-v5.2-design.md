# TextMamba3D V5.2: ET Segmentation Improvement Design

**Date:** 2026-03-23
**Status:** Approved
**Goal:** Improve ET Dice from 79.10% to ≥82.0% through loss function replacement and learnable boundary refinement

## Context

TextMamba3D V5.0 achieves 84.79% mean Dice on BraTS2020, but ET (79.10%) lags behind TC (85.60%) and WT (89.67%). Literature review identified two orthogonal improvement vectors: asymmetric loss functions for class imbalance and learnable edge enhancement for boundary precision.

**Constraints:**
- Single A100 40GB, max 2 experiments
- V5.0 checkpoint available for fine-tuning
- Must not regress TC/WT performance

## Design

### 1. Focal Tversky Loss (FTL)

Replaces `DiceLoss` in `CombinedLoss`. Standard Dice treats FP and FN symmetrically; FTL introduces asymmetric weighting to penalize missed ET regions (FN) more heavily than over-segmentation (FP).

```
TI(p, g) = sum(p*g) / (sum(p*g) + α*sum(p*(1-g)) + β*sum((1-p)*g))
FTL = (1 - TI)^γ
```

**Hyperparameters:** α=0.3, β=0.7, γ=4/3 (≈1.33, original paper recommendation)

All three hyperparameters exposed in YAML config (not hardcoded), enabling future sweeps:
```yaml
loss:
  ftl_alpha: 0.3
  ftl_beta: 0.7
  ftl_gamma: 1.33
```

FTL is computed per-class (excluding background, matching existing `include_background=False`) and weighted by class weights `[3.0, 1.0, 4.0]` (classes 1-3). A smoothing term `smooth=1.0` guards against zero-division when a class is absent from both prediction and target. CE, Edge Loss, and Contrastive Loss remain unchanged. Deep supervision auxiliary losses also use FTL with the same α/β/γ.

**Note on compounded emphasis:** ET class weight 4.0 × β=0.7 FN penalty creates strong recall pressure. Monitor per-class precision/recall during training to detect ET over-prediction.

**File:** `losses/focal_tversky_loss.py` (new) + update `losses/__init__.py`

### 2. Edge Enhancement (EE) Module

Lightweight learnable boundary attention inserted in decoder skip connections at Stages 1, 2, 3.

```
skip_feat [B, N, C]
  → reshape to [B, C, D, H, W]
  → DepthwiseConv3d(3x3x3) → PointwiseConv3d(1x1x1) → Sigmoid
  → edge_attention map
  → output = skip_feat * edge_attention + skip_feat (residual)
  → flatten to [B, N, C]
```

- ~42K params per stage, ~126K total (negligible vs model size)
- **Initialization for near-identity:** Sigmoid bias initialized to **-3** (sigmoid(-3)≈0.047), so initial output ≈ `0.047*x + x = 1.047x`, preserving V5.0 checkpoint feature magnitudes
- Stage 0 excluded (128³ resolution, memory risk)
- Complements existing Sobel-based Edge Loss: loss provides training signal, EE provides representational capacity

**File:** New `models/edge_enhance.py` + modify `models/decoder_3d.py`

### 3. Experiment Design

| | Exp 1 (FTL+EE) | Exp 2 (FTL only) |
|---|---|---|
| Loss | FocalTverskyLoss + CE + Edge + Contrastive | FocalTverskyLoss + CE + Edge + Contrastive |
| EdgeEnhance | Stage 1, 2, 3 skip connections | None |
| Config | `configs/a100_v5.2a.yaml` | `configs/a100_v5.2b.yaml` |

**Shared training config (fine-tuning from best_v5.0.pth):**
- Epochs: 80, LR: 5e-5, Warmup: 5, Early stopping patience: 20
- All other params identical to V5.0

**Execution order:** Exp 2 first (smaller change, validates FTL direction), then Exp 1.

**Evaluation:** Full 8-config matrix (text/notext × raw/PP/TTA/TTA+PP)

### 4. Success Criteria

| Metric | Target | V5.0 Baseline |
|--------|--------|---------------|
| ET Dice (text+TTA+PP) | ≥ 82.0% | 79.10% |
| ET HD95 (text+TTA) | ≤ 3.0mm | 3.26mm |
| TC Dice | ≥ 85.60% | 85.60% |
| WT Dice | ≥ 89.67% | 89.67% |

### 5. Execution Environment

- **Runtime:** Google Colab A100 40GB via VS Code Colab plugin
- **Entry point:** Jupyter notebook (not CLI)
- **Code sync:** Local edit → Colab runtime execution
- **Checkpoints/data:** Google Drive mounted at `/content/drive/MyDrive/TextMamba3D/`
- **V5.0 checkpoint:** `checkpoints/best_v5.0.pth` on Drive

### 6. Ablation Analysis

- Exp 2 vs V5.0 → FTL independent contribution
- Exp 1 vs Exp 2 → EE module marginal contribution
- **Limitation:** No EE-only arm (without FTL). Positive Exp 1 vs Exp 2 result shows EE helps *with* FTL, not necessarily with standard Dice.
- Success criteria: **either** Exp 1 or Exp 2 reaching target suffices.

### 7. Rollback Plan

If both experiments regress TC/WT beyond tolerance, abandon V5.2 and return to V5.0 baseline. Investigate whether FTL α/β/γ need per-class tuning before concluding the approach is unviable.
