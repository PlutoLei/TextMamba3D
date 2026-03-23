# TextMamba3D Code Review Fix Plan

## Phase 1: HIGH Priority Fixes

### 1.1 FocalTverskyLoss weight re-normalization bug
- **File:** `losses/focal_tversky_loss.py`
- **Issue:** Empty classes cause dynamic weight re-normalization per batch
- **Fix:** Use fixed total weight sum (all configured weights), not just present classes

### 1.2 fusion.py double LayerNorm
- **File:** `models/fusion.py`
- **Fix:** Cache `self.t2i_norm_kv(x)` result, reuse for k and v

### 1.3 train.py misleading SSM backend print
- **File:** `train.py`
- **Fix:** Import `MAMBA3_IS_REAL` from mamba_block, print correct backend name

### 1.4 Missing use_edge_enhance in inference/eval
- **Files:** `inference.py`, `evaluate_full.py`
- **Fix:** Read `use_edge_enhance` from config and pass to TextMamba3D

### 1.5 EdgeEnhance position in decoder
- **File:** `models/decoder_3d.py`
- **Fix:** Apply EdgeEnhance to skip BEFORE adding to decoder features

### 1.6 Mamba2 + gradient_checkpointing guard
- **File:** `train.py`
- **Fix:** Runtime warning + auto-disable if both are true

### 1.7 use_mamba3 naming clarity
- **File:** `train.py`, `mamba_block.py`
- **Fix:** Add runtime log showing actual backend (not rename — too many downstream refs)

## Phase 2: MEDIUM Priority Efficiency

### 2.1 CrossScanBiMamba3DBlock rearrange optimization
- **File:** `models/mamba_block.py`
- **Fix:** Replace einops with torch.reshape+permute

### 2.2 FocalTverskyLoss vectorization
- **File:** `losses/focal_tversky_loss.py`
- **Fix:** Replace Python for-loop with vectorized tensor ops

### 2.3 EdgeLoss Sobel fusion
- **File:** `losses/edge_loss.py`
- **Fix:** Merge 3 conv3d into 1

## Status
- [ ] Phase 1 complete
- [ ] Phase 2 complete
- [ ] Tests pass
- [ ] Pushed to GitHub
