# V7.0 Boundary Loss + Hierarchy Loss Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Boundary Loss (signed distance transform) and Hierarchy Loss (ET⊆TC⊆WT) to TextMamba3D, fine-tune from V5.0 with alpha annealing.

**Architecture:** Boundary Loss computed online from cropped mask in `__getitem__` (5ms, negligible). CombinedLoss accepts `distance_map` and `epoch`, computes linear alpha annealing. Hierarchy Loss as fixed-weight regularizer. Precompute script provided for validation/inference.

**Tech Stack:** PyTorch, scipy.ndimage.distance_transform_edt, nibabel, pyyaml

**Spec:** `docs/specs/2026-03-30-v7-boundary-hierarchy-design.md`

---

## File Structure

```
losses/
├── boundary_loss.py          — EXISTS (69 lines). BoundaryLoss + compute_distance_map
├── hierarchy_loss.py         — EXISTS (40 lines). HierarchyLoss
├── __init__.py               — MODIFY: add boundary + hierarchy to CombinedLoss
data/
├── brats_textbrats_dataset.py — MODIFY: compute distance_map in __getitem__
train.py                      — MODIFY: pass distance_map + epoch to criterion
scripts/
├── precompute_distance_maps.py — CREATE: precompute .npy for validation
configs/autoresearch/
├── V7.0_boundary_hierarchy.yaml — CREATE: V7.0 config
tests/
├── test_boundary_loss.py     — CREATE
├── test_hierarchy_loss.py    — CREATE
├── test_combined_loss_v7.py  — CREATE
TextMamba3D_V7.0.ipynb        — CREATE: training notebook
```

---

## Chunk 1: Loss Module Tests

### Task 1: Test boundary_loss.py

**Files:**
- Test: `tests/test_boundary_loss.py`
- Existing: `losses/boundary_loss.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_boundary_loss.py
import numpy as np
import torch
from losses.boundary_loss import compute_distance_map, BoundaryLoss


def test_compute_distance_map_shape():
    mask = np.zeros((16, 16, 16), dtype=np.int64)
    mask[4:12, 4:12, 4:12] = 1  # edema
    mask[6:10, 6:10, 6:10] = 3  # ET
    dist = compute_distance_map(mask, num_classes=4)
    assert dist.shape == (4, 16, 16, 16)
    assert dist.dtype == np.float32


def test_compute_distance_map_signs():
    """Inside region should be negative, outside positive."""
    mask = np.zeros((16, 16, 16), dtype=np.int64)
    mask[4:12, 4:12, 4:12] = 1
    dist = compute_distance_map(mask, num_classes=4)
    # Class 1: inside region should be negative
    assert dist[1, 8, 8, 8] < 0  # center of region
    assert dist[1, 0, 0, 0] > 0  # outside region
    # Class 0 (background): outside tumor should be negative (inside bg)
    assert dist[0, 0, 0, 0] < 0  # inside background
    assert dist[0, 8, 8, 8] > 0  # outside background (inside tumor)


def test_compute_distance_map_empty_class():
    """Class not present should have large positive distances."""
    mask = np.zeros((16, 16, 16), dtype=np.int64)
    dist = compute_distance_map(mask, num_classes=4)
    # Class 1,2,3 not present — all positive (far from boundary)
    assert dist[1].min() > 0
    assert dist[2].min() > 0
    assert dist[3].min() > 0


def test_boundary_loss_forward():
    pred = torch.softmax(torch.randn(2, 4, 8, 8, 8), dim=1)
    dist = torch.randn(2, 4, 8, 8, 8)
    loss_fn = BoundaryLoss(include_background=False)
    loss = loss_fn(pred, dist)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_boundary_loss_perfect_prediction():
    """Perfect prediction should give lower loss than random."""
    mask = np.zeros((8, 8, 8), dtype=np.int64)
    mask[2:6, 2:6, 2:6] = 1
    dist = compute_distance_map(mask, num_classes=4)
    dist_t = torch.from_numpy(dist).unsqueeze(0)

    # Perfect prediction
    perfect = torch.zeros(1, 4, 8, 8, 8)
    perfect[0, 0] = torch.from_numpy((mask == 0).astype(np.float32))
    perfect[0, 1] = torch.from_numpy((mask == 1).astype(np.float32))

    # Random prediction
    random_pred = torch.softmax(torch.randn(1, 4, 8, 8, 8), dim=1)

    loss_fn = BoundaryLoss(include_background=False)
    loss_perfect = loss_fn(perfect, dist_t)
    loss_random = loss_fn(random_pred, dist_t)
    assert loss_perfect < loss_random
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/leiyuxuan/Documents/PyCharm_Project/TextMamba3D && python -m pytest tests/test_boundary_loss.py -v`
Expected: PASS (5 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/test_boundary_loss.py
git commit -m "test: boundary loss unit tests"
```

### Task 2: Test hierarchy_loss.py

**Files:**
- Test: `tests/test_hierarchy_loss.py`
- Existing: `losses/hierarchy_loss.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_hierarchy_loss.py
import torch
from losses.hierarchy_loss import HierarchyLoss


def test_hierarchy_no_violation():
    """Valid hierarchy (ET < TC < WT) should give ~0 loss."""
    loss_fn = HierarchyLoss()
    # Create logits where class 2 (edema) dominates, then class 1, then class 3
    # This gives WT > TC > ET naturally
    pred = torch.zeros(1, 4, 4, 4, 4)
    pred[0, 0] = 0.1   # background
    pred[0, 2] = 2.0   # edema (big)
    pred[0, 1] = 1.0   # necrotic (medium)
    pred[0, 3] = 0.5   # ET (small)
    loss = loss_fn(pred)
    assert loss.item() < 0.01  # near zero


def test_hierarchy_with_violation():
    """Violation (ET > TC possible) should give positive loss."""
    loss_fn = HierarchyLoss()
    pred = torch.zeros(1, 4, 4, 4, 4)
    pred[0, 0] = 0.1
    pred[0, 1] = 0.1   # necrotic very low
    pred[0, 2] = 0.1   # edema very low
    pred[0, 3] = 5.0   # ET very high — violates ET <= TC
    loss = loss_fn(pred)
    assert loss.item() > 0.01  # should be positive


def test_hierarchy_grad_flows():
    """Loss should be differentiable."""
    loss_fn = HierarchyLoss()
    pred = torch.randn(2, 4, 4, 4, 4, requires_grad=True)
    loss = loss_fn(pred)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/leiyuxuan/Documents/PyCharm_Project/TextMamba3D && python -m pytest tests/test_hierarchy_loss.py -v`
Expected: PASS (3 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/test_hierarchy_loss.py
git commit -m "test: hierarchy loss unit tests"
```

---

## Chunk 2: CombinedLoss Integration

### Task 3: Add boundary + hierarchy to CombinedLoss

**Files:**
- Modify: `losses/__init__.py`
- Test: `tests/test_combined_loss_v7.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_combined_loss_v7.py
import torch
import numpy as np
from losses import CombinedLoss
from losses.boundary_loss import compute_distance_map


def test_combined_loss_with_boundary():
    criterion = CombinedLoss(
        use_boundary=True,
        boundary_max_weight=1.0,
        use_hierarchy=True,
        hierarchy_weight=0.1,
    )
    pred = torch.randn(2, 4, 8, 8, 8)
    target = torch.randint(0, 4, (2, 8, 8, 8))

    # Compute distance maps
    dist_maps = []
    for i in range(2):
        dm = compute_distance_map(target[i].numpy(), num_classes=4)
        dist_maps.append(torch.from_numpy(dm))
    dist_maps = torch.stack(dist_maps)

    result = criterion(pred, target, distance_map=dist_maps, epoch=50, total_epochs=80)
    assert 'total' in result
    assert 'boundary' in result
    assert 'hierarchy' in result
    assert torch.isfinite(result['total'])


def test_combined_loss_alpha_annealing():
    criterion = CombinedLoss(
        use_boundary=True,
        boundary_max_weight=1.0,
    )
    pred = torch.randn(1, 4, 8, 8, 8)
    target = torch.randint(0, 4, (1, 8, 8, 8))
    dm = compute_distance_map(target[0].numpy(), num_classes=4)
    dist_maps = torch.from_numpy(dm).unsqueeze(0)

    # Epoch 0: alpha=0, boundary should be 0
    r0 = criterion(pred, target, distance_map=dist_maps, epoch=0, total_epochs=80)
    # Epoch 80: alpha=1.0, boundary should be nonzero
    r80 = criterion(pred, target, distance_map=dist_maps, epoch=80, total_epochs=80)
    assert abs(r0.get('boundary', torch.tensor(0.0)).item()) < 1e-6
    assert abs(r80['boundary'].item()) > 1e-6


def test_combined_loss_without_boundary():
    """Backward compat: no boundary/hierarchy when disabled."""
    criterion = CombinedLoss()
    pred = torch.randn(2, 4, 8, 8, 8)
    target = torch.randint(0, 4, (2, 8, 8, 8))
    result = criterion(pred, target)
    assert 'total' in result
    assert 'boundary' not in result
    assert 'hierarchy' not in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_combined_loss_v7.py -v`
Expected: FAIL (CombinedLoss doesn't accept boundary params yet)

- [ ] **Step 3: Modify losses/__init__.py**

Add to `CombinedLoss.__init__` parameters:
```python
    # V7.0: Boundary + Hierarchy Loss
    use_boundary: bool = False,
    boundary_max_weight: float = 1.0,
    use_hierarchy: bool = False,
    hierarchy_weight: float = 0.1,
```

Add to `__init__` body:
```python
    self.use_boundary = use_boundary
    self.boundary_max_weight = boundary_max_weight
    if use_boundary:
        from losses.boundary_loss import BoundaryLoss
        self.boundary_loss = BoundaryLoss(include_background=False)

    self.use_hierarchy = use_hierarchy
    self.hierarchy_weight = hierarchy_weight
    if use_hierarchy:
        from losses.hierarchy_loss import HierarchyLoss
        self.hierarchy_loss = HierarchyLoss()
```

Add `distance_map`, `epoch`, `total_epochs` to `forward` signature:
```python
def forward(
    self, pred, target,
    img_feat=None, text_feat=None, pixel_feat=None,
    mask=None, mask_orig=None, aux_preds=None,
    distance_map=None, epoch=0, total_epochs=80,
):
```

Add to `forward` body, before `return scores`:
```python
    # V7.0: Boundary Loss with linear alpha annealing
    if self.use_boundary and distance_map is not None:
        alpha = min(1.0, epoch / max(1, total_epochs)) * self.boundary_max_weight
        pred_soft = torch.softmax(pred.float(), dim=1)
        bl = self.boundary_loss(pred_soft, distance_map.float())
        scores['boundary'] = bl * alpha
        total = total + scores['boundary']

    # V7.0: Hierarchy Loss (fixed weight)
    if self.use_hierarchy:
        hl = self.hierarchy_loss(pred.float())
        scores['hierarchy'] = hl * self.hierarchy_weight
        total = total + scores['hierarchy']
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_combined_loss_v7.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run all existing loss tests to verify no regression**

Run: `python -m pytest tests/ -v -k "loss or Loss"`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add losses/__init__.py tests/test_combined_loss_v7.py
git commit -m "feat(v7): integrate boundary + hierarchy loss into CombinedLoss"
```

---

## Chunk 3: Dataset + Train Pipeline

### Task 4: Dataset computes distance_map in __getitem__

**Files:**
- Modify: `data/brats_textbrats_dataset.py`

- [ ] **Step 1: Add distance_map computation to __getitem__**

After the transform is applied (line ~233), add:

```python
# V7.0: compute distance map from cropped mask for boundary loss
from losses.boundary_loss import compute_distance_map
dist_map = compute_distance_map(mask.numpy(), num_classes=4)
result['distance_map'] = torch.from_numpy(dist_map)
```

This computes the distance map AFTER cropping, which is correct for patch-based training and avoids modifying any transform code.

- [ ] **Step 2: Verify dataset returns distance_map**

Run quick test in Python:
```bash
cd /Users/leiyuxuan/Documents/PyCharm_Project/TextMamba3D && python -c "
from data.brats_textbrats_dataset import TextBraTSDataset
ds = TextBraTSDataset('./data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData', split='train')
sample = ds[0]
print('distance_map' in sample, sample['distance_map'].shape if 'distance_map' in sample else 'missing')
"
```
Expected: `True torch.Size([4, 128, 128, 128])`

- [ ] **Step 3: Commit**

```bash
git add data/brats_textbrats_dataset.py
git commit -m "feat(v7): compute distance_map in dataset __getitem__"
```

### Task 5: train.py passes distance_map and epoch to criterion

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Modify train_epoch to pass distance_map**

At the criterion call (line ~142), add `distance_map` and `epoch`:

```python
# Extract distance_map from batch (V7.0)
distance_map = batch.get('distance_map', None)
if distance_map is not None:
    distance_map = distance_map.to(device)

loss = criterion(
    pred,
    mask,
    img_feat,
    text_feat,
    pixel_feat=pixel_feat,
    mask_orig=mask,
    aux_preds=aux_preds,
    distance_map=distance_map,
    epoch=epoch,
    total_epochs=config.get('training', {}).get('epochs', 200) if isinstance(config, dict) else 200,
)['total']
```

Note: `train_epoch` receives `epoch` as a parameter already. For `total_epochs`, read from config or pass as argument.

- [ ] **Step 2: Add total_epochs parameter to train_epoch**

Add `total_epochs=200` to `train_epoch` signature, and update the caller in `main()` to pass `config['training']['epochs']`.

- [ ] **Step 3: Verify train.py compiles**

Run: `python -m py_compile train.py && echo OK`
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add train.py
git commit -m "feat(v7): pass distance_map and epoch to criterion in training loop"
```

---

## Chunk 4: Config + Precompute Script + Notebook

### Task 6: Precompute distance maps script

**Files:**
- Create: `scripts/precompute_distance_maps.py`

- [ ] **Step 1: Write script**

```python
#!/usr/bin/env python3
"""Precompute signed distance maps for all BraTS cases.

Usage: python scripts/precompute_distance_maps.py --data-dir <path>

Saves {case_name}_distance_map.npy in each case directory.
Used for validation/inference (training computes online from cropped mask).
"""
import argparse
import os
import numpy as np
import nibabel as nib
from losses.boundary_loss import compute_distance_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--num-classes', type=int, default=4)
    args = parser.parse_args()

    cases = sorted(d for d in os.listdir(args.data_dir)
                   if os.path.isdir(os.path.join(args.data_dir, d))
                   and d.startswith('BraTS'))
    print(f'Found {len(cases)} cases')

    for i, case in enumerate(cases):
        case_dir = os.path.join(args.data_dir, case)
        seg_path = os.path.join(case_dir, f'{case}_seg.nii')
        if not os.path.exists(seg_path):
            seg_path += '.gz'
        if not os.path.exists(seg_path):
            print(f'  SKIP {case}: no seg file')
            continue

        out_path = os.path.join(case_dir, f'{case}_distance_map.npy')
        if os.path.exists(out_path):
            continue

        mask = nib.load(seg_path).get_fdata().astype(np.int64)
        mask[mask == 4] = 3  # BraTS label remap

        dist = compute_distance_map(mask, num_classes=args.num_classes)
        np.save(out_path, dist)

        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(cases)} done')

    print(f'Done: {len(cases)} cases')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/precompute_distance_maps.py
git commit -m "feat(v7): precompute distance maps script"
```

### Task 7: V7.0 Config

**Files:**
- Create: `configs/autoresearch/V7.0_boundary_hierarchy.yaml`

- [ ] **Step 1: Generate config from V5.0 base**

```python
from autoresearch.config_generator import generate_l1_config, save_config
cfg = generate_l1_config(
    base_config='configs/archive/textbrats_a100_v5.yaml',
    overrides={
        'loss': {
            'use_boundary': True,
            'boundary_max_weight': 1.0,
            'use_hierarchy': True,
            'hierarchy_weight': 0.1,
        },
        'training': {
            'lr': 0.0001,
            'epochs': 190,
            'patience': 40,
            'gradient_checkpointing': False,
        },
        'experiment': {
            'name': 'V7.0_boundary_hierarchy',
            'description': 'Boundary Loss (linear annealing) + Hierarchy Loss, resume from V5.0',
        },
    },
    experiment_name='V7.0_boundary_hierarchy',
)
save_config(cfg, 'configs/autoresearch/V7.0_boundary_hierarchy.yaml')
```

- [ ] **Step 2: Commit**

```bash
git add configs/autoresearch/V7.0_boundary_hierarchy.yaml
git commit -m "feat(v7): V7.0 config with boundary + hierarchy loss"
```

### Task 8: V7.0 Notebook

**Files:**
- Create: `TextMamba3D_V7.0.ipynb`

Notebook structure (8 cells):
1. Setup (mount Drive, install deps, clone/pull, extract data)
2. Precompute distance maps (run script on BraTS data)
3. Smoke test (2 samples, 1 epoch, verify loss components)
4. Training (resume from V5.0, reset optimizer, 80 epochs)
5. Evaluation: text+TTA
6. Evaluation: notext+TTA
7. Results comparison vs V5.0 baseline

The smoke test cell must verify:
- `boundary` key appears in loss dict
- `hierarchy` key appears in loss dict
- Alpha increases with epoch

Training cell uses `!python -u train.py` (not subprocess) for visible output, with:
```
--config configs/autoresearch/V7.0_boundary_hierarchy.yaml
--resume best_v5.0.pth
--reset-optimizer
--reset-lr
--no-text-ratio 0.15
--grad-accum 2
```

- [ ] **Step 1: Generate notebook using notebook_writer**
- [ ] **Step 2: Verify notebook is valid JSON**
- [ ] **Step 3: Commit**

```bash
git add TextMamba3D_V7.0.ipynb
git commit -m "feat(v7): V7.0 training notebook"
```

- [ ] **Step 4: Push all to GitHub**

```bash
git push origin main
```

---

## Execution Summary

| Task | Files | Tests | Estimated Time |
|------|-------|-------|----------------|
| 1 | test_boundary_loss.py | 5 tests | 5 min |
| 2 | test_hierarchy_loss.py | 3 tests | 3 min |
| 3 | losses/__init__.py + test | 3 tests | 10 min |
| 4 | brats_textbrats_dataset.py | manual verify | 5 min |
| 5 | train.py | compile check | 5 min |
| 6 | precompute script | N/A | 3 min |
| 7 | config yaml | N/A | 2 min |
| 8 | notebook | N/A | 10 min |

**Total:** 8 tasks, 11 tests, ~43 min implementation
