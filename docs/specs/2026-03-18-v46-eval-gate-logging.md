# V4.6 Eval Ablation + Gate Logging Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add TextScaleGate value logging to V4.6 training, and create a comprehensive V4.6 evaluation notebook with 10-config ablation, TC regression diagnosis, and cross-version ensemble comparison.

**Architecture:** Two deliverables: (1) a patch cell in the existing H100 training notebook that injects gate logging into train.py; (2) a new eval notebook (`TextMamba3D_H100_V4.6_eval.ipynb`) containing the full ablation matrix, TC ablation, and ensemble comparison. Both follow the Colab notebook pattern where Python files are patched at runtime.

**Tech Stack:** PyTorch, MONAI-style sliding window, scipy.stats (Wilcoxon), matplotlib, NumPy

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `TextMamba3D_H100_V4.6.ipynb` | Add 1 cell: gate logging patch for train.py |
| Modify | `TextMamba3D/train.py` | Add `log_gate_values()` function + call in training loop |
| Create | `TextMamba3D_H100_V4.6_eval.ipynb` | Full eval notebook: 10-config ablation + TC ablation + ensemble comparison |

---

## Chunk 1: P0-2 — TextScaleGate Value Logging

### Task 1: Add gate logging function to train.py

**Files:**
- Modify: `TextMamba3D/train.py:466-474` (after validation, before checkpoint saving)

- [ ] **Step 1: Add `log_gate_values` helper function**

Insert after `validate()` function definition (after line 219), before `def main():` (line 222):

```python
@torch.no_grad()
def log_gate_values(model, writer, epoch):
    """Log TextScaleGate sigmoid values to TensorBoard.

    Extracts the learned gate bias from each scale's TextScaleGate,
    computes sigmoid(bias) as the "resting" gate value (when raw == fused),
    and logs per-scale values.
    """
    text_gate = getattr(model, 'text_gate', None)
    if text_gate is None:
        return

    gate_values = {}
    for i, gate in enumerate(text_gate.gates):
        # Gate bias determines default behavior: sigmoid(bias) = default text contribution
        bias = gate.gate_proj.bias.item()
        gate_val = torch.sigmoid(torch.tensor(bias)).item()
        scale_name = f'scale_{i+1}'  # scales 1,2,3 (skip scale 0 which has no text fusion)
        gate_values[scale_name] = gate_val
        writer.add_scalar(f'Gate/{scale_name}_bias', bias, epoch)
        writer.add_scalar(f'Gate/{scale_name}_sigmoid', gate_val, epoch)

    # Summary line
    vals = [f'{k}={v:.4f}' for k, v in gate_values.items()]
    print(f'  Gate values: {", ".join(vals)}')
    return gate_values
```

- [ ] **Step 2: Call `log_gate_values` in training loop**

Insert after the TensorBoard logging block (after line 474, `writer.add_scalar('Loss/contrastive_weight', ...)`):

```python
        # Log TextScaleGate values (V4.6)
        log_gate_values(model, writer, epoch)
```

- [ ] **Step 3: Verify the edit**

Run: `python -c "import ast; ast.parse(open('train.py').read()); print('OK')"` — should print OK.

- [ ] **Step 4: Commit**

```bash
git add train.py
git commit -m "feat(train): add TextScaleGate value logging to TensorBoard"
```

### Task 2: Add gate logging patch cell to H100 V4.6 notebook

**Files:**
- Modify: `TextMamba3D_H100_V4.6.ipynb` — insert new code cell after cell 9 (config + patch scripts) and before cell 10 (verify patches)

- [ ] **Step 5: Insert gate logging patch cell**

New code cell with the following content. This cell patches `train.py` to add the `log_gate_values` function and its call in the training loop. The patch uses Python string operations to inject code at the right location.

```python
# ── Cell: Patch train.py with TextScaleGate logging (V4.6 P0-2) ──
import re

train_path = 'train.py'
with open(train_path, 'r') as f:
    src = f.read()

# 1. Add log_gate_values function before def main():
gate_fn = '''
@torch.no_grad()
def log_gate_values(model, writer, epoch):
    """Log TextScaleGate sigmoid values to TensorBoard."""
    text_gate = getattr(model, 'text_gate', None)
    if text_gate is None:
        return
    gate_values = {}
    for i, gate in enumerate(text_gate.gates):
        bias = gate.gate_proj.bias.item()
        gate_val = torch.sigmoid(torch.tensor(bias)).item()
        scale_name = f'scale_{i+1}'
        gate_values[scale_name] = gate_val
        writer.add_scalar(f'Gate/{scale_name}_bias', bias, epoch)
        writer.add_scalar(f'Gate/{scale_name}_sigmoid', gate_val, epoch)
    vals = [f'{k}={v:.4f}' for k, v in gate_values.items()]
    print(f'  Gate values: {", ".join(vals)}')
    return gate_values


'''

if 'log_gate_values' not in src:
    src = src.replace('\ndef main():', gate_fn + 'def main():')
    print('[PATCH] Added log_gate_values function')

# 2. Add call in training loop after contrastive_weight logging
gate_call = "        # Log TextScaleGate values (V4.6)\n        log_gate_values(model, writer, epoch)\n"
anchor = "        writer.add_scalar('Loss/contrastive_weight', criterion.contrastive_weight, epoch)"

if 'log_gate_values(model' not in src and anchor in src:
    src = src.replace(anchor, anchor + '\n\n' + gate_call)
    print('[PATCH] Added log_gate_values call in training loop')

with open(train_path, 'w') as f:
    f.write(src)

# Verify syntax
import ast
ast.parse(src)
print('[OK] train.py syntax valid after gate logging patch')
```

- [ ] **Step 6: Verify notebook cell ordering**

After insertion, the notebook cell order should be:
- Cell 9: Config + patch scripts
- Cell 10 (NEW): Gate logging patch
- Cell 11: Verify all patches (was cell 10)

---

## Chunk 2: P0-1 — V4.6 Eval Notebook with 10-Config Ablation

### Task 3: Create eval notebook — Setup cells (cells 0-4)

**Files:**
- Create: `TextMamba3D_H100_V4.6_eval.ipynb`

- [ ] **Step 7: Cell 0 — Title and ablation matrix (markdown)**

```markdown
# TextMamba3D V4.6 — H100 Comprehensive Evaluation

## Inference Matrix

| Phase | Config | Text | TTA | PP | Ensemble | Purpose |
|-------|--------|------|-----|----|----------|---------|
| A | 1 | V4.6 text | - | - | - | Baseline |
| A | 2 | V4.6 text | - | ON | - | +PP only |
| B | 3 | V4.6 text | ON | - | - | +TTA only |
| B | 4 | V4.6 text | ON | ON | - | Best single model candidate |
| A | 5 | V4.6 no-text | - | - | - | No-text baseline |
| A | 6 | V4.6 no-text | - | ON | - | No-text +PP |
| B | 7 | V4.6 no-text | ON | - | - | No-text +TTA |
| B | 8 | V4.6 no-text | ON | ON | - | No-text +TTA+PP |
| C | 9 | Ensemble (V4.4+V4.6) | - | - | ON | Cross-version ensemble |
| C | 10 | Ensemble (V4.4+V4.6) | - | ON | ON | Ensemble +PP |

## Eval Infrastructure

- **TTA:** 8-fold spatial flip (already in `evaluate_full.py --tta`)
- **PP:** ET connected-component filter (`--postprocess --et-min-size 500`)
- **Ensemble:** Offline softmax probability averaging from `--save-preds`
- **Metrics:** Dice (ET, TC, WT, Mean) + HD95 (ET, TC, WT)
```

- [ ] **Step 8: Cell 1 — Environment setup (code)**

```python
# ── Cell 1: Environment Setup ──
import subprocess, os, sys

# Install dependencies (Colab)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'mamba-ssm', 'causal-conv1d', 'transformers',
                'nibabel', 'tensorboard', 'pyyaml', 'tqdm', 'scipy'], check=True)

# GPU check
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")
else:
    print("WARNING: No GPU detected")
```

- [ ] **Step 9: Cell 2 — Restore code and data from Drive (code)**

```python
# ── Cell 2: Restore Code & Data ──
from google.colab import drive
drive.mount('/content/drive')

DRIVE_BASE = '/content/drive/MyDrive/TextMamba3D'
DRIVE_CKPT = f'{DRIVE_BASE}/checkpoints'

# Extract code
import zipfile
code_zip = f'{DRIVE_BASE}/TextMamba3D_code.zip'
if os.path.exists(code_zip):
    with zipfile.ZipFile(code_zip, 'r') as z:
        z.extractall('/content/TextMamba3D')
    os.chdir('/content/TextMamba3D')
    print(f"Code extracted. CWD: {os.getcwd()}")

# Extract data
data_zip = f'{DRIVE_BASE}/BraTS2020_TrainingData.zip'
data_dir = './data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
if not os.path.exists(data_dir):
    with zipfile.ZipFile(data_zip, 'r') as z:
        z.extractall('./data/BraTS2020/BraTS2020_TrainingData')
cases = [d for d in os.listdir(data_dir) if d.startswith('BraTS20')]
print(f"Data: {len(cases)} cases found")
```

- [ ] **Step 10: Cell 3 — Apply V4.6 code patches (code)**

```python
# ── Cell 3: Apply V4.4 + V4.6 Code Patches ──
# (same as training notebook cells 4-8: SeqCA, RMSNorm, CrossScaleSkip, TextScaleGate, ET-enrichment)
# This cell copies the patch logic from the training notebook.
# In practice, if running from the same Drive zip that was already patched during training,
# these patches are already applied. We verify instead of re-patching.

from models.fusion import SequentialCrossAttention, MultiScaleSeqCA
print(f"[OK] SeqCA loaded: {SequentialCrossAttention}")

try:
    from models.fusion import RMSNorm, CrossScaleSkipAttention, TextScaleGate, MultiScaleTextGate
    print(f"[OK] V4.6 modules loaded: RMSNorm, CrossScaleSkipAttention, TextScaleGate, MultiScaleTextGate")
except ImportError as e:
    print(f"[ERROR] V4.6 modules not found — run training notebook patches first: {e}")
    raise
```

- [ ] **Step 11: Cell 4 — Checkpoint verification (code)**

```python
# ── Cell 4: Verify Checkpoints ──
import os

v46_ckpt = f'{DRIVE_CKPT}/best_v4.6.pth'
v45_ckpt = f'{DRIVE_CKPT}/best_v4.5.pth'
v44_ckpt = f'{DRIVE_CKPT}/best_v4.4.pth'

for name, path in [('V4.6', v46_ckpt), ('V4.5', v45_ckpt), ('V4.4', v44_ckpt)]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1024**2
        print(f"[OK] {name}: {path} ({size_mb:.1f} MB)")
    else:
        print(f"[MISSING] {name}: {path}")

# V4.6 config for evaluation
V46_CONFIG = 'configs/textbrats_v8_h100.yaml'
assert os.path.exists(V46_CONFIG), f"Config not found: {V46_CONFIG}"
print(f"[OK] Config: {V46_CONFIG}")
```

### Task 4: Eval notebook — Phase A: Baseline inference (cells 5-6)

- [ ] **Step 12: Cell 5 — Phase A: 4 baseline runs (code)**

```python
# ── Cell 5: Phase A — Baseline Inference (no TTA) ──
# Runs configs 1, 2, 5, 6 from the ablation matrix
# Saves predictions for offline ensemble in Phase C
import subprocess, os

os.makedirs('eval_preds/v46_text', exist_ok=True)
os.makedirs('eval_preds/v46_notext', exist_ok=True)

runs_a = [
    # (label, use_text, postprocess, save_dir, log_file)
    ('Config 1: V4.6 text baseline',     True,  False, 'eval_preds/v46_text',   'eval_a1.log'),
    ('Config 2: V4.6 text +PP',          True,  True,  None,                     'eval_a2.log'),
    ('Config 5: V4.6 no-text baseline',  False, False, 'eval_preds/v46_notext', 'eval_a5.log'),
    ('Config 6: V4.6 no-text +PP',       False, True,  None,                     'eval_a6.log'),
]

for label, use_text, pp, save_dir, log_file in runs_a:
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"{'='*60}")

    cmd = [
        'python', 'evaluate_full.py',
        '--config', V46_CONFIG,
        '--checkpoint', v46_ckpt,
        '--split', 'test',
    ]
    if not use_text:
        cmd.append('--no-text')
    if pp:
        cmd.extend(['--postprocess', '--et-min-size', '500'])
    if save_dir:
        cmd.extend(['--save-preds', save_dir])

    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    print(f"  -> Saved to {log_file} (exit code: {result.returncode})")
```

- [ ] **Step 13: Cell 6 — Phase B: TTA runs (code)**

```python
# ── Cell 6: Phase B — TTA Inference ──
# Runs configs 3, 4, 7, 8
runs_b = [
    ('Config 3: V4.6 text +TTA',          True,  False, 'eval_b3.log'),
    ('Config 4: V4.6 text +TTA+PP',       True,  True,  'eval_b4.log'),
    ('Config 7: V4.6 no-text +TTA',       False, False, 'eval_b7.log'),
    ('Config 8: V4.6 no-text +TTA+PP',    False, True,  'eval_b8.log'),
]

for label, use_text, pp, log_file in runs_b:
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"{'='*60}")

    cmd = [
        'python', 'evaluate_full.py',
        '--config', V46_CONFIG,
        '--checkpoint', v46_ckpt,
        '--split', 'test',
        '--tta',
    ]
    if not use_text:
        cmd.append('--no-text')
    if pp:
        cmd.extend(['--postprocess', '--et-min-size', '500'])

    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    print(f"  -> Saved to {log_file} (exit code: {result.returncode})")
```

### Task 5: Eval notebook — Phase C: Ensemble + offline ablation (cells 7-9)

- [ ] **Step 14: Cell 7 — Phase C: V4.4 baseline predictions (code)**

```python
# ── Cell 7: Phase C — V4.4 Baseline Predictions for Ensemble ──
# Generate V4.4 predictions to combine with V4.6 (and compare with V4.4+V4.5)
import subprocess, os

os.makedirs('eval_preds/v44_text', exist_ok=True)

# V4.4 config: same as V4.6 but without text_gate and cross_scale_skip
import yaml
with open(V46_CONFIG, 'r') as f:
    v44_cfg = yaml.safe_load(f)
v44_cfg['model']['use_text_gate'] = False
v44_cfg['model']['use_cross_scale_skip'] = False
v44_config_path = 'configs/ablation_v44_eval.yaml'
with open(v44_config_path, 'w') as f:
    yaml.dump(v44_cfg, f)

cmd = [
    'python', 'evaluate_full.py',
    '--config', v44_config_path,
    '--checkpoint', v44_ckpt,
    '--split', 'test',
    '--save-preds', 'eval_preds/v44_text',
]

print("Running V4.4 text baseline (for ensemble)...")
with open('eval_c_v44.log', 'w') as f:
    result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
print(f"  -> exit code: {result.returncode}")
```

- [ ] **Step 15: Cell 8 — Offline ensemble: V4.4+V4.6 (code)**

```python
# ── Cell 8: Offline Ensemble — V4.4 + V4.6 Probability Averaging ──
import numpy as np
import os
from scipy.ndimage import label as ndimage_label

def ensemble_predictions(dir_a, dir_b, output_dir, et_min_size=500, apply_pp=False):
    """Average softmax predictions from two models, optionally apply ET post-processing."""
    os.makedirs(output_dir, exist_ok=True)
    files_a = sorted([f for f in os.listdir(dir_a) if f.endswith('_pred.npy')])
    results = []

    for fname in files_a:
        case_name = fname.replace('_pred.npy', '')
        path_b = os.path.join(dir_b, fname)
        if not os.path.exists(path_b):
            print(f"  [SKIP] {case_name}: not found in {dir_b}")
            continue

        pred_a = np.load(os.path.join(dir_a, fname))
        pred_b = np.load(os.path.join(dir_b, fname))

        # If predictions are argmax (integer labels), convert to one-hot for averaging
        if pred_a.dtype in (np.int32, np.int64, np.uint8):
            num_classes = 4
            onehot_a = np.eye(num_classes)[pred_a]  # [D,H,W,C]
            onehot_b = np.eye(num_classes)[pred_b]
            avg = (onehot_a + onehot_b) / 2.0
            ensemble_pred = avg.argmax(axis=-1).astype(np.int32)
        else:
            # Already softmax probabilities
            avg = (pred_a + pred_b) / 2.0
            ensemble_pred = avg.argmax(axis=0 if avg.shape[0] == 4 else -1).astype(np.int32)

        if apply_pp:
            from evaluate_full import postprocess_et
            ensemble_pred = postprocess_et(ensemble_pred, et_class=3, min_size=et_min_size)

        np.save(os.path.join(output_dir, fname), ensemble_pred)
        results.append({'case': case_name, 'pred': ensemble_pred})

    return results

# Config 9: Ensemble V4.4+V4.6 (no PP)
print("Generating ensemble V4.4+V4.6 predictions...")
ens_results_no_pp = ensemble_predictions(
    'eval_preds/v44_text', 'eval_preds/v46_text',
    'eval_preds/ensemble_v44_v46', apply_pp=False
)
print(f"  -> {len(ens_results_no_pp)} cases")

# Config 10: Ensemble V4.4+V4.6 +PP
ens_results_pp = ensemble_predictions(
    'eval_preds/v44_text', 'eval_preds/v46_text',
    'eval_preds/ensemble_v44_v46_pp', apply_pp=True, et_min_size=500
)
print(f"  -> {len(ens_results_pp)} cases (with PP)")
```

- [ ] **Step 16: Cell 9 — Compute ensemble metrics + aggregate all 10 configs (code)**

```python
# ── Cell 9: Aggregate All 10 Configurations ──
import re
import numpy as np

def parse_eval_log(log_path):
    """Parse evaluate_full.py log output into per-case metrics."""
    results = []
    with open(log_path, 'r') as f:
        for line in f:
            # Match: "  BraTS20_Training_XXX: Dice=0.XXXX (ET=..., TC=..., WT=...) HD95_ET=..."
            m = re.match(
                r'\s+(BraTS20_\w+): Dice=([\d.]+) '
                r'\(ET=([\d.]+), TC=([\d.]+), WT=([\d.]+)\)'
                r'(?: HD95_ET=([\d.nan]+))?',
                line
            )
            if m:
                results.append({
                    'case': m.group(1),
                    'dice_mean': float(m.group(2)),
                    'dice_ET': float(m.group(3)),
                    'dice_TC': float(m.group(4)),
                    'dice_WT': float(m.group(5)),
                })
    return results

def compute_ensemble_metrics(ensemble_dir, data_dir, dataset):
    """Compute Dice metrics for ensemble predictions against ground truth."""
    from utils.metrics import dice_score_brats_regions
    import torch, nibabel as nib

    results = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        case_name = sample['case_name']
        mask = sample['mask'].numpy()  # [D, H, W]

        pred_path = os.path.join(ensemble_dir, f'{case_name}_pred.npy')
        if not os.path.exists(pred_path):
            continue
        pred = np.load(pred_path)

        # Convert to one-hot for dice_score_brats_regions
        pred_onehot = torch.zeros(1, 4, *pred.shape)
        for c in range(4):
            pred_onehot[0, c] = torch.from_numpy((pred == c).astype(np.float32))

        dice = dice_score_brats_regions(pred_onehot, torch.from_numpy(mask).unsqueeze(0))
        dice['case'] = case_name
        results.append(dice)
    return results

# Parse Phase A + B logs
config_labels = [
    ('Config 1: V4.6 text',           'eval_a1.log'),
    ('Config 2: V4.6 text +PP',       'eval_a2.log'),
    ('Config 3: V4.6 text +TTA',      'eval_b3.log'),
    ('Config 4: V4.6 text +TTA+PP',   'eval_b4.log'),
    ('Config 5: V4.6 no-text',        'eval_a5.log'),
    ('Config 6: V4.6 no-text +PP',    'eval_a6.log'),
    ('Config 7: V4.6 no-text +TTA',   'eval_b7.log'),
    ('Config 8: V4.6 no-text +TTA+PP','eval_b8.log'),
]

all_configs = {}
for label, log_file in config_labels:
    if os.path.exists(log_file):
        results = parse_eval_log(log_file)
        means = {
            'dice_ET': np.mean([r['dice_ET'] for r in results]),
            'dice_TC': np.mean([r['dice_TC'] for r in results]),
            'dice_WT': np.mean([r['dice_WT'] for r in results]),
            'dice_mean': np.mean([r['dice_mean'] for r in results]),
            'n_cases': len(results),
            'per_case': results,
        }
        all_configs[label] = means
        print(f"{label}: Mean={means['dice_mean']:.4f} (ET={means['dice_ET']:.4f}, TC={means['dice_TC']:.4f}, WT={means['dice_WT']:.4f})")
    else:
        print(f"[SKIP] {log_file} not found")

# Phase C: Ensemble metrics (requires loading dataset for ground truth)
from transformers import AutoTokenizer
from data.brats_textbrats_dataset import TextBraTSDataset
import yaml

with open(V46_CONFIG, 'r') as f:
    eval_config = yaml.safe_load(f)
model_cfg = eval_config['model']
tokenizer = AutoTokenizer.from_pretrained(
    model_cfg.get('text_model_path') or 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
)
test_dataset = TextBraTSDataset(
    data_dir=eval_config['data']['data_dir'],
    split='test', transform=None, tokenizer=tokenizer,
    max_text_len=model_cfg.get('text_max_len', 192),
    train_ratio=eval_config['data'].get('train_ratio', 0.596),
    val_ratio=eval_config['data'].get('val_ratio', 0.149),
    et_enriched=eval_config['data'].get('et_enriched', False),
    enriched_prob=1.0,  # Always use enriched text at eval
)

for ens_label, ens_dir in [
    ('Config 9: Ensemble V4.4+V4.6', 'eval_preds/ensemble_v44_v46'),
    ('Config 10: Ensemble V4.4+V4.6 +PP', 'eval_preds/ensemble_v44_v46_pp'),
]:
    if os.path.exists(ens_dir):
        ens_results = compute_ensemble_metrics(ens_dir, eval_config['data']['data_dir'], test_dataset)
        means = {
            'dice_ET': np.mean([r['dice_ET'] for r in ens_results]),
            'dice_TC': np.mean([r['dice_TC'] for r in ens_results]),
            'dice_WT': np.mean([r['dice_WT'] for r in ens_results]),
            'dice_mean': np.mean([r['dice_mean'] for r in ens_results]),
            'n_cases': len(ens_results),
            'per_case': ens_results,
        }
        all_configs[ens_label] = means
        print(f"{ens_label}: Mean={means['dice_mean']:.4f} (ET={means['dice_ET']:.4f}, TC={means['dice_TC']:.4f}, WT={means['dice_WT']:.4f})")

print(f"\nTotal configs evaluated: {len(all_configs)}")
```

### Task 6: Eval notebook — Ablation visualization (cell 10)

- [ ] **Step 17: Cell 10 — 10-config ablation visualization (code)**

```python
# ── Cell 10: Ablation Visualization ──
import matplotlib.pyplot as plt
import numpy as np

config_names = list(all_configs.keys())
n = len(config_names)

# Extract metrics
et_scores = [all_configs[c]['dice_ET'] for c in config_names]
tc_scores = [all_configs[c]['dice_TC'] for c in config_names]
wt_scores = [all_configs[c]['dice_WT'] for c in config_names]
mean_scores = [all_configs[c]['dice_mean'] for c in config_names]

# Baseline deltas (config 1)
baseline_mean = mean_scores[0] if mean_scores else 0
deltas = [m - baseline_mean for m in mean_scores]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})

# Top: Grouped bar chart
x = np.arange(n)
w = 0.2
bars_et = ax1.bar(x - 1.5*w, et_scores, w, label='ET', color='#e74c3c', alpha=0.85)
bars_tc = ax1.bar(x - 0.5*w, tc_scores, w, label='TC', color='#3498db', alpha=0.85)
bars_wt = ax1.bar(x + 0.5*w, wt_scores, w, label='WT', color='#2ecc71', alpha=0.85)
bars_mn = ax1.bar(x + 1.5*w, mean_scores, w, label='Mean', color='#9b59b6', alpha=0.85)

# Value labels
for bars in [bars_et, bars_tc, bars_wt, bars_mn]:
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=6)

ax1.set_ylabel('Dice Score')
ax1.set_title('TextMamba3D V4.6 — 10-Config Ablation (H100)')
short_names = [c.split(': ')[1] if ': ' in c else c for c in config_names]
ax1.set_xticks(x)
ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
ax1.legend(loc='lower right')
ax1.set_ylim(0.70, 0.95)
ax1.grid(axis='y', alpha=0.3)

# Bottom: Delta chart
colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in deltas]
ax2.bar(x, deltas, 0.6, color=colors, alpha=0.8)
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
for i, d in enumerate(deltas):
    ax2.annotate(f'{d:+.4f}', xy=(i, d), xytext=(0, 3 if d >= 0 else -12),
                 textcoords='offset points', ha='center', fontsize=7)
ax2.set_ylabel('Mean Dice Delta vs Baseline')
ax2.set_xticks(x)
ax2.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('v4.6_eval_ablation.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"Saved: v4.6_eval_ablation.png")
```

- [ ] **Step 18: Commit eval notebook Chunk 2**

```bash
git add TextMamba3D_H100_V4.6_eval.ipynb
git commit -m "feat(eval): add V4.6 comprehensive 10-config ablation notebook"
```

---

## Chunk 3: P1-1 — TC Regression Ablation

### Task 7: TC ablation cells (cells 11-13)

- [ ] **Step 19: Cell 11 — TC ablation header (markdown)**

```markdown
## TC Regression Diagnosis (V4.6)

Three-way comparison to determine if V4.6 fixed the TC regression observed in V4.5:

| Test | Checkpoint | Text Input | Purpose |
|------|-----------|------------|---------|
| T1 | V4.6 | Original text only | Isolate enriched text effect |
| T2 | V4.4 | Original text only | Reproduce V4.4 baseline |
| T3 | V4.6 | Enriched text | Reference (V4.6 production) |

Additionally: **Gate analysis** — extract per-case gate values for the top TC regression cases.
```

- [ ] **Step 20: Cell 12 — TC ablation inference (code)**

```python
# ── Cell 12: TC Ablation — Three Inference Runs ──
import subprocess, yaml, os

# Create V4.6 config with original text only (no enrichment)
with open(V46_CONFIG, 'r') as f:
    v46_orig_cfg = yaml.safe_load(f)
v46_orig_cfg['data']['et_enriched'] = False
v46_orig_config = 'configs/ablation_v46_orig.yaml'
with open(v46_orig_config, 'w') as f:
    yaml.dump(v46_orig_cfg, f)

# V4.4 config (no text_gate, no cross_scale_skip, no ET enrichment)
with open(V46_CONFIG, 'r') as f:
    v44_cfg = yaml.safe_load(f)
v44_cfg['model']['use_text_gate'] = False
v44_cfg['model']['use_cross_scale_skip'] = False
v44_cfg['data']['et_enriched'] = False
v44_ablation_config = 'configs/ablation_v44_tc.yaml'
with open(v44_ablation_config, 'w') as f:
    yaml.dump(v44_cfg, f)

tc_tests = [
    ('Test 1: V4.6 + original text', v46_orig_config, v46_ckpt, 'tc_test1.log'),
    ('Test 2: V4.4 + original text', v44_ablation_config, v44_ckpt, 'tc_test2.log'),
    ('Test 3: V4.6 + enriched text', V46_CONFIG, v46_ckpt, 'tc_test3.log'),
]

for label, config, ckpt, log_file in tc_tests:
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"{'='*60}")

    cmd = [
        'python', 'evaluate_full.py',
        '--config', config,
        '--checkpoint', ckpt,
        '--split', 'test',
    ]

    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    print(f"  -> {log_file} (exit code: {result.returncode})")
```

- [ ] **Step 21: Cell 13 — TC Wilcoxon analysis + gate analysis (code)**

```python
# ── Cell 13: TC Regression Analysis — Wilcoxon + Gate Values ──
import numpy as np
from scipy import stats

# Parse the three test logs
d1 = {r['case']: r for r in parse_eval_log('tc_test1.log')}  # V4.6 + original
d2 = {r['case']: r for r in parse_eval_log('tc_test2.log')}  # V4.4 + original
d3 = {r['case']: r for r in parse_eval_log('tc_test3.log')}  # V4.6 + enriched
common = sorted(set(d1) & set(d2) & set(d3))
print(f"Common cases: {len(common)}")

# ─── Diagnostic 1: Enriched text effect (V4.6 enriched - V4.6 original) ───
print("\n" + "="*60)
print("Diagnostic 1: Enriched Text Effect at Inference")
print("="*60)
for metric in ['dice_ET', 'dice_TC', 'dice_WT', 'dice_mean']:
    vals_enriched = [d3[c][metric] for c in common]
    vals_original = [d1[c][metric] for c in common]
    delta = np.mean(vals_enriched) - np.mean(vals_original)
    _, p = stats.wilcoxon(vals_enriched, vals_original)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {metric}: delta={delta:+.4f}, p={p:.4f} ({sig})")

# ─── Diagnostic 2: Training weight change (V4.6 original - V4.4 original) ───
print("\n" + "="*60)
print("Diagnostic 2: V4.6 vs V4.4 Training Weight Change")
print("="*60)
for metric in ['dice_ET', 'dice_TC', 'dice_WT', 'dice_mean']:
    vals_v46 = [d1[c][metric] for c in common]
    vals_v44 = [d2[c][metric] for c in common]
    delta = np.mean(vals_v46) - np.mean(vals_v44)
    _, p = stats.wilcoxon(vals_v46, vals_v44)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {metric}: delta={delta:+.4f}, p={p:.4f} ({sig})")

# ─── Compare with V4.5 TC regression ───
print("\n" + "="*60)
print("V4.6 vs V4.5 TC Regression Comparison")
print("="*60)
v46_tc = np.mean([d3[c]['dice_TC'] for c in common])
v44_tc = np.mean([d2[c]['dice_TC'] for c in common])
print(f"  V4.4 TC: {v44_tc:.4f}")
print(f"  V4.6 TC: {v46_tc:.4f}")
print(f"  Delta:   {v46_tc - v44_tc:+.4f}")
print(f"  V4.5 TC was: 0.8394 (reference from V4.5 eval)")
if v46_tc > 0.8394:
    print(f"  -> V4.6 RECOVERED TC regression (+{v46_tc - 0.8394:.4f} vs V4.5)")
else:
    print(f"  -> V4.6 TC still below V4.5 ({v46_tc - 0.8394:+.4f})")

# ─── Top TC regression cases (V4.6 vs V4.4) ───
print("\n" + "="*60)
print("Top 10 TC Regression Cases (V4.6 enriched vs V4.4)")
print("="*60)
tc_deltas = [(c, d3[c]['dice_TC'] - d2[c]['dice_TC']) for c in common]
tc_deltas.sort(key=lambda x: x[1])
print(f"{'Case':<30} {'V4.4 TC':>8} {'V4.6 TC':>8} {'Delta':>8}")
for case, delta in tc_deltas[:10]:
    print(f"  {case:<28} {d2[case]['dice_TC']:>8.4f} {d3[case]['dice_TC']:>8.4f} {delta:>+8.4f}")

# ─── Gate value analysis for TC regression cases ───
print("\n" + "="*60)
print("TextScaleGate Analysis")
print("="*60)
# Load model and extract gate values on worst-TC cases
import torch
from evaluate_full import load_model
import yaml

with open(V46_CONFIG, 'r') as f:
    gate_cfg = yaml.safe_load(f)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gate_model = load_model(gate_cfg, v46_ckpt, device)

text_gate = getattr(gate_model, 'text_gate', None)
if text_gate is not None:
    print("Learned gate biases (sigmoid):")
    for i, gate in enumerate(text_gate.gates):
        bias = gate.gate_proj.bias.item()
        sigmoid_val = torch.sigmoid(torch.tensor(bias)).item()
        print(f"  Scale {i+1}: bias={bias:.4f}, sigmoid={sigmoid_val:.4f}")
        # Compare with init: sigmoid(2.0) = 0.8808
        diff = sigmoid_val - 0.8808
        direction = "text MORE active" if diff > 0 else "text LESS active"
        print(f"           vs init (0.8808): {diff:+.4f} ({direction})")
else:
    print("  [WARN] No TextScaleGate found in model")

del gate_model
torch.cuda.empty_cache()
```

- [ ] **Step 22: Commit TC ablation**

```bash
git add TextMamba3D_H100_V4.6_eval.ipynb
git commit -m "feat(eval): add TC regression diagnosis with Wilcoxon tests and gate analysis"
```

---

## Chunk 4: P1-2 — Cross-Version Ensemble Comparison

### Task 8: Ensemble comparison cells (cells 14-16)

- [ ] **Step 23: Cell 14 — Ensemble comparison header (markdown)**

```markdown
## Cross-Version Ensemble Comparison

Compare ensemble performance across version pairs:
- **V4.4 + V4.6** (new) vs **V4.4 + V4.5** (baseline from V4.5 eval)
- Hypothesis: V4.6's new modules (TextScaleGate, CrossScaleSkip) create more diverse error patterns, leading to stronger ensemble complementarity
```

- [ ] **Step 24: Cell 15 — V4.5 predictions + ensemble comparison (code)**

```python
# ── Cell 15: V4.4+V4.5 Ensemble for Comparison ──
import subprocess, os, yaml
import numpy as np

# Generate V4.5 predictions (if not already available)
os.makedirs('eval_preds/v45_text', exist_ok=True)

# V4.5 config: no text_gate, no cross_scale_skip, but with ET enrichment
with open(V46_CONFIG, 'r') as f:
    v45_cfg = yaml.safe_load(f)
v45_cfg['model']['use_text_gate'] = False
v45_cfg['model']['use_cross_scale_skip'] = False
v45_config_path = 'configs/ablation_v45_eval.yaml'
with open(v45_config_path, 'w') as f:
    yaml.dump(v45_cfg, f)

if not os.listdir('eval_preds/v45_text'):
    print("Generating V4.5 predictions...")
    cmd = [
        'python', 'evaluate_full.py',
        '--config', v45_config_path,
        '--checkpoint', v45_ckpt,
        '--split', 'test',
        '--save-preds', 'eval_preds/v45_text',
    ]
    with open('eval_v45_baseline.log', 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    print(f"  -> exit code: {result.returncode}")
else:
    print("V4.5 predictions already exist, skipping inference")

# Generate V4.4+V4.5 ensemble
os.makedirs('eval_preds/ensemble_v44_v45', exist_ok=True)
ens_45_results = ensemble_predictions(
    'eval_preds/v44_text', 'eval_preds/v45_text',
    'eval_preds/ensemble_v44_v45', apply_pp=True, et_min_size=500
)
print(f"V4.4+V4.5 ensemble: {len(ens_45_results)} cases")
```

- [ ] **Step 25: Cell 16 — Head-to-head ensemble comparison visualization (code)**

```python
# ── Cell 16: Ensemble Comparison — V4.4+V4.6 vs V4.4+V4.5 ──
import matplotlib.pyplot as plt
import numpy as np

# Compute metrics for both ensembles
ens_46_metrics = compute_ensemble_metrics('eval_preds/ensemble_v44_v46_pp', None, test_dataset)
ens_45_metrics = compute_ensemble_metrics('eval_preds/ensemble_v44_v45', None, test_dataset)

# Build case-matched comparison
ens_46_by_case = {r['case']: r for r in ens_46_metrics if 'case' in r}
ens_45_by_case = {r['case']: r for r in ens_45_metrics if 'case' in r}
common_ens = sorted(set(ens_46_by_case) & set(ens_45_by_case))

print(f"Matched cases: {len(common_ens)}")

# Aggregate
metrics_summary = {}
for metric in ['dice_ET', 'dice_TC', 'dice_WT', 'dice_mean']:
    v46_vals = [ens_46_by_case[c][metric] for c in common_ens]
    v45_vals = [ens_45_by_case[c][metric] for c in common_ens]
    delta = np.mean(v46_vals) - np.mean(v45_vals)
    _, p = stats.wilcoxon(v46_vals, v45_vals)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    metrics_summary[metric] = {
        'v46_ens': np.mean(v46_vals),
        'v45_ens': np.mean(v45_vals),
        'delta': delta,
        'p': p,
        'sig': sig,
    }
    print(f"{metric}: V4.4+V4.6={np.mean(v46_vals):.4f}, V4.4+V4.5={np.mean(v45_vals):.4f}, "
          f"delta={delta:+.4f} ({sig}, p={p:.4f})")

# Visualization: scatter plot per-case comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, metric, color, title in zip(
    axes,
    ['dice_ET', 'dice_TC', 'dice_WT'],
    ['#e74c3c', '#3498db', '#2ecc71'],
    ['ET (Enhancing Tumor)', 'TC (Tumor Core)', 'WT (Whole Tumor)'],
):
    v46_vals = [ens_46_by_case[c][metric] for c in common_ens]
    v45_vals = [ens_45_by_case[c][metric] for c in common_ens]

    ax.scatter(v45_vals, v46_vals, alpha=0.5, s=20, color=color)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('V4.4+V4.5 Ensemble Dice')
    ax.set_ylabel('V4.4+V4.6 Ensemble Dice')
    ax.set_title(f'{title}\ndelta={metrics_summary[metric]["delta"]:+.4f} ({metrics_summary[metric]["sig"]})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(alpha=0.2)

    # Count wins
    wins_46 = sum(1 for a, b in zip(v46_vals, v45_vals) if a > b)
    wins_45 = sum(1 for a, b in zip(v46_vals, v45_vals) if b > a)
    ax.text(0.05, 0.95, f'V4.6 wins: {wins_46}', transform=ax.transAxes, fontsize=9, va='top')
    ax.text(0.05, 0.88, f'V4.5 wins: {wins_45}', transform=ax.transAxes, fontsize=9, va='top')

plt.suptitle('Ensemble Comparison: V4.4+V4.6 vs V4.4+V4.5 (per-case Dice)', fontsize=14)
plt.tight_layout()
plt.savefig('v4.6_ensemble_comparison.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"Saved: v4.6_ensemble_comparison.png")
```

- [ ] **Step 26: Cell 17 — Final summary table (code)**

```python
# ── Cell 17: Final Summary ──
print("=" * 80)
print("V4.6 EVALUATION SUMMARY")
print("=" * 80)

# Best from each category
categories = {
    'Best single (no TTA)': ['Config 1', 'Config 2', 'Config 5', 'Config 6'],
    'Best single (+TTA)': ['Config 3', 'Config 4', 'Config 7', 'Config 8'],
    'Best ensemble': ['Config 9', 'Config 10'],
}

for cat, prefixes in categories.items():
    matching = {k: v for k, v in all_configs.items() if any(k.startswith(p) for p in prefixes)}
    if matching:
        best_key = max(matching, key=lambda k: matching[k]['dice_mean'])
        best = matching[best_key]
        print(f"\n{cat}: {best_key}")
        print(f"  Mean Dice: {best['dice_mean']:.4f} (ET={best['dice_ET']:.4f}, TC={best['dice_TC']:.4f}, WT={best['dice_WT']:.4f})")

# V4.5 reference comparison
print("\n" + "-" * 80)
print("V4.5 Reference (from V4.5 eval notebook):")
print("  Best single (V4.5 text +TTA): Mean=0.8468 (ET=0.7842, TC=0.8528, WT=0.9034)")
print("  Best ensemble (V4.4+V4.5 +PP): Mean=0.8422 (ET=0.7814, TC=0.8497, WT=0.8953)")
print("-" * 80)

# TC regression status
if 'Config 1: V4.6 text' in all_configs:
    v46_tc = all_configs['Config 1: V4.6 text']['dice_TC']
    v45_tc_ref = 0.8394  # V4.5 baseline TC
    v44_tc_ref = 0.8513  # V4.4 baseline TC
    print(f"\nTC Regression Status:")
    print(f"  V4.4 TC baseline: {v44_tc_ref:.4f}")
    print(f"  V4.5 TC baseline: {v45_tc_ref:.4f} (regression: {v45_tc_ref - v44_tc_ref:+.4f})")
    print(f"  V4.6 TC baseline: {v46_tc:.4f} (vs V4.5: {v46_tc - v45_tc_ref:+.4f}, vs V4.4: {v46_tc - v44_tc_ref:+.4f})")
    if v46_tc > v45_tc_ref:
        print(f"  VERDICT: TextScaleGate + CrossScaleSkip IMPROVED TC over V4.5")
    elif v46_tc > v44_tc_ref:
        print(f"  VERDICT: TC fully recovered to V4.4 level")
    else:
        print(f"  VERDICT: TC regression persists — consider class weight adjustment")

print("\n" + "=" * 80)
```

- [ ] **Step 27: Final commit**

```bash
git add TextMamba3D_H100_V4.6_eval.ipynb
git commit -m "feat(eval): add cross-version ensemble comparison and final summary"
```

---

## Execution Notes

### Notebook Cell Order (TextMamba3D_H100_V4.6_eval.ipynb)

| Cell | Type | Content |
|------|------|---------|
| 0 | markdown | Title + 10-config ablation matrix |
| 1 | code | Environment setup |
| 2 | code | Restore code & data from Drive |
| 3 | code | Verify V4.6 patches |
| 4 | code | Checkpoint verification |
| 5 | code | Phase A: Baseline inference (configs 1,2,5,6) |
| 6 | code | Phase B: TTA inference (configs 3,4,7,8) |
| 7 | code | Phase C: V4.4 predictions for ensemble |
| 8 | code | Offline ensemble: V4.4+V4.6 |
| 9 | code | Aggregate all 10 configs |
| 10 | code | Ablation visualization |
| 11 | markdown | TC ablation header |
| 12 | code | TC ablation: 3 inference runs |
| 13 | code | TC Wilcoxon analysis + gate analysis |
| 14 | markdown | Ensemble comparison header |
| 15 | code | V4.5 predictions + V4.4+V4.5 ensemble |
| 16 | code | Ensemble scatter plot comparison |
| 17 | code | Final summary table |

### Dependencies

- `evaluate_full.py` already has `--tta`, `--postprocess`, `--save-preds` flags (verified)
- `parse_eval_log()` function is defined in cell 9 and reused in cell 13
- `ensemble_predictions()` function is defined in cell 8 and reused in cell 15
- `compute_ensemble_metrics()` function is defined in cell 9 and reused in cell 16
- `test_dataset` is loaded in cell 9 and reused in cells 15-16
