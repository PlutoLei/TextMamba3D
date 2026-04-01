# V8.0 Two-Stage Pretraining + 3-Model Ensemble Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (1) Pretrain TextMamba3D on BraTS2021 (1251 cases) then fine-tune on BraTS2020+TextBraTS for ET breakthrough. (2) Build 3-model ensemble (V5.0+V6.0+V7.0) as immediate baseline improvement.

**Architecture:** Stage 1 uses existing train.py with `--no-text-ratio 1.0` on BraTS2021 data. Stage 2 resumes from Stage 1 checkpoint with `--reset-optimizer` on BraTS2020+TextBraTS. Ensemble uses saved softmax probs from Drive. Almost zero code changes needed.

**Tech Stack:** PyTorch, nibabel, existing train.py/evaluate_full.py

---

## File Structure

```
# Part A: V8.0 Two-Stage
configs/autoresearch/
├── V8.0_stage1_pretrain.yaml   — CREATE: BraTS2021 pretrain config
├── V8.0_stage2_finetune.yaml   — CREATE: BraTS2020+TextBraTS fine-tune config
TextMamba3D_V8.0.ipynb           — CREATE: two-stage notebook

# Part B: 3-Model Ensemble
TextMamba3D_Ensemble_V50_V60_V70.ipynb — CREATE: 3-model ensemble notebook
```

**No code changes required.** All functionality exists:
- `--no-text-ratio 1.0` for text-free training
- `--resume --reset-optimizer --reset-lr` for Stage 2
- `strict=False` checkpoint loading
- `--save-preds` for softmax probs
- `scripts/prepare_brats.py` for data prep

---

## Part A: V8.0 Two-Stage Pretraining

### Task 1: Download BraTS2021 to Google Drive

No code needed. This is a manual + notebook step.

- [ ] **Step 1: Create download cell in notebook**

BraTS2021 download options:
- Kaggle: `https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1` (~5 GB)
- Synapse: `https://www.synapse.org/#!Synapse:syn25829067`

Notebook cell downloads via Kaggle API:
```python
# Install kaggle, configure API key, download
!pip install -q kaggle
!mkdir -p ~/.kaggle
# User needs to upload kaggle.json or set env var
!kaggle datasets download -d dschettler8845/brats-2021-task1 -p /content/brats2021_raw
!unzip -q /content/brats2021_raw/*.zip -d /content/brats2021_raw/
```

Then run prepare script:
```python
!python scripts/prepare_brats.py --input /content/brats2021_raw --output /content/BraTS2021
```

- [ ] **Step 2: Commit notebook**

### Task 2: Stage 1 Config (BraTS2021 pretrain)

**Files:**
- Create: `configs/autoresearch/V8.0_stage1_pretrain.yaml`

- [ ] **Step 1: Generate config**

```python
from autoresearch.config_generator import generate_l1_config, save_config

cfg = generate_l1_config(
    base_config='configs/archive/textbrats_a100_v5.yaml',
    overrides={
        'data': {
            'data_dir': '/content/BraTS2021',
            'dataset_type': 'brats',  # NOT textbrats
            'batch_size': 4,  # more data = can use larger batch
        },
        'training': {
            'lr': 0.0001,
            'epochs': 200,
            'patience': 40,
            'no_text_ratio': 1.0,  # 100% no text
            'gradient_checkpointing': False,
            'contrastive_warmup_epochs': 9999,  # disable contrastive
        },
        'loss': {
            'contrastive_weight': 0.0,  # no text = no contrastive
            'use_boundary': True,
            'boundary_max_weight': 1.0,
            'use_hierarchy': True,
            'hierarchy_weight': 0.1,
        },
        'experiment': {
            'name': 'V8.0_stage1_pretrain',
            'description': 'Stage 1: BraTS2021 pretrain (1251 cases, no text)',
        },
    },
    experiment_name='V8.0_stage1_pretrain',
)
save_config(cfg, 'configs/autoresearch/V8.0_stage1_pretrain.yaml')
```

- [ ] **Step 2: Commit**

```bash
git add configs/autoresearch/V8.0_stage1_pretrain.yaml
git commit -m "feat(v8): Stage 1 pretrain config (BraTS2021, no text)"
```

### Task 3: Stage 2 Config (BraTS2020+TextBraTS fine-tune)

**Files:**
- Create: `configs/autoresearch/V8.0_stage2_finetune.yaml`

- [ ] **Step 1: Generate config**

```python
cfg = generate_l1_config(
    base_config='configs/archive/textbrats_a100_v5.yaml',
    overrides={
        'training': {
            'lr': 0.00005,  # lower lr for fine-tune
            'epochs': 100,
            'patience': 30,
            'gradient_checkpointing': False,
        },
        'loss': {
            'use_boundary': True,
            'boundary_max_weight': 1.0,
            'use_hierarchy': True,
            'hierarchy_weight': 0.1,
        },
        'experiment': {
            'name': 'V8.0_stage2_finetune',
            'description': 'Stage 2: BraTS2020+TextBraTS fine-tune from Stage 1',
        },
    },
    experiment_name='V8.0_stage2_finetune',
)
save_config(cfg, 'configs/autoresearch/V8.0_stage2_finetune.yaml')
```

- [ ] **Step 2: Commit**

```bash
git add configs/autoresearch/V8.0_stage2_finetune.yaml
git commit -m "feat(v8): Stage 2 fine-tune config (BraTS2020+TextBraTS)"
```

### Task 4: V8.0 Notebook

**Files:**
- Create: `TextMamba3D_V8.0.ipynb`

Notebook structure (10 cells):

1. **Setup** — mount Drive, install deps, clone repo
2. **Download BraTS2021** — Kaggle API or manual upload
3. **Prepare BraTS2021** — run prepare_brats.py, verify case count
4. **Stage 1 Smoke Test** — 2 samples, 1 epoch, verify no-text path works
5. **Stage 1 Training** — `!python -u train.py --config V8.0_stage1_pretrain.yaml --no-text-ratio 1.0 --grad-accum 1`
6. **Stage 1 Eval (no text)** — evaluate on BraTS2020 test set without text
7. **Stage 2 Training** — `!python -u train.py --config V8.0_stage2_finetune.yaml --resume best_V8.0_stage1.pth --reset-optimizer --reset-lr --no-text-ratio 0.15 --grad-accum 2`
8. **Stage 2 Eval** — text+TTA and notext+TTA
9. **Results comparison** — V5.0 vs V8.0

Key CLI commands:

Stage 1:
```bash
python -u train.py \
    --config configs/autoresearch/V8.0_stage1_pretrain.yaml \
    --no-text-ratio 1.0 \
    --grad-accum 1
```

Stage 2:
```bash
python -u train.py \
    --config configs/autoresearch/V8.0_stage2_finetune.yaml \
    --resume checkpoints/best_V8.0_stage1.pth \
    --reset-optimizer \
    --reset-lr \
    --no-text-ratio 0.15 \
    --grad-accum 2
```

- [ ] **Step 1: Generate notebook**
- [ ] **Step 2: Verify valid JSON**
- [ ] **Step 3: Commit**

```bash
git add TextMamba3D_V8.0.ipynb
git commit -m "feat(v8): two-stage training notebook (BraTS2021 pretrain + TextBraTS fine-tune)"
```

---

## Part B: 3-Model Ensemble (V5.0 + V6.0 + V7.0)

### Task 5: 3-Model Ensemble Notebook

**Files:**
- Create: `TextMamba3D_Ensemble_V50_V60_V70.ipynb`

Key difference from 2-model ensemble: V7.0 probs need to be saved first.

Notebook structure (8 cells):

1. **Setup** — mount Drive, install deps, clone, data, patch fusion.py
2. **Save V7.0 predictions** — `evaluate_full.py --save-preds` (V5.0 and V6.0 probs already on Drive)
3. **3-model uniform ensemble** — equal weights (1/3 each)
4. **Grid search** — search w50, w60, w70 (with constraint w50+w60+w70=1)
5. **Results comparison**

For the grid search, use a simplified approach:
- Fix w70 at [0.1, 0.2, 0.3]
- Search w50 from 0.3 to 0.7 in 0.1 steps
- w60 = 1 - w50 - w70
- Total: 3 × 5 = 15 combos (manageable on CPU)

To avoid the OOM issue from before:
- Process cases ONE AT A TIME (don't load all 95 into memory)
- Use numpy argmax + manual dice (no torch tensors for full volumes)

- [ ] **Step 1: Generate notebook**
- [ ] **Step 2: Commit**

```bash
git add TextMamba3D_Ensemble_V50_V60_V70.ipynb
git commit -m "feat: 3-model ensemble notebook (V5.0+V6.0+V7.0)"
```

- [ ] **Step 3: Push all**

```bash
git push origin main
```

---

## Execution Order

| Priority | Task | GPU Time | Dependency |
|----------|------|----------|------------|
| **1** | Task 5: 3-model ensemble | ~30 min (V7.0 inference only) | V7.0 checkpoint on Drive |
| **2** | Task 1: Download BraTS2021 | 0 (download) | Kaggle account |
| **3** | Tasks 2-3: Configs | 0 | None |
| **4** | Task 4: V8.0 Stage 1 training | ~10-15h | BraTS2021 data |
| **5** | Task 4: V8.0 Stage 2 fine-tune | ~3-5h | Stage 1 checkpoint |

**Recommended:** Run Task 5 (ensemble) immediately while downloading BraTS2021 data for V8.0.
