# Text-Aware Copy-Paste Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a novel text-image-mask joint augmentation system where Copy-Paste augmentation automatically generates synchronized quantitative ET text descriptions, producing consistent (image, text, mask) triplets.

**Architecture:** Three modules — (1) `et_quantitative_text.py` computes ET statistics from any mask and generates quantitative English text, (2) Dataset integration appends quantitative text after transforms (so it reflects augmented data), (3) Offline paraphrase script generates 10 text variants per case via LLM API. The quantitative text module is pure numpy (no torch), tested independently.

**Tech Stack:** Python, numpy, scipy.ndimage (already in project). Claude API for paraphrase generation (optional, offline script).

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `data/et_quantitative_text.py` | Compute ET stats from mask → generate quantitative English text | **Create** |
| `tests/test_et_quantitative_text.py` | Unit tests for quantitative text generation | **Create** |
| `data/brats_textbrats_dataset.py` | Append quantitative text to expert text after transforms | **Modify** (lines 208-228) |
| `configs/default.yaml` | Add `et_quantitative: false` config option | **Modify** |
| `scripts/generate_text_paraphrases.py` | Offline LLM paraphrase generation (10 variants/case) | **Create** |

---

### Task 1: Quantitative ET Text Generator

**Files:**
- Create: `data/et_quantitative_text.py`
- Test: `tests/test_et_quantitative_text.py`

- [ ] **Step 1: Write failing tests for `compute_et_stats`**

```python
# tests/test_et_quantitative_text.py
"""Tests for quantitative ET text generation."""
import numpy as np
import os
import importlib.util

# Direct import to avoid torch dependency (same pattern as test_postprocess.py)
_mod_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "et_quantitative_text.py")
_spec = importlib.util.spec_from_file_location("et_quantitative_text", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_et_stats = _mod.compute_et_stats
generate_quantitative_et_text = _mod.generate_quantitative_et_text


def _make_volume(shape=(64, 64, 64), fill=0):
    return np.full(shape, fill, dtype=np.int64)


def _place_sphere(vol, center, radius, label):
    z, y, x = np.ogrid[:vol.shape[0], :vol.shape[1], :vol.shape[2]]
    dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    vol[dist <= radius] = label
    return vol


class TestComputeEtStats:
    def test_no_tumor(self):
        mask = _make_volume()
        stats = compute_et_stats(mask)
        assert stats["et_voxels"] == 0
        assert stats["et_ratio"] == 0.0

    def test_with_et(self):
        mask = _make_volume()
        _place_sphere(mask, (32, 32, 32), 15, 2)  # edema
        _place_sphere(mask, (32, 32, 32), 10, 1)  # NCR
        _place_sphere(mask, (32, 32, 32), 7, 3)   # ET
        stats = compute_et_stats(mask)
        assert stats["et_voxels"] > 0
        assert 0 < stats["et_ratio"] < 1.0
        assert stats["n_et_clusters"] >= 1
        assert stats["side"] in ("left", "right")
        assert "position" in stats

    def test_small_et(self):
        mask = _make_volume()
        _place_sphere(mask, (32, 32, 32), 15, 2)
        _place_sphere(mask, (32, 32, 32), 10, 1)
        mask[32, 32, 32] = 3  # single voxel ET
        stats = compute_et_stats(mask)
        assert stats["et_voxels"] == 1
        assert stats["size_category"] == "minimal"

    def test_multiple_clusters(self):
        mask = _make_volume()
        _place_sphere(mask, (32, 32, 32), 15, 2)
        mask[10:13, 10:13, 10:13] = 3  # cluster 1
        mask[50:53, 50:53, 50:53] = 3  # cluster 2
        stats = compute_et_stats(mask)
        assert stats["n_et_clusters"] == 2


class TestGenerateQuantitativeEtText:
    def test_no_et_text(self):
        mask = _make_volume()
        text = generate_quantitative_et_text(mask)
        assert "no enhancing" in text.lower() or "absent" in text.lower()

    def test_with_et_text(self):
        mask = _make_volume()
        _place_sphere(mask, (32, 32, 32), 15, 2)
        _place_sphere(mask, (32, 32, 32), 10, 1)
        _place_sphere(mask, (32, 32, 32), 7, 3)
        text = generate_quantitative_et_text(mask)
        assert "%" in text  # should contain ratio
        assert len(text) > 20

    def test_text_varies_with_et_size(self):
        mask1 = _make_volume()
        _place_sphere(mask1, (32, 32, 32), 15, 2)
        mask1[32, 32, 32] = 3  # tiny ET

        mask2 = _make_volume()
        _place_sphere(mask2, (32, 32, 32), 15, 2)
        _place_sphere(mask2, (32, 32, 32), 10, 1)
        _place_sphere(mask2, (32, 32, 32), 7, 3)  # large ET

        text1 = generate_quantitative_et_text(mask1)
        text2 = generate_quantitative_et_text(mask2)
        assert text1 != text2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd E:/VSCode_Project/TextMamba3D && python -m pytest tests/test_et_quantitative_text.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `data/et_quantitative_text.py`**

```python
# data/et_quantitative_text.py
"""Quantitative ET text generation from segmentation masks.

Computes ET statistics (volume ratio, cluster count, spatial location,
size category) and generates structured English text. Pure numpy — no
torch dependency. Used in two contexts:
  1. Offline: precompute for all training cases
  2. Online: in __getitem__ after transforms (reflects augmented data)

BraTS labels: 0=bg, 1=NCR, 2=ED, 3=ET
"""

import numpy as np
from scipy.ndimage import label as ndimage_label


def compute_et_stats(mask: np.ndarray) -> dict:
    """Compute ET statistics from a BraTS label map.

    Args:
        mask: [D, H, W] integer label map (0-3).

    Returns:
        dict with keys: et_voxels, wt_voxels, tc_voxels, et_ratio,
        et_tc_ratio, n_et_clusters, size_category, side, position.
    """
    et_mask = mask == 3
    wt_mask = mask > 0
    tc_mask = (mask == 1) | (mask == 3)

    et_voxels = int(et_mask.sum())
    wt_voxels = int(wt_mask.sum())
    tc_voxels = int(tc_mask.sum())

    et_ratio = et_voxels / max(1, wt_voxels)
    et_tc_ratio = et_voxels / max(1, tc_voxels)

    # Cluster count
    if et_voxels > 0:
        _, n_clusters = ndimage_label(et_mask)
    else:
        n_clusters = 0

    # Size category
    if et_voxels == 0:
        size_cat = "absent"
    elif et_voxels < 50:
        size_cat = "minimal"
    elif et_voxels < 500:
        size_cat = "small"
    elif et_voxels < 2000:
        size_cat = "moderate"
    else:
        size_cat = "large"

    # Spatial location
    side = ""
    position = ""
    if et_voxels > 0:
        coords = np.argwhere(et_mask)
        centroid = coords.mean(axis=0)
        norm = centroid / np.array(mask.shape)
        # BraTS [D, H, W]: axis 0=depth(SI), axis 1=height(AP), axis 2=width(LR)
        side = "left" if norm[2] > 0.5 else "right"
        ap = "anterior" if norm[1] > 0.5 else "posterior"
        si = "superior" if norm[0] > 0.5 else "inferior"
        position = f"{ap} {si}"

    return {
        "et_voxels": et_voxels,
        "wt_voxels": wt_voxels,
        "tc_voxels": tc_voxels,
        "et_ratio": float(et_ratio),
        "et_tc_ratio": float(et_tc_ratio),
        "n_et_clusters": n_clusters,
        "size_category": size_cat,
        "side": side,
        "position": position,
    }


def generate_quantitative_et_text(mask: np.ndarray) -> str:
    """Generate quantitative ET description from a BraTS label map.

    Args:
        mask: [D, H, W] integer label map (0-3).

    Returns:
        English text describing ET quantitatively.
    """
    stats = compute_et_stats(mask)

    if stats["et_voxels"] == 0:
        return "Enhancing tumor component is absent in this case."

    parts = []

    # Size and ratio
    ratio_pct = stats["et_ratio"] * 100
    tc_pct = stats["et_tc_ratio"] * 100
    parts.append(
        f"The enhancing component constitutes {ratio_pct:.1f}% of the "
        f"whole tumor volume ({stats['size_category']} enhancement)"
    )

    # TC ratio
    parts.append(f"representing {tc_pct:.1f}% of the tumor core")

    # Cluster info
    n = stats["n_et_clusters"]
    if n == 1:
        parts.append("forming a single contiguous cluster")
    else:
        parts.append(f"distributed across {n} separate clusters")

    # Location
    if stats["side"] and stats["position"]:
        parts.append(
            f"centered in the {stats['side']} {stats['position']} region"
        )

    return ", ".join(parts) + "."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd E:/VSCode_Project/TextMamba3D && python -m pytest tests/test_et_quantitative_text.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd E:/VSCode_Project/TextMamba3D
git add data/et_quantitative_text.py tests/test_et_quantitative_text.py
git commit -m "feat: add quantitative ET text generator from masks"
```

---

### Task 2: Dataset Integration — Dynamic Quantitative Text

**Files:**
- Modify: `data/brats_textbrats_dataset.py:33-46` (constructor args)
- Modify: `data/brats_textbrats_dataset.py:208-228` (text assembly in `__getitem__`)
- Modify: `configs/default.yaml` (add config key)

- [ ] **Step 1: Add `et_quantitative` parameter to `TextBraTSDataset.__init__`**

In `data/brats_textbrats_dataset.py`, add to constructor signature (after `enriched_prob`):

```python
        et_quantitative: bool = False,  # Append quantitative ET stats to text
```

And store it:

```python
        self.et_quantitative = et_quantitative
```

- [ ] **Step 2: Add quantitative text after transforms in `__getitem__`**

In `data/brats_textbrats_dataset.py`, after the existing text assembly block (line ~228), before tokenization (line ~231), insert:

```python
        # Quantitative ET text (computed AFTER transforms so it reflects augmented data)
        if self.et_quantitative and self.split == 'train':
            from data.et_quantitative_text import generate_quantitative_et_text
            # mask is [D, H, W] tensor after transforms; convert to numpy
            mask_np = mask.numpy() if hasattr(mask, 'numpy') else mask
            if mask_np.ndim == 4:  # [1, D, H, W] → squeeze
                mask_np = mask_np.squeeze(0)
            qt = generate_quantitative_et_text(mask_np)
            text = text + " " + qt
```

- [ ] **Step 3: Add config key to `configs/default.yaml`**

Under `augmentation:` section, add:

```yaml
  et_quantitative: false     # Append quantitative ET stats to text (dynamic, post-augmentation)
```

- [ ] **Step 4: Wire config in `train.py`**

In `train.py`, where the dataset is constructed, pass the new parameter. Find the `TextBraTSDataset` construction and add:

```python
        et_quantitative=aug_cfg.get('et_quantitative', False),
```

- [ ] **Step 5: Verify no import errors**

Run: `cd E:/VSCode_Project/TextMamba3D && python -c "from data.brats_textbrats_dataset import TextBraTSDataset; print('OK')"`
Expected: `OK` (on a machine with torch installed)

- [ ] **Step 6: Commit**

```bash
cd E:/VSCode_Project/TextMamba3D
git add data/brats_textbrats_dataset.py configs/default.yaml train.py
git commit -m "feat: integrate dynamic quantitative ET text into dataset pipeline"
```

---

### Task 3: Text Paraphrase Generation Script

**Files:**
- Create: `scripts/generate_text_paraphrases.py`

- [ ] **Step 1: Write the paraphrase generation script**

```python
# scripts/generate_text_paraphrases.py
"""Generate text paraphrases for TextBraTS dataset using Claude API.

Reads expert text descriptions and generates N paraphrases per case,
preserving clinical meaning while varying wording, sentence order,
and detail level.

Usage:
    python scripts/generate_text_paraphrases.py <data_dir> [--n 10] [--dry-run]

Output:
    <case_dir>/<case_name>_paraphrase_01.txt
    <case_dir>/<case_name>_paraphrase_02.txt
    ...
"""

import os
import sys
import json
import argparse
import time


PARAPHRASE_PROMPT = """You are a neuroradiologist writing clinical MRI reports.
Rewrite the following brain tumor MRI description in a different style while
preserving ALL clinical information (location, signal characteristics, tumor
components, mass effect). Vary the sentence structure, word choice, and level
of detail. Do NOT add or remove any clinical findings.

Original description:
{text}

Write exactly one rewritten version. Output only the rewritten text, nothing else."""


def generate_paraphrases(text: str, n: int = 10, model: str = "claude-sonnet-4-20250514") -> list:
    """Generate N paraphrases of a clinical text using Claude API."""
    try:
        import anthropic
    except ImportError:
        print("Error: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()
    paraphrases = []

    for i in range(n):
        response = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0.9,
            messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
        )
        paraphrases.append(response.content[0].text.strip())
        time.sleep(0.5)  # Rate limiting

    return paraphrases


def main():
    parser = argparse.ArgumentParser(description="Generate text paraphrases")
    parser.add_argument("data_dir", help="BraTS data directory")
    parser.add_argument("--n", type=int, default=10, help="Paraphrases per case")
    parser.add_argument("--dry-run", action="store_true", help="Print first case only")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    args = parser.parse_args()

    cases = sorted(
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    )

    for i, case_name in enumerate(cases):
        case_dir = os.path.join(args.data_dir, case_name)
        text_file = os.path.join(case_dir, f"{case_name}_flair_text.txt")

        if not os.path.exists(text_file):
            continue

        with open(text_file, "r", encoding="utf-8") as f:
            original = f.read().strip()

        if args.dry_run:
            print(f"Case: {case_name}")
            print(f"Original ({len(original)} chars): {original[:100]}...")
            paraphrases = generate_paraphrases(original, n=1, model=args.model)
            print(f"Paraphrase: {paraphrases[0][:100]}...")
            break

        paraphrases = generate_paraphrases(original, n=args.n, model=args.model)

        for j, para in enumerate(paraphrases):
            out_path = os.path.join(case_dir, f"{case_name}_paraphrase_{j+1:02d}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(para)

        print(f"[{i+1}/{len(cases)}] {case_name}: {len(paraphrases)} paraphrases")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add paraphrase loading to dataset**

In `data/brats_textbrats_dataset.py`, add a method to load random paraphrase:

```python
    def _load_random_paraphrase(self, case_dir: str, case_name: str) -> Optional[str]:
        """Load a random paraphrase if available."""
        import glob
        pattern = os.path.join(case_dir, f"{case_name}_paraphrase_*.txt")
        files = glob.glob(pattern)
        if not files:
            return None
        chosen = files[np.random.randint(0, len(files))]
        with open(chosen, 'r', encoding='utf-8') as f:
            return f.read().strip()
```

In `__getitem__`, after loading expert text (line ~213), add:

```python
        # Text paraphrase augmentation (training only)
        if self.split == 'train' and np.random.random() < 0.3:
            para = self._load_random_paraphrase(case_dir, case_name)
            if para:
                original_text = para
```

- [ ] **Step 3: Commit**

```bash
cd E:/VSCode_Project/TextMamba3D
git add scripts/generate_text_paraphrases.py data/brats_textbrats_dataset.py
git commit -m "feat: add text paraphrase generation and loading"
```

---

### Task 4: V5.3 Config for Combined Augmentation

**Files:**
- Create: `configs/a100_v5.3.yaml`

- [ ] **Step 1: Create V5.3 experiment config**

```yaml
# configs/a100_v5.3.yaml
# V5.3: Text-Aware Copy-Paste + Quantitative ET Text + ET Oversampling
# Fine-tune from V5.0 checkpoint

data:
  data_dir: "/content/drive/MyDrive/BraTS2020/BraTS2020_TrainingData"
  dataset_type: "textbrats"    # REQUIRED: uses TextBraTSDataset (not BraTS2021)
  patch_size: [128, 128, 128]
  batch_size: 1
  num_workers: 2
  train_ratio: 0.6
  val_ratio: 0.15

augmentation:
  use_elastic: true
  use_modality_dropout: true
  use_copy_paste: true         # NEW: Tumor Copy-Paste
  copy_paste_prob: 0.3
  use_et_oversample: true      # NEW: ET-weighted sampling
  et_boost_factor: 3.0
  et_quantitative: true        # NEW: Dynamic quantitative ET text

model:
  img_size: [128, 128, 128]
  in_channels: 4
  out_channels: 4
  embed_dim: 48                # Must match V5.0 checkpoint
  depths: [2, 2, 2, 2]
  d_state: 16
  text_embed_dim: 256          # Must match V5.0 checkpoint (NOT 768)
  text_max_len: 192
  use_pretrained_text: true
  unfreeze_text_layers: 2
  text_model_path: "/content/drive/MyDrive/pretrained/pubmedbert"
  dropout: 0.1
  use_text_gate: true
  text_gate_init_bias: 2.0
  use_mamba3: true
  headdim: 48
  fusion_type: "seqca"

loss:
  dice_weight: 1.0
  ce_weight: 1.0
  contrastive_weight: 0.0
  class_weights: [0.25, 3.0, 1.0, 4.0]

training:
  epochs: 200
  lr: 0.00005
  weight_decay: 0.01
  warmup_epochs: 10
  patience: 40
  no_text_ratio: 0.1
  gradient_checkpointing: false  # Mamba2 incompatible with torch checkpoint
  gradient_clip_norm: 1.0
  gradient_accumulation: 4
  deep_supervision: false
  use_amp: true
  resume_from: "/content/drive/MyDrive/TextMamba3D/checkpoints/best_v5.0.pth"
  reset_lr: true
  reset_optimizer: true
  es_metric: "dice_ET"

eval:
  metrics: ["dice", "hd95"]

inference:
  overlap: 0.5
  use_tta: true
  tta_flips: 8
```

- [ ] **Step 2: Commit**

```bash
cd E:/VSCode_Project/TextMamba3D
git add configs/a100_v5.3.yaml
git commit -m "feat: add V5.3 config with text-aware augmentation"
```

---

## Execution Notes

- Tasks 1-3 can run on Windows (no torch needed for Task 1 tests, Task 3 is a script)
- Task 2 integration requires torch (verify on Colab)
- Task 3 paraphrase generation requires `ANTHROPIC_API_KEY` env var and `pip install anthropic`
- V5.3 training runs on Colab A100 with `configs/a100_v5.3.yaml`

## Post-Completion Validation

On Colab, run evaluation comparing:
1. V5.0 baseline (no augmentation)
2. V5.3 with `--advanced-pp --et-min-size 200 --et-wt-ratio 0.03`
3. V5.3 without advanced PP (isolate training augmentation effect)
