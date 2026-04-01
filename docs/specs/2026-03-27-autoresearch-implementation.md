# AutoResearch Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local orchestrator that auto-generates experiments, writes notebook cells, parses results, and cascades from zero-cost inference optimizations through hyperparameter search to module-level A/B testing.

**Architecture:** Python orchestrator on Mac manages an experiment queue (JSON). It generates notebook cells for Colab execution, collects results from Drive/git, and uses Claude API to propose next hypotheses. Three layers cascade: L0 (inference-only) → L1 (hyperparam search) → L2 (architecture, needs user approval).

**Tech Stack:** Python 3.12, nbformat (notebook generation), anthropic SDK (hypothesis engine), pyyaml, json

**Spec:** `docs/specs/2026-03-27-autoresearch-design.md`

---

## File Structure

```
autoresearch/
├── __init__.py              — package init
├── orchestrator.py          — main loop: queue → generate → collect → analyze
├── experiments.json         — experiment queue (seeded with L0 experiments)
├── results.json             — collected results log
├── notebook_writer.py       — generate/inject cells into .ipynb
├── result_collector.py      — parse eval output from notebook/Drive
├── hypothesis_engine.py     — Claude API integration for next-experiment proposals
├── config_generator.py      — generate YAML configs for L1 hyperparam search
└── cell_templates/
    ├── eval_cell.py.j2      — Jinja2 template for evaluation cell
    └── train_cell.py.j2     — Jinja2 template for training cell
```

**Existing files modified:**
- None. AutoResearch is a standalone module that generates notebooks and configs.

---

## Chunk 1: Core Data Structures + Experiment Queue

### Task 1: Experiment and Result schemas

**Files:**
- Create: `autoresearch/__init__.py`
- Create: `autoresearch/experiments.json`
- Create: `autoresearch/results.json`
- Test: `tests/test_autoresearch_schemas.py`

- [ ] **Step 1: Write schema test**

```python
# tests/test_autoresearch_schemas.py
import json
import os

def test_experiments_json_valid():
    with open('autoresearch/experiments.json') as f:
        data = json.load(f)
    assert 'queue' in data
    assert 'completed' in data
    assert isinstance(data['queue'], list)
    for exp in data['queue']:
        assert 'id' in exp
        assert 'layer' in exp
        assert exp['layer'] in [0, 1, 2]
        assert 'type' in exp
        assert 'params' in exp

def test_results_json_valid():
    with open('autoresearch/results.json') as f:
        data = json.load(f)
    assert 'baseline' in data
    assert data['baseline']['mean_dice'] == 0.8479
    assert 'experiments' in data
    assert isinstance(data['experiments'], list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/leiyuxuan/Documents/PyCharm_Project/TextMamba3D && python -m pytest tests/test_autoresearch_schemas.py -v`
Expected: FAIL (files not found)

- [ ] **Step 3: Create __init__.py**

```python
# autoresearch/__init__.py
"""AutoResearch: self-evolving experiment framework for TextMamba3D."""
```

- [ ] **Step 4: Create experiments.json with L0 queue**

```json
{
  "queue": [
    {
      "id": "L0-1",
      "layer": 0,
      "type": "eval_sweep",
      "name": "ET/WT ratio relabeling",
      "description": "Relabel ET as NCR when ET/WT volume ratio < threshold",
      "params": {
        "checkpoint": "best_v5.0.pth",
        "config": "configs/archive/textbrats_a100_v5.yaml",
        "split": "test",
        "use_text": true,
        "tta": true,
        "sweep": {
          "et_wt_ratio": [0.02, 0.03, 0.04, 0.05]
        }
      }
    },
    {
      "id": "L0-2",
      "layer": 0,
      "type": "eval_sweep",
      "name": "Advanced PP parameter sweep",
      "description": "Sweep et_min_size and wt_min_size with best et_wt_ratio",
      "params": {
        "checkpoint": "best_v5.0.pth",
        "config": "configs/archive/textbrats_a100_v5.yaml",
        "split": "test",
        "use_text": true,
        "tta": true,
        "sweep": {
          "et_min_size": [50, 100, 200, 500],
          "wt_min_size": [200, 500, 1000]
        }
      }
    },
    {
      "id": "L0-3",
      "layer": 0,
      "type": "eval_compare",
      "name": "TTA ablation",
      "description": "Compare with and without TTA (BraTS 2023 winner found TTA can hurt)",
      "params": {
        "checkpoint": "best_v5.0.pth",
        "config": "configs/archive/textbrats_a100_v5.yaml",
        "split": "test",
        "use_text": true,
        "configs": [
          {"name": "with_tta", "tta": true, "advanced_pp": true},
          {"name": "no_tta", "tta": false, "advanced_pp": true}
        ]
      }
    },
    {
      "id": "L0-4",
      "layer": 0,
      "type": "eval_sweep",
      "name": "ET probability boost",
      "description": "Multiply ET softmax channel by boost factor before argmax",
      "params": {
        "checkpoint": "best_v5.0.pth",
        "config": "configs/archive/textbrats_a100_v5.yaml",
        "split": "test",
        "use_text": true,
        "tta": true,
        "sweep": {
          "et_boost": [1.0, 1.1, 1.2, 1.3, 1.5]
        }
      }
    }
  ],
  "completed": []
}
```

- [ ] **Step 5: Create results.json**

```json
{
  "baseline": {
    "version": "v5.0",
    "checkpoint": "best_v5.0.pth",
    "mean_dice": 0.8479,
    "dice_ET": 0.7910,
    "dice_TC": 0.8560,
    "dice_WT": 0.8967,
    "config": "text+TTA+PP"
  },
  "best": {
    "mean_dice": 0.8479,
    "experiment_id": "baseline"
  },
  "experiments": []
}
```

- [ ] **Step 6: Run tests**

Run: `cd /Users/leiyuxuan/Documents/PyCharm_Project/TextMamba3D && python -m pytest tests/test_autoresearch_schemas.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add autoresearch/ tests/test_autoresearch_schemas.py
git commit -m "feat(autoresearch): experiment queue and result schemas with L0 seeds"
```

---

### Task 2: Result Collector

**Files:**
- Create: `autoresearch/result_collector.py`
- Test: `tests/test_result_collector.py`

- [ ] **Step 1: Write test**

```python
# tests/test_result_collector.py
from autoresearch.result_collector import parse_eval_output

SAMPLE_OUTPUT = """
============================================================
Results: test split, 95 cases, text=True
============================================================
  dice_ET: 0.7910 +/- 0.2300
  dice_TC: 0.8560 +/- 0.1710
  dice_WT: 0.8967 +/- 0.1021
  dice_mean: 0.8479 +/- 0.1321
  hd95_ET: 3.50 +/- 17.29
  hd95_TC: 2.39 +/- 17.49
  hd95_WT: 2.91 +/- 13.49
============================================================
"""

def test_parse_eval_output():
    metrics = parse_eval_output(SAMPLE_OUTPUT)
    assert metrics['dice_ET'] == 0.7910
    assert metrics['dice_TC'] == 0.8560
    assert metrics['dice_WT'] == 0.8967
    assert metrics['dice_mean'] == 0.8479

def test_parse_empty():
    metrics = parse_eval_output("no metrics here")
    assert metrics == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_result_collector.py -v`
Expected: FAIL

- [ ] **Step 3: Implement result_collector.py**

```python
# autoresearch/result_collector.py
"""Parse evaluation output from evaluate_full.py."""
import re


def parse_eval_output(text: str) -> dict:
    """Extract dice/hd95 metrics from evaluate_full.py stdout."""
    metrics = {}
    for key in ['dice_ET', 'dice_TC', 'dice_WT', 'dice_mean',
                'hd95_ET', 'hd95_TC', 'hd95_WT']:
        match = re.search(rf'{key}: ([\d.]+) \+/- ([\d.]+)', text)
        if match:
            metrics[key] = float(match.group(1))
            metrics[f'{key}_std'] = float(match.group(2))
    return metrics


def is_improvement(metrics: dict, baseline_mean_dice: float = 0.8479) -> bool:
    """Check if experiment improved over baseline."""
    return metrics.get('dice_mean', 0) > baseline_mean_dice
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_result_collector.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch/result_collector.py tests/test_result_collector.py
git commit -m "feat(autoresearch): result collector with eval output parser"
```

---

### Task 3: Config Generator

**Files:**
- Create: `autoresearch/config_generator.py`
- Test: `tests/test_config_generator.py`

- [ ] **Step 1: Write test**

```python
# tests/test_config_generator.py
import yaml
from autoresearch.config_generator import generate_l1_config

def test_generate_l1_config():
    cfg = generate_l1_config(
        base_config='configs/archive/textbrats_a100_v5.yaml',
        overrides={'training': {'lr': 0.0002, 'epochs': 140}},
        experiment_name='L1-test-lr0002',
    )
    assert cfg['training']['lr'] == 0.0002
    assert cfg['training']['epochs'] == 140
    assert cfg['experiment']['name'] == 'L1-test-lr0002'

def test_generate_l1_config_preserves_base():
    cfg = generate_l1_config(
        base_config='configs/archive/textbrats_a100_v5.yaml',
        overrides={'training': {'lr': 0.001}},
        experiment_name='test',
    )
    # Base values preserved
    assert cfg['model']['embed_dim'] == 48
    assert cfg['data']['batch_size'] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_generator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement config_generator.py**

```python
# autoresearch/config_generator.py
"""Generate YAML configs for L1 hyperparameter search."""
import copy
import os
import yaml


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base dict."""
    result = copy.deepcopy(base)
    for k, v in overrides.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def generate_l1_config(
    base_config: str,
    overrides: dict,
    experiment_name: str,
) -> dict:
    """Generate a config for L1 hyperparameter search.

    Args:
        base_config: path to base YAML config
        overrides: nested dict of values to override
        experiment_name: name for the experiment
    Returns:
        merged config dict
    """
    with open(base_config) as f:
        base = yaml.safe_load(f)

    cfg = _deep_merge(base, overrides)
    cfg.setdefault('experiment', {})
    cfg['experiment']['name'] = experiment_name
    return cfg


def save_config(cfg: dict, path: str) -> str:
    """Save config dict as YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_config_generator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch/config_generator.py tests/test_config_generator.py
git commit -m "feat(autoresearch): config generator with deep merge"
```

---

## Chunk 2: Notebook Writer + Layer 0 Notebook

### Task 4: Notebook Writer

**Files:**
- Create: `autoresearch/notebook_writer.py`
- Test: `tests/test_notebook_writer.py`

- [ ] **Step 1: Write test**

```python
# tests/test_notebook_writer.py
import json
from autoresearch.notebook_writer import generate_eval_cell, create_notebook

def test_generate_eval_cell():
    cell = generate_eval_cell(
        experiment_id='L0-1',
        checkpoint='best_v5.0.pth',
        config='configs/archive/textbrats_a100_v5.yaml',
        split='test',
        use_text=True,
        tta=True,
        extra_flags=['--et-wt-ratio', '0.03', '--advanced-pp'],
    )
    assert cell['cell_type'] == 'code'
    src = ''.join(cell['source'])
    assert 'evaluate_full.py' in src
    assert '--et-wt-ratio' in src
    assert 'L0-1' in src

def test_create_notebook():
    cells = [
        generate_eval_cell('test', 'ckpt.pth', 'cfg.yaml', 'test', True, False, [])
    ]
    nb = create_notebook(cells, title='Test Notebook')
    assert nb['nbformat'] == 4
    assert len(nb['cells']) == 2  # title + 1 eval cell
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_notebook_writer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement notebook_writer.py**

```python
# autoresearch/notebook_writer.py
"""Generate Jupyter notebook cells for Colab execution."""
import json
import os


def _code_cell(source: str, experiment_id: str = '') -> dict:
    """Create a notebook code cell."""
    lines = source.strip().split('\n')
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {'experiment_id': experiment_id},
        'outputs': [],
        'source': [l + '\n' for l in lines[:-1]] + [lines[-1]],
    }


def _md_cell(source: str) -> dict:
    """Create a notebook markdown cell."""
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [source],
    }


def generate_eval_cell(
    experiment_id: str,
    checkpoint: str,
    config: str,
    split: str,
    use_text: bool,
    tta: bool,
    extra_flags: list[str],
) -> dict:
    """Generate a single evaluation cell."""
    text_flag = '--use-text' if use_text else '--no-text'
    tta_flag = '--tta' if tta else ''
    extras = ' '.join(extra_flags)

    source = f"""# Experiment: {experiment_id}
import subprocess, os, json
os.chdir(REPO_DIR)

DRIVE_CKPT = os.path.join(DRIVE_BASE, 'checkpoints')
ckpt = os.path.join(DRIVE_CKPT, '{checkpoint}')
assert os.path.exists(ckpt), f'Checkpoint not found: {{ckpt}}'

print('=' * 70)
print('Experiment: {experiment_id}')
print('=' * 70)

cmd = ['python', '-u', 'evaluate_full.py',
       '--config', '{config}',
       '--checkpoint', ckpt,
       '--split', '{split}',
       '{text_flag}',
       '--overlap', '0.5']
{f"cmd.append('{tta_flag}')" if tta else '# TTA disabled'}
cmd.extend({repr(extra_flags)})

ret = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True)
print(ret.stdout[-1500:] if len(ret.stdout) > 1500 else ret.stdout)
if ret.returncode != 0:
    print('STDERR:', ret.stderr[-500:])

# Save result
import re
metrics = {{}}
for key in ['dice_ET', 'dice_TC', 'dice_WT', 'dice_mean']:
    m = re.search(rf'{{key}}: ([\\d.]+) \\+/- ([\\d.]+)', ret.stdout)
    if m:
        metrics[key] = float(m.group(1))

result = {{'id': '{experiment_id}', 'metrics': metrics}}
results_path = os.path.join(DRIVE_BASE, 'autoresearch_results.json')
try:
    with open(results_path) as f:
        all_results = json.load(f)
except FileNotFoundError:
    all_results = []
all_results.append(result)
with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'Result saved: {{metrics}}')"""

    return _code_cell(source, experiment_id)


def generate_train_cell(
    experiment_id: str,
    config_path: str,
    resume_checkpoint: str,
    extra_args: list[str],
) -> dict:
    """Generate a training cell."""
    extras = ' '.join(extra_args)
    source = f"""# Training: {experiment_id}
import os, shutil, glob
os.chdir(REPO_DIR)

DRIVE_CKPT = os.path.join(DRIVE_BASE, 'checkpoints')

# Clean local checkpoints
for f in glob.glob(os.path.join(REPO_DIR, 'checkpoints', '*.pth')):
    os.remove(f)

ckpt = os.path.join(DRIVE_CKPT, '{resume_checkpoint}')
print('Training: {experiment_id}')
!python -u train.py \\
    --config {config_path} \\
    --resume "{{ckpt}}" \\
    {extras}

# Sync
for f in glob.glob(os.path.join(REPO_DIR, 'checkpoints', '*.pth')):
    shutil.copy2(f, os.path.join(DRIVE_CKPT, os.path.basename(f)))
print('Checkpoints synced')"""

    return _code_cell(source, experiment_id)


def create_notebook(cells: list[dict], title: str) -> dict:
    """Create a complete notebook from cells."""
    title_cell = _md_cell(f'# {title}')
    return {
        'nbformat': 4,
        'nbformat_minor': 0,
        'metadata': {
            'colab': {'provenance': []},
            'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
        },
        'cells': [title_cell] + cells,
    }


def save_notebook(nb: dict, path: str) -> None:
    """Save notebook to .ipynb file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_notebook_writer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch/notebook_writer.py tests/test_notebook_writer.py
git commit -m "feat(autoresearch): notebook writer for eval and train cells"
```

---

### Task 5: Generate Layer 0 Notebook

**Files:**
- Create: `autoresearch/generate_l0.py`

- [ ] **Step 1: Write the L0 notebook generator**

```python
# autoresearch/generate_l0.py
"""Generate Layer 0 (zero-cost inference optimization) notebook."""
import json
from autoresearch.notebook_writer import (
    generate_eval_cell, create_notebook, save_notebook, _md_cell, _code_cell,
)


def generate_l0_notebook(output_path: str) -> str:
    """Generate L0 notebook with all inference-time experiments."""
    with open('autoresearch/experiments.json') as f:
        data = json.load(f)

    # Setup cell
    setup = _code_cell("""# AutoResearch Layer 0: Zero-Cost Inference Optimization
from google.colab import drive
drive.mount('/content/drive')

!pip install -q mamba-ssm causal-conv1d einops transformers nibabel pyyaml tqdm scipy

import os, subprocess, shutil, zipfile
REPO_DIR = '/content/TextMamba3D'
DRIVE_BASE = '/content/drive/MyDrive/TextMamba3D'

# Clone/pull repo
git_dir = os.path.join(REPO_DIR, '.git')
if os.path.isdir(git_dir):
    os.chdir(REPO_DIR)
    subprocess.run(['git', 'pull'], check=True)
else:
    subprocess.run(['git', 'clone', '--depth', '1',
        'https://github.com/PlutoLei/TextMamba3D.git', REPO_DIR], check=True)
    os.chdir(REPO_DIR)

# Data
DATA_DIR = './data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
if not os.path.exists(DATA_DIR):
    with zipfile.ZipFile(f'{DRIVE_BASE}/TextBraTS_data.zip', 'r') as zf:
        zf.extractall(os.path.dirname(DATA_DIR))
ET_CACHE = f'{DRIVE_BASE}/et_enriched.zip'
if os.path.exists(ET_CACHE):
    with zipfile.ZipFile(ET_CACHE, 'r') as zf:
        zf.extractall(DATA_DIR)
print('Setup complete')""")

    cells = [setup]

    # Generate cells for each L0 experiment
    for exp in data['queue']:
        if exp['layer'] != 0:
            continue

        cells.append(_md_cell(f"## {exp['id']}: {exp['name']}\\n\\n{exp['description']}"))

        params = exp['params']
        if exp['type'] == 'eval_sweep':
            sweep = params.get('sweep', {})
            # Generate one cell per sweep combination
            sweep_keys = list(sweep.keys())
            if len(sweep_keys) == 1:
                key = sweep_keys[0]
                for val in sweep[key]:
                    flag_map = {
                        'et_wt_ratio': ['--advanced-pp', '--et-wt-ratio', str(val)],
                        'et_min_size': ['--advanced-pp', '--et-min-size', str(val)],
                        'wt_min_size': ['--advanced-pp', '--wt-min-size', str(val)],
                        'et_boost': ['--et-boost', str(val)],
                    }
                    flags = flag_map.get(key, [f'--{key}', str(val)])
                    cell_id = f"{exp['id']}_{key}={val}"
                    cells.append(generate_eval_cell(
                        cell_id, params['checkpoint'], params['config'],
                        params['split'], params['use_text'], params.get('tta', False),
                        flags,
                    ))
        elif exp['type'] == 'eval_compare':
            for cfg in params.get('configs', []):
                flags = []
                if cfg.get('tta'):
                    flags.append('--tta')
                if cfg.get('advanced_pp'):
                    flags.append('--advanced-pp')
                cell_id = f"{exp['id']}_{cfg['name']}"
                cells.append(generate_eval_cell(
                    cell_id, params['checkpoint'], params['config'],
                    params['split'], params['use_text'], cfg.get('tta', False),
                    flags,
                ))

    # Summary cell
    cells.append(_code_cell("""# Layer 0 Summary
import json, os

results_path = os.path.join(DRIVE_BASE, 'autoresearch_results.json')
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)
    print(f'Total experiments: {len(results)}')
    baseline = 0.8479
    improvements = [r for r in results if r['metrics'].get('dice_mean', 0) > baseline]
    print(f'Improvements over baseline: {len(improvements)}')
    for r in sorted(results, key=lambda x: x['metrics'].get('dice_mean', 0), reverse=True)[:5]:
        m = r['metrics']
        print(f"  {r['id']}: Mean={m.get('dice_mean', 0):.4f} ET={m.get('dice_ET', 0):.4f}")
else:
    print('No results found')"""))

    nb = create_notebook(cells, 'AutoResearch Layer 0: Zero-Cost Inference Optimization')
    save_notebook(nb, output_path)
    return output_path


if __name__ == '__main__':
    path = generate_l0_notebook('TextMamba3D_AutoResearch_L0.ipynb')
    print(f'Generated: {path}')
```

- [ ] **Step 2: Run generator**

Run: `cd /Users/leiyuxuan/Documents/PyCharm_Project/TextMamba3D && python -m autoresearch.generate_l0`
Expected: `TextMamba3D_AutoResearch_L0.ipynb` created

- [ ] **Step 3: Verify notebook is valid**

Run: `python -c "import json; json.load(open('TextMamba3D_AutoResearch_L0.ipynb')); print('Valid')"`
Expected: `Valid`

- [ ] **Step 4: Commit**

```bash
git add autoresearch/generate_l0.py TextMamba3D_AutoResearch_L0.ipynb
git commit -m "feat(autoresearch): Layer 0 notebook generator"
```

---

## Chunk 3: Orchestrator Main Loop

### Task 6: Orchestrator

**Files:**
- Create: `autoresearch/orchestrator.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Write test**

```python
# tests/test_orchestrator.py
from autoresearch.orchestrator import Orchestrator

def test_next_experiment_returns_l0_first():
    orch = Orchestrator('autoresearch/experiments.json', 'autoresearch/results.json')
    exp = orch.next_experiment()
    assert exp is not None
    assert exp['layer'] == 0

def test_record_result():
    orch = Orchestrator('autoresearch/experiments.json', 'autoresearch/results.json')
    exp = orch.next_experiment()
    orch.record_result(exp['id'], {'dice_mean': 0.85, 'dice_ET': 0.80})
    assert len(orch.results['experiments']) == 1

def test_is_improvement():
    orch = Orchestrator('autoresearch/experiments.json', 'autoresearch/results.json')
    assert orch.is_improvement({'dice_mean': 0.85}) is True
    assert orch.is_improvement({'dice_mean': 0.84}) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_orchestrator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement orchestrator.py**

```python
# autoresearch/orchestrator.py
"""AutoResearch orchestrator: manage experiment lifecycle."""
import json
import os
from datetime import datetime


class Orchestrator:
    """Manages experiment queue, results, and layer cascading."""

    def __init__(self, experiments_path: str, results_path: str):
        self.experiments_path = experiments_path
        self.results_path = results_path

        with open(experiments_path) as f:
            self.experiments = json.load(f)
        with open(results_path) as f:
            self.results = json.load(f)

    def next_experiment(self) -> dict | None:
        """Get next experiment from queue (L0 first, then L1, then L2)."""
        queue = self.experiments['queue']
        if not queue:
            return None
        # Sort by layer to ensure L0 runs first
        queue.sort(key=lambda x: x['layer'])
        return queue[0]

    def record_result(self, experiment_id: str, metrics: dict) -> None:
        """Record experiment result and move from queue to completed."""
        result = {
            'id': experiment_id,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'improved': self.is_improvement(metrics),
        }
        self.results['experiments'].append(result)

        # Update best if improved
        if self.is_improvement(metrics):
            self.results['best'] = {
                'mean_dice': metrics['dice_mean'],
                'experiment_id': experiment_id,
            }

        # Move from queue to completed
        self.experiments['queue'] = [
            e for e in self.experiments['queue'] if e['id'] != experiment_id
        ]
        completed_exp = next(
            (e for e in self.experiments.get('completed', [])
             if e['id'] == experiment_id), None
        )
        if not completed_exp:
            self.experiments.setdefault('completed', []).append({
                'id': experiment_id, 'result': result
            })

        self._save()

    def is_improvement(self, metrics: dict) -> bool:
        """Check if metrics beat current best."""
        return metrics.get('dice_mean', 0) > self.results['best']['mean_dice']

    def consecutive_finetune_failures(self) -> int:
        """Count consecutive L1 fine-tune experiments without improvement."""
        count = 0
        for exp in reversed(self.results['experiments']):
            if exp.get('id', '').startswith('L1') and not exp.get('improved'):
                count += 1
            else:
                break
        return count

    def should_train_from_scratch(self) -> bool:
        """Return True if 3 consecutive fine-tunes failed."""
        return self.consecutive_finetune_failures() >= 3

    def status(self) -> str:
        """Return human-readable status."""
        queue_by_layer = {}
        for e in self.experiments['queue']:
            queue_by_layer.setdefault(e['layer'], []).append(e)

        lines = [
            f"Best: {self.results['best']['mean_dice']:.4f} "
            f"({self.results['best']['experiment_id']})",
            f"Queue: {len(self.experiments['queue'])} experiments",
        ]
        for layer in sorted(queue_by_layer):
            lines.append(f"  L{layer}: {len(queue_by_layer[layer])}")
        lines.append(
            f"Completed: {len(self.results['experiments'])} experiments"
        )
        improvements = sum(
            1 for e in self.results['experiments'] if e.get('improved')
        )
        lines.append(f"Improvements: {improvements}")
        return '\n'.join(lines)

    def _save(self) -> None:
        """Persist state to disk."""
        with open(self.experiments_path, 'w') as f:
            json.dump(self.experiments, f, indent=2)
        with open(self.results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_orchestrator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(autoresearch): orchestrator with queue management and layer cascading"
```

---

## Chunk 4: Hypothesis Engine (Claude API)

### Task 7: Hypothesis Engine

**Files:**
- Create: `autoresearch/hypothesis_engine.py`
- Test: `tests/test_hypothesis_engine.py`

- [ ] **Step 1: Write test**

```python
# tests/test_hypothesis_engine.py
from autoresearch.hypothesis_engine import format_results_for_analysis

def test_format_results():
    results = {
        'baseline': {'mean_dice': 0.8479, 'dice_ET': 0.7910},
        'experiments': [
            {'id': 'L0-1', 'metrics': {'dice_mean': 0.8500, 'dice_ET': 0.8000}, 'improved': True},
            {'id': 'L0-2', 'metrics': {'dice_mean': 0.8400, 'dice_ET': 0.7800}, 'improved': False},
        ]
    }
    prompt = format_results_for_analysis(results)
    assert 'baseline' in prompt.lower()
    assert 'L0-1' in prompt
    assert '0.8500' in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hypothesis_engine.py -v`
Expected: FAIL

- [ ] **Step 3: Implement hypothesis_engine.py**

```python
# autoresearch/hypothesis_engine.py
"""Claude API integration for hypothesis generation."""
import json
import os


def format_results_for_analysis(results: dict) -> str:
    """Format experiment results into a prompt for Claude API."""
    lines = [
        "# TextMamba3D AutoResearch Results",
        "",
        f"## Baseline: V5.0 Mean Dice = {results['baseline']['mean_dice']:.4f}",
        f"  ET={results['baseline'].get('dice_ET', 'N/A')}, "
        f"TC={results['baseline'].get('dice_TC', 'N/A')}, "
        f"WT={results['baseline'].get('dice_WT', 'N/A')}",
        "",
        "## Experiments (chronological):",
    ]
    for exp in results.get('experiments', []):
        m = exp['metrics']
        status = 'IMPROVED' if exp.get('improved') else 'no improvement'
        lines.append(
            f"- {exp['id']}: Mean={m.get('dice_mean', 0):.4f} "
            f"ET={m.get('dice_ET', 0):.4f} [{status}]"
        )
    return '\n'.join(lines)


SYSTEM_PROMPT = """You are an ML research assistant for brain tumor segmentation.
Given experiment results, propose the next experiment to try.

Rules:
- Layer 0 (inference-only): post-processing params, TTA config, probability calibration
- Layer 1 (training): learning rate, loss weights, augmentation settings
- Layer 2 (architecture): new modules, loss functions — mark as NEEDS_APPROVAL

Output format (JSON):
{
  "id": "L1-xxx",
  "layer": 1,
  "type": "train",
  "name": "short description",
  "rationale": "why this might work based on results so far",
  "params": { ... config overrides ... },
  "needs_approval": false
}"""


def generate_hypothesis(results: dict, api_key: str | None = None) -> dict | None:
    """Call Claude API to generate next experiment hypothesis.

    Returns experiment dict or None if API unavailable.
    """
    if api_key is None:
        api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("No ANTHROPIC_API_KEY set. Skipping hypothesis generation.")
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = format_results_for_analysis(results)

        response = client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{'role': 'user', 'content': prompt}],
        )
        text = response.content[0].text

        # Extract JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"Hypothesis generation failed: {e}")

    return None
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_hypothesis_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch/hypothesis_engine.py tests/test_hypothesis_engine.py
git commit -m "feat(autoresearch): hypothesis engine with Claude API integration"
```

---

## Chunk 5: CLI Entry Point

### Task 8: CLI and Main Entry

**Files:**
- Create: `autoresearch/__main__.py`

- [ ] **Step 1: Implement CLI**

```python
# autoresearch/__main__.py
"""AutoResearch CLI entry point.

Usage:
    python -m autoresearch status          — show current state
    python -m autoresearch next            — show next experiment
    python -m autoresearch generate-l0     — generate Layer 0 notebook
    python -m autoresearch record ID JSON  — record experiment result
    python -m autoresearch hypothesize     — generate next hypothesis via Claude API
"""
import json
import sys

from autoresearch.orchestrator import Orchestrator

EXPERIMENTS = 'autoresearch/experiments.json'
RESULTS = 'autoresearch/results.json'


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]
    orch = Orchestrator(EXPERIMENTS, RESULTS)

    if cmd == 'status':
        print(orch.status())

    elif cmd == 'next':
        exp = orch.next_experiment()
        if exp:
            print(json.dumps(exp, indent=2))
        else:
            print("Queue empty. Run 'hypothesize' to generate new experiments.")

    elif cmd == 'generate-l0':
        from autoresearch.generate_l0 import generate_l0_notebook
        path = generate_l0_notebook('TextMamba3D_AutoResearch_L0.ipynb')
        print(f'Generated: {path}')

    elif cmd == 'record' and len(sys.argv) >= 4:
        exp_id = sys.argv[2]
        metrics = json.loads(sys.argv[3])
        orch.record_result(exp_id, metrics)
        improved = orch.is_improvement(metrics)
        print(f"Recorded {exp_id}: {'IMPROVED!' if improved else 'no improvement'}")
        if orch.should_train_from_scratch():
            print("WARNING: 3 consecutive fine-tune failures. Consider training from scratch.")

    elif cmd == 'hypothesize':
        from autoresearch.hypothesis_engine import generate_hypothesis
        hypothesis = generate_hypothesis(orch.results)
        if hypothesis:
            print(json.dumps(hypothesis, indent=2))
            if hypothesis.get('needs_approval'):
                print("\n*** This is a Layer 2 experiment. Needs your approval. ***")
        else:
            print("Could not generate hypothesis. Set ANTHROPIC_API_KEY.")

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Test CLI**

Run: `cd /Users/leiyuxuan/Documents/PyCharm_Project/TextMamba3D && python -m autoresearch status`
Expected: Shows baseline and queue status

- [ ] **Step 3: Commit**

```bash
git add autoresearch/__main__.py
git commit -m "feat(autoresearch): CLI entry point with status/next/generate/record/hypothesize"
```

---

## Execution Summary

| Chunk | Tasks | Files Created | Tests |
|-------|-------|--------------|-------|
| 1 | 1-3 | 6 source + 3 test | 7 test functions |
| 2 | 4-5 | 2 source + 1 test + 1 notebook | 2 test functions |
| 3 | 6 | 1 source + 1 test | 3 test functions |
| 4 | 7 | 1 source + 1 test | 1 test function |
| 5 | 8 | 1 source | manual CLI test |

**Total:** 8 tasks, 11 source files, 5 test files, 13 test functions

**First milestone:** After Chunk 2, Layer 0 notebook is ready to run on Colab.
**Full milestone:** After Chunk 5, orchestrator CLI is operational.
