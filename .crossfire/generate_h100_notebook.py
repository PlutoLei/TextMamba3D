"""Generate TextMamba3D_H100_V4.6.ipynb — H100 80GB optimized training notebook.

Reads V4.6 source files and constructs a self-contained Colab notebook with
all patches inline (SeqCA + V4.6 modules + decoder + model + config + train.py).
"""

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "TextMamba3D_H100_V4.6.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


# ---------------------------------------------------------------------------
# Read V4.6 source files
# ---------------------------------------------------------------------------
fusion_py = (REPO / "models" / "fusion.py").read_text(encoding="utf-8")
decoder_py = (REPO / "models" / "decoder_3d.py").read_text(encoding="utf-8")
textmamba_py = (REPO / "models" / "textmamba3d.py").read_text(encoding="utf-8")

# Extract V4.6-only modules from fusion.py (lines after CrossScaleSkipAttention header)
v46_marker = "# Cross-Scale Skip Attention (AttnRes-inspired, V4.6)"
v46_start = fusion_py.index(v46_marker)
v46_modules = fusion_py[v46_start:]

# Modify textmamba3d.py for notebook: use MultiScaleSeqCA instead of MultiScalePixelTextAttention
# (because SeqCA is patched in via cell 4, replacing the base PixelTextCrossAttention)
nb_textmamba = textmamba_py.replace(
    "from .fusion import MultiScalePixelTextAttention, MultiScaleTextGate",
    "from .fusion import MultiScaleSeqCA, MultiScaleTextGate",
).replace(
    "self.multi_scale_attn = MultiScalePixelTextAttention(",
    "self.multi_scale_attn = MultiScaleSeqCA(",
)
assert "MultiScalePixelTextAttention" not in nb_textmamba, \
    "FATAL: MultiScalePixelTextAttention replacement failed in textmamba3d.py"
assert "MultiScaleSeqCA" in nb_textmamba, \
    "FATAL: MultiScaleSeqCA not found after replacement"

# ---------------------------------------------------------------------------
# Cell 0: Title
# ---------------------------------------------------------------------------
cell_0 = md("""\
# TextMamba3D \u2014 H100 Training Pipeline (v4.6)

**V4.6: AttnRes-inspired Cross-Scale Skip Attention + Text Scale Gate**

| Feature | Description |
|---------|-------------|
| Direction A | CrossScaleSkipAttention supplements decoder skip connections with cross-scale context |
| Direction B | TextScaleGate adaptively mixes raw vs text-fused features per scale |
| H100 80GB | batch_size=8, no gradient_checkpointing, sw_batch_size=4, num_workers=8 |

Config: `configs/textbrats_v8_h100.yaml`
""")

# ---------------------------------------------------------------------------
# Cell 1: Mount Drive + install
# ---------------------------------------------------------------------------
cell_1 = code("""\
# Mount Google Drive (run in Colab web UI if using VS Code plugin)
from google.colab import drive
drive.mount('/content/drive')

# Install packages (cached on Drive)
!nvidia-smi 2>/dev/null || echo "No GPU detected (CPU mode)"
!pip install -q --cache-dir=/content/drive/MyDrive/pip_cache mamba-ssm causal-conv1d transformers nibabel tensorboard pyyaml tqdm
""")

# ---------------------------------------------------------------------------
# Cell 2: Extract code + data
# ---------------------------------------------------------------------------
cell_2 = code("""\
import os, zipfile, shutil

REPO_DIR = '/content/TextMamba3D'
DRIVE_BASE = '/content/drive/MyDrive/TextMamba3D'
DRIVE_CODE_ZIP = os.path.join(DRIVE_BASE, 'TextMamba3D_code.zip')
DRIVE_CODE_DIR = os.path.join(DRIVE_BASE, 'TextMamba3D_code')

# Code source priority: VS Code plugin sync > Drive zip > Drive folder
tm_file = os.path.join(REPO_DIR, 'models/textmamba3d.py')
if os.path.exists(tm_file):
    print(f'Local code available at {REPO_DIR} (VS Code plugin)')
elif os.path.exists(DRIVE_CODE_ZIP):
    print(f'Extracting code from {DRIVE_CODE_ZIP}...')
    os.makedirs(REPO_DIR, exist_ok=True)
    with zipfile.ZipFile(DRIVE_CODE_ZIP, 'r') as zf:
        zf.extractall(REPO_DIR)
    print(f'Extracted to {REPO_DIR}')
elif os.path.exists(DRIVE_CODE_DIR):
    print(f'Copying code from {DRIVE_CODE_DIR}...')
    shutil.copytree(DRIVE_CODE_DIR, REPO_DIR)
    print(f'Copied to {REPO_DIR}')
else:
    raise FileNotFoundError(
        f'Code not found. Please either:\\n'
        f'  1. Use VS Code Colab plugin to sync local project\\n'
        f'  2. Upload TextMamba3D_code.zip to {DRIVE_BASE} on Google Drive'
    )

os.chdir(REPO_DIR)
print(f'Working directory: {os.getcwd()}')

# Extract BraTS data from Drive
DATA_ZIP = os.path.join(DRIVE_BASE, "TextBraTS_data.zip")
DATA_DIR = os.path.join(REPO_DIR, "data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")

if not os.path.exists(DATA_DIR):
    os.makedirs(os.path.dirname(DATA_DIR), exist_ok=True)
    if os.path.exists(DATA_ZIP):
        print(f"Extracting {DATA_ZIP}...")
        with zipfile.ZipFile(DATA_ZIP, 'r') as zf:
            zf.extractall(os.path.dirname(DATA_DIR))
        if os.path.exists(DATA_DIR):
            print(f"Data extracted. Cases: {len(os.listdir(DATA_DIR))}")
        else:
            print(f"ERROR: Expected path not found after extraction: {DATA_DIR}")
            print("Actual contents:", os.listdir(os.path.dirname(DATA_DIR)))
    else:
        print(f"ERROR: {DATA_ZIP} not found on Drive")
else:
    print(f"Data already exists. Cases: {len(os.listdir(DATA_DIR))}")

# Count samples
if os.path.exists(DATA_DIR):
    cases = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    print(f"Total BraTS cases: {len(cases)}")
""")

# ---------------------------------------------------------------------------
# Cell 3: Patches header
# ---------------------------------------------------------------------------
cell_3 = md("""\
## Code Patches (V4.5 + V4.6)

**V4.5 patches:** SeqCA fusion, ET-enriched dataset
**V4.6 patches:** RMSNorm, CrossScaleSkipAttention, TextScaleGate, MultiScaleTextGate + modified decoder + model

All patches are idempotent (safe to re-run).
""")

# ---------------------------------------------------------------------------
# Cell 4: [v4.4-1] Append SeqCA to fusion.py (from V4.5 notebook)
# ---------------------------------------------------------------------------
# Build the SeqCA code string for inline patching
seqca_lines = []
in_seqca = False
for line in fusion_py.splitlines():
    if "# Sequential Cross-Attention (TextBraTS-inspired" in line:
        in_seqca = True
    if in_seqca:
        if "# Cross-Scale Skip Attention (AttnRes-inspired" in line:
            break
        seqca_lines.append(line)

# Remove trailing empty lines
while seqca_lines and seqca_lines[-1].strip() == "":
    seqca_lines.pop()

seqca_block = "\n".join(seqca_lines)

cell_4 = code(f"""\
import pathlib

# [v4.4-1] Append Sequential Cross-Attention classes to models/fusion.py
fusion_path = pathlib.Path('models/fusion.py')
content = fusion_path.read_text(encoding='utf-8')

if 'SequentialCrossAttention' in content:
    print("[v4.4-1] SeqCA already exists in fusion.py, skipping")
else:
    seqca_code = '''

{seqca_block}
'''
    fusion_path.write_text(content + seqca_code, encoding='utf-8')
    print("[v4.4-1] Appended SeqCA + MultiScaleSeqCA to fusion.py")

# Verify
content = fusion_path.read_text(encoding='utf-8')
assert 'SequentialCrossAttention' in content, "SeqCA not found!"
assert 'MultiScaleSeqCA' in content, "MultiScaleSeqCA not found!"
print(f"  fusion.py size: {{len(content)}} chars")
""")

# ---------------------------------------------------------------------------
# Cell 5: [v4.6-1] Append V4.6 modules to fusion.py
# ---------------------------------------------------------------------------
cell_5 = code(f"""\
import pathlib

# [v4.6-1] Append V4.6 modules to fusion.py:
#   RMSNorm, CrossScaleSkipAttention, TextScaleGate, MultiScaleTextGate
fusion_path = pathlib.Path('models/fusion.py')
content = fusion_path.read_text(encoding='utf-8')

if 'CrossScaleSkipAttention' in content:
    print("[v4.6-1] V4.6 modules already exist in fusion.py, skipping")
else:
    v46_code = '''

# ---------------------------------------------------------------------------
{v46_modules}'''
    fusion_path.write_text(content + v46_code, encoding='utf-8')
    print("[v4.6-1] Appended V4.6 modules to fusion.py")

# Verify
content = fusion_path.read_text(encoding='utf-8')
for cls in ['RMSNorm', 'CrossScaleSkipAttention', 'TextScaleGate', 'MultiScaleTextGate']:
    assert cls in content, f"{{cls}} not found in fusion.py!"
print(f"  fusion.py final size: {{len(content)}} chars, all V4.6 modules present")
""")

# ---------------------------------------------------------------------------
# Cell 6: [v4.6-2] Overwrite decoder_3d.py
# ---------------------------------------------------------------------------
cell_6 = code(f"""\
import pathlib

# [v4.6-2] Overwrite decoder_3d.py with V4.6 version
# Includes CrossScaleSkipAttention integration (supplements, not replaces, skip connections)
decoder_path = pathlib.Path('models/decoder_3d.py')

decoder_content = '''{decoder_py}'''

decoder_path.write_text(decoder_content, encoding='utf-8')
print("[v4.6-2] Overwritten decoder_3d.py with V4.6 version")
print(f"  Features: CrossScaleSkipAttention (supplemental), PatchExpanding3D, deep supervision")
""")

# ---------------------------------------------------------------------------
# Cell 7: [v4.6-3] Overwrite textmamba3d.py (SeqCA + V4.6 features)
# ---------------------------------------------------------------------------
cell_7 = code(f"""\
import pathlib

# [v4.6-3] Overwrite textmamba3d.py with V4.6 version
# Uses MultiScaleSeqCA (from V4.4 patch) + V4.6 TextScaleGate + use_cross_scale_skip
tm_path = pathlib.Path('models/textmamba3d.py')

tm_content = '''{nb_textmamba}'''

tm_path.write_text(tm_content, encoding='utf-8')
print("[v4.6-3] Overwritten textmamba3d.py with V4.6 version")
print("  Imports: MultiScaleSeqCA, MultiScaleTextGate")
print("  New params: use_text_gate, use_cross_scale_skip, text_gate_init_bias")
""")

# ---------------------------------------------------------------------------
# Cell 8: [v4.5-1] ET-enriched dataset patch (from V4.5)
# ---------------------------------------------------------------------------
cell_8 = code("""\
import pathlib, shutil

# [v4.5-1] Patch brats_textbrats_dataset.py: add ET-enriched stochastic selection
ds_path = pathlib.Path('data/brats_textbrats_dataset.py')
ds_content = ds_path.read_text(encoding='utf-8')

if 'et_enriched' in ds_content:
    print("[v4.5-1] ET-enriched patch already applied, skipping")
else:
    NL = chr(10)
    # Add params to __init__
    ds_content = ds_content.replace(
        "        seed: int = 42," + NL + "    ):",
        "        seed: int = 42," + NL +
        "        et_enriched: bool = False," + NL +
        "        enriched_prob: float = 0.5," + NL +
        "    ):"
    )
    # Store new attributes
    ds_content = ds_content.replace(
        "        self.use_text_features = use_text_features" + NL,
        "        self.use_text_features = use_text_features" + NL +
        "        self.et_enriched = et_enriched" + NL +
        "        self.enriched_prob = enriched_prob" + NL
    )
    # Add _load_enriched_text method
    method = (NL +
        '    def _load_enriched_text(self, case_dir, case_name):' + NL +
        '        path = os.path.join(case_dir, f"{case_name}_et_enriched.txt")' + NL +
        '        if os.path.exists(path):' + NL +
        "            with open(path, 'r', encoding='utf-8') as f:" + NL +
        '                return f.read().strip()' + NL +
        '        return None' + NL + NL
    )
    ds_content = ds_content.replace(
        "    def _load_text_features(",
        method + "    def _load_text_features("
    )
    # Replace text loading with stochastic selection
    old_text_load = (
        "        # Load expert text (NO information leakage!)" + NL +
        "        text = self._load_text(case_dir, case_name)"
    )
    new_text_load = (
        "        # Load expert text (NO information leakage!)" + NL +
        "        original_text = self._load_text(case_dir, case_name)" + NL +
        NL +
        "        # LaCLIP stochastic selection" + NL +
        "        if self.et_enriched:" + NL +
        "            enriched = self._load_enriched_text(case_dir, case_name)" + NL +
        "            if self.split == 'train':" + NL +
        "                if enriched and np.random.random() < self.enriched_prob:" + NL +
        "                    text = original_text + ' ' + enriched" + NL +
        "                else:" + NL +
        "                    text = original_text" + NL +
        "            else:" + NL +
        "                text = (original_text + ' ' + enriched) if enriched else original_text" + NL +
        "        else:" + NL +
        "            text = original_text"
    )
    ds_content = ds_content.replace(old_text_load, new_text_load)
    ds_path.write_text(ds_content, encoding='utf-8')
    print("[v4.5-1] Patched ET-enriched stochastic selection into dataset")

# Verify
ds_content = ds_path.read_text(encoding='utf-8')
assert 'et_enriched' in ds_content, "et_enriched param not found!"
assert '_load_enriched_text' in ds_content, "enriched text loader not found!"
print(f"  Dataset patched: {len(ds_content)} chars")
""")

# ---------------------------------------------------------------------------
# Cell 9: [v4.6-4] Config + train.py/evaluate_full.py patches
# ---------------------------------------------------------------------------
cell_9 = code("""\
import pathlib, yaml

# [v4.6-4a] Create configs/textbrats_v8_h100.yaml (H100 80GB optimized)
config = {
    'data': {
        'data_dir': './data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData',
        'dataset_type': 'textbrats',
        'patch_size': [128, 128, 128],
        'batch_size': 8,                   # H100: 4->8 (80GB allows double batch)
        'num_workers': 8,                  # H100: more CPU cores available
        'train_ratio': 0.596,
        'val_ratio': 0.149,
        'et_enriched': True,
        'enriched_prob': 0.5,
    },
    'model': {
        'img_size': [128, 128, 128],
        'in_channels': 4, 'out_channels': 4,
        'embed_dim': 48, 'depths': [2, 2, 2, 2],
        'dropout': 0.1, 'text_embed_dim': 256, 'text_max_len': 192,
        'use_pretrained_text': True, 'unfreeze_text_layers': 2,
        'text_model_path': None,
        # V4.6 features
        'use_cross_scale_skip': True,
        'use_text_gate': True,
        'text_gate_init_bias': 2.0,
    },
    'loss': {
        'dice_weight': 1.0, 'ce_weight': 1.0, 'edge_weight': 1.0,
        'contrastive_weight': 0.05, 'temperature': 0.07,
        'class_weights': [0.25, 3.0, 1.0, 4.0],
    },
    'augmentation': {'use_elastic': True, 'use_modality_dropout': True},
    'training': {
        'epochs': 200, 'lr': 0.0001, 'weight_decay': 0.01,
        'warmup_epochs': 10, 'contrastive_warmup_epochs': 30,
        'patience': 40, 'gradient_accumulation': 1,
        'gradient_checkpointing': False,   # H100: 80GB enough, skip recompute overhead
        'deep_supervision': True,
        'ds_weights': [0.2, 0.1, 0.05], 'use_amp': True,
        'no_text_ratio': 0.15, 'gradient_clip_norm': 1.0,
    },
    'eval': {
        'metrics': ['dice', 'hd95'],
        'sliding_window': True, 'sw_overlap': 0.5,
        'sw_batch_size': 4,                # H100: 2->4 for faster eval
    },
    'experiment': {
        'name': 'TextMamba3D_H100_v4.6_attnres',
        'description': 'V4.6: SeqCA + CrossScaleSkipAttention + TextScaleGate + ET-Enriched (H100 80GB)',
    },
}

pathlib.Path('configs').mkdir(exist_ok=True)
with open('configs/textbrats_v8_h100.yaml', 'w') as f:
    f.write('# configs/textbrats_v8_h100.yaml\\n')
    f.write('# V4.6: AttnRes-inspired Cross-Scale Skip Attention + Text Scale Gate\\n')
    f.write('# H100 80GB optimized (vs A100 40GB version in textbrats_v8.yaml)\\n\\n')
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print("[v4.6-4a] Created configs/textbrats_v8_h100.yaml")

# [v4.6-4b] Patch train.py: forward V4.6 config fields to model constructor
NL = chr(10)
train_path = pathlib.Path('train.py')
train_content = train_path.read_text(encoding='utf-8')

if 'use_text_gate' in train_content:
    print("[v4.6-4b] train.py already patched, skipping")
else:
    # Insert V4.6 config forwarding after dropout line
    old_line = "        dropout=config['model'].get('dropout', 0.0),"
    new_lines = (
        "        dropout=config['model'].get('dropout', 0.0)," + NL +
        "        use_text_gate=config['model'].get('use_text_gate', False)," + NL +
        "        use_cross_scale_skip=config['model'].get('use_cross_scale_skip', False)," + NL +
        "        text_gate_init_bias=config['model'].get('text_gate_init_bias', 2.0),"
    )
    train_content = train_content.replace(old_line, new_lines)
    train_path.write_text(train_content, encoding='utf-8')
    print("[v4.6-4b] Patched train.py with V4.6 config forwarding")

# [v4.6-4c] Patch evaluate_full.py: forward V4.6 config fields
# NOTE: evaluate_full.py uses 'model_cfg' alias, NOT 'config["model"]'
eval_path = pathlib.Path('evaluate_full.py')
eval_content = eval_path.read_text(encoding='utf-8')

if 'use_text_gate' in eval_content:
    print("[v4.6-4c] evaluate_full.py already patched, skipping")
else:
    old_line = "        dropout=model_cfg.get('dropout', 0.0),"
    new_lines = (
        "        dropout=model_cfg.get('dropout', 0.0)," + NL +
        "        use_text_gate=model_cfg.get('use_text_gate', False)," + NL +
        "        use_cross_scale_skip=model_cfg.get('use_cross_scale_skip', False)," + NL +
        "        text_gate_init_bias=model_cfg.get('text_gate_init_bias', 2.0),"
    )
    eval_content = eval_content.replace(old_line, new_lines)
    eval_path.write_text(eval_content, encoding='utf-8')
    print("[v4.6-4c] Patched evaluate_full.py with V4.6 config forwarding")

# Post-patch assertions
train_check = pathlib.Path('train.py').read_text(encoding='utf-8')
eval_check = pathlib.Path('evaluate_full.py').read_text(encoding='utf-8')
assert 'use_text_gate' in train_check, "FATAL: train.py V4.6 patch failed!"
assert 'use_text_gate' in eval_check, "FATAL: evaluate_full.py V4.6 patch failed!"
print("All config + script patches applied and verified!")
""")

# ---------------------------------------------------------------------------
# Cell 10: Patch verification
# ---------------------------------------------------------------------------
cell_10 = code("""\
import sys, importlib
sys.path.insert(0, '.')

# Force reimport after patches (prefix match to avoid clearing transformers.models.*)
for mod_name in list(sys.modules.keys()):
    if mod_name == 'models' or mod_name.startswith('models.'):
        del sys.modules[mod_name]

# Verify V4.6 modules exist
from models.fusion import (
    SequentialCrossAttention, MultiScaleSeqCA,
    RMSNorm, CrossScaleSkipAttention, TextScaleGate, MultiScaleTextGate,
)
print("V4.4 modules: SequentialCrossAttention, MultiScaleSeqCA")
print("V4.6 modules: RMSNorm, CrossScaleSkipAttention, TextScaleGate, MultiScaleTextGate")

# Verify decoder has CrossScaleSkipAttention support
from models.decoder_3d import MambaDecoder3D
import inspect
sig = inspect.signature(MambaDecoder3D.__init__)
assert 'use_cross_scale_skip' in sig.parameters, "decoder missing use_cross_scale_skip param!"
print("Decoder: use_cross_scale_skip parameter present")

# Verify textmamba3d has V4.6 params
from models.textmamba3d import TextMamba3D
sig = inspect.signature(TextMamba3D.__init__)
for param in ['use_text_gate', 'use_cross_scale_skip', 'text_gate_init_bias']:
    assert param in sig.parameters, f"TextMamba3D missing {param}!"
print("TextMamba3D: use_text_gate, use_cross_scale_skip, text_gate_init_bias present")

# Verify config
import yaml
with open('configs/textbrats_v8_h100.yaml') as f:
    cfg = yaml.safe_load(f)
assert cfg['model']['use_cross_scale_skip'] is True
assert cfg['model']['use_text_gate'] is True
assert cfg['data']['batch_size'] == 8, "H100 batch_size should be 8"
assert cfg['training']['gradient_checkpointing'] is False, "H100 should not use grad ckpt"
print("Config: V4.6 features enabled, H100 optimizations confirmed")

print()
print("All V4.6 patches verified!")
""")

# ---------------------------------------------------------------------------
# Cell 11: ET preprocessing
# ---------------------------------------------------------------------------
cell_11 = code("""\
import os, sys
os.chdir(REPO_DIR)

DATA_DIR = "./data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

# Check if already generated
sample_case = sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)))[0]
sample_enriched = os.path.join(DATA_DIR, sample_case, f"{sample_case}_et_enriched.txt")

if os.path.exists(sample_enriched):
    count = sum(
        1 for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
        and os.path.exists(os.path.join(DATA_DIR, d, f"{d}_et_enriched.txt"))
    )
    print(f"ET-enriched text already generated for {count} cases, skipping")
    for case in sorted(os.listdir(DATA_DIR))[:3]:
        path = os.path.join(DATA_DIR, case, f"{case}_et_enriched.txt")
        if os.path.exists(path):
            with open(path, 'r') as f:
                print(f"  {case}: {f.read().strip()[:120]}...")
else:
    print("Generating ET-enriched text descriptions from T1ce images...")
    sys.path.insert(0, '.')
    from data.et_text_enrichment import process_all_cases
    results = process_all_cases(DATA_DIR)

    no_enhancement = sum(1 for desc in results.values() if "No significant" in desc)
    total = len(results)
    print(f"Total: {total}, No enhancement: {no_enhancement} ({no_enhancement/total*100:.1f}%)")
    for name, desc in list(results.items())[:5]:
        print(f"  {name}: {desc}")
""")

# ---------------------------------------------------------------------------
# Cell 12: Training section header
# ---------------------------------------------------------------------------
cell_12 = md("""\
## Training (H100 80GB)

| Parameter | A100 40GB | H100 80GB |
|-----------|-----------|-----------|
| batch_size | 4 | **8** |
| gradient_checkpointing | true | **false** |
| sw_batch_size | 2 | **4** |
| num_workers | 4 | **8** |
| gradient_accumulation | 1 | 1 |
""")

# ---------------------------------------------------------------------------
# Cell 13: Checkpoint sync
# ---------------------------------------------------------------------------
cell_13 = code("""\
import os, shutil, glob

DRIVE_CKPT = os.path.join(DRIVE_BASE, "checkpoints")
os.makedirs(DRIVE_CKPT, exist_ok=True)

def sync_checkpoints_to_drive():
    local_ckpt = os.path.join(REPO_DIR, "checkpoints")
    if not os.path.exists(local_ckpt):
        return
    for f in glob.glob(os.path.join(local_ckpt, "*.pth")):
        dst = os.path.join(DRIVE_CKPT, os.path.basename(f))
        shutil.copy2(f, dst)
    print(f"Synced checkpoints to {DRIVE_CKPT}")

# Clean local checkpoints (fresh start for v4.6)
for f in glob.glob(os.path.join(REPO_DIR, "checkpoints/*.pth")):
    os.remove(f)
print("Cleaned local checkpoints for v4.6 fresh start")
print("(Previous best checkpoints preserved on Drive)")
""")

# ---------------------------------------------------------------------------
# Cell 14: Run training
# ---------------------------------------------------------------------------
cell_14 = code("""\
os.chdir(REPO_DIR)
os.environ["DRIVE_CKPT_DIR"] = DRIVE_CKPT

# V4.6 training: SeqCA + CrossScaleSkipAttention + TextScaleGate + ET-Enriched (H100 80GB)
!python -u train.py \\
    --config configs/textbrats_v8_h100.yaml \\
    --no-text-ratio 0.15 \\
    --grad-accum 1 \\
    2>&1 | tee training_v4.6_h100.log | grep --line-buffered -E "(Epoch [0-9]+:|train_loss=|Best |Error|Traceback)"

# Sync and save
sync_checkpoints_to_drive()

best_ckpt = os.path.join(DRIVE_CKPT, "best_v4.6.pth")
local_best = os.path.join(REPO_DIR, "checkpoints/best.pth")
if os.path.exists(local_best):
    shutil.copy2(local_best, best_ckpt)
    print(f"Best checkpoint saved: {best_ckpt}")
""")

# ---------------------------------------------------------------------------
# Cell 15: Evaluation section header
# ---------------------------------------------------------------------------
cell_15 = md("""\
## Evaluation
""")

# ---------------------------------------------------------------------------
# Cell 16: Full-volume eval
# ---------------------------------------------------------------------------
cell_16 = code("""\
os.chdir(REPO_DIR)

ckpt = os.path.join(REPO_DIR, "checkpoints/best.pth")
if not os.path.exists(ckpt):
    ckpt = os.path.join(DRIVE_CKPT, "best_v4.6.pth")

if os.path.exists(ckpt):
    print("=" * 60)
    print("Evaluation: With Text (SeqCA + TextScaleGate + CrossScaleSkip)")
    print("=" * 60)
    !python evaluate_full.py \\
        --config configs/textbrats_v8_h100.yaml \\
        --checkpoint "{ckpt}" \\
        --split test \\
        --use-text \\
        --overlap 0.5

    print()

    print("=" * 60)
    print("Evaluation: Without Text (fusion bypassed)")
    print("=" * 60)
    !python evaluate_full.py \\
        --config configs/textbrats_v8_h100.yaml \\
        --checkpoint "{ckpt}" \\
        --split test \\
        --no-text \\
        --overlap 0.5

    print()
    print("=" * 60)
    print("Compare: with-text Dice - without-text Dice = text guidance delta")
    print("V4.5 baseline: Mean Dice 83.48%")
    print("V4.6 target: >= 84%")
    print("=" * 60)
else:
    print(f"No checkpoint found at {ckpt}")
    print("Run training first")
""")

# ---------------------------------------------------------------------------
# Cell 17: Results visualization
# ---------------------------------------------------------------------------
cell_17 = code("""\
import matplotlib.pyplot as plt

# Placeholder: fill in actual results after training
v45_dice = {'ET': 0.0, 'TC': 0.0, 'WT': 0.0, 'Mean': 83.48}
v46_dice = {'ET': 0.0, 'TC': 0.0, 'WT': 0.0, 'Mean': 0.0}  # Fill after eval

if v46_dice['Mean'] == 0.0:
    print("V4.6 results not yet filled in.")
    print("Update v45_dice and v46_dice dictionaries after evaluation, then re-run this cell.")
else:
    labels = list(v45_dice.keys())
    v45_vals = list(v45_dice.values())
    v46_vals = list(v46_dice.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart comparison
    x = range(len(labels))
    w = 0.35
    ax1.bar([i - w/2 for i in x], v45_vals, w, label='V4.5', color='steelblue', alpha=0.8)
    ax1.bar([i + w/2 for i in x], v46_vals, w, label='V4.6 (H100)', color='coral', alpha=0.8)
    ax1.set_ylabel('Dice (%)')
    ax1.set_title('V4.5 vs V4.6 Dice Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Delta chart
    deltas = [v46 - v45 for v45, v46 in zip(v45_vals, v46_vals)]
    colors = ['green' if d >= 0 else 'red' for d in deltas]
    ax2.bar(labels, deltas, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Delta (%)')
    ax2.set_title('V4.6 - V4.5 Improvement')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('v46_h100_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: v46_h100_comparison.png")
""")

# ---------------------------------------------------------------------------
# Cell 18: Resume section header
# ---------------------------------------------------------------------------
cell_18 = md("""\
## Resume Training (After Disconnect)
""")

# ---------------------------------------------------------------------------
# Cell 19: Resume
# ---------------------------------------------------------------------------
cell_19 = code("""\
import os, shutil
os.chdir(REPO_DIR)
os.environ["DRIVE_CKPT_DIR"] = DRIVE_CKPT

resume_ckpt = os.path.join(DRIVE_CKPT, "last.pth")
if os.path.exists(resume_ckpt):
    print(f"Resuming from {resume_ckpt}")
    !python train.py \\
        --config configs/textbrats_v8_h100.yaml \\
        --resume "{resume_ckpt}" \\
        --no-text-ratio 0.15 \\
        --grad-accum 1

    sync_checkpoints_to_drive()

    best_ckpt = os.path.join(DRIVE_CKPT, "best_v4.6.pth")
    local_best = os.path.join(REPO_DIR, "checkpoints/best.pth")
    if os.path.exists(local_best):
        shutil.copy2(local_best, best_ckpt)
        print(f"Best checkpoint saved: {best_ckpt}")
else:
    print("No checkpoint to resume from.")
    print(f"Expected: {resume_ckpt}")
    print("Run training first")
""")

# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------
cells = [
    cell_0, cell_1, cell_2, cell_3, cell_4, cell_5, cell_6, cell_7,
    cell_8, cell_9, cell_10, cell_11, cell_12, cell_13, cell_14,
    cell_15, cell_16, cell_17, cell_18, cell_19,
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12",
        },
        "accelerator": "GPU",
        "gpuClass": "premium",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Generated {OUT}")
print(f"  Cells: {len(cells)}")
print(f"  Size: {OUT.stat().st_size / 1024:.1f} KB")
