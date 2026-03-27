"""Generate the Layer 0 AutoResearch Colab notebook.

Reads ``autoresearch/experiments.json`` and produces a ready-to-run
notebook that evaluates all L0 experiments on Colab.

Usage::

    python -m autoresearch.generate_l0
"""
from __future__ import annotations

import json
import os
from itertools import product

from autoresearch.notebook_writer import (
    _code_cell,
    _md_cell,
    create_notebook,
    generate_eval_cell,
    save_notebook,
)

# ----- flag mapping: sweep param name -> CLI flags -----
_SWEEP_FLAG_MAP: dict[str, callable] = {
    "et_wt_ratio": lambda v: ["--advanced-pp", "--et-wt-ratio", str(v)],
    "et_min_size": lambda v: ["--advanced-pp", "--et-min-size", str(v)],
    "wt_min_size": lambda v: ["--advanced-pp", "--wt-min-size", str(v)],
    "et_boost":    lambda v: ["--et-boost", str(v)],
}

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_PATH = os.path.join(HERE, "experiments.json")
OUTPUT_NOTEBOOK = "TextMamba3D_AutoResearch_L0.ipynb"


def _setup_cell() -> dict:
    """生成 Colab 初始化 cell: 挂载 Drive、安装依赖、克隆/拉取代码、解压数据。"""
    lines = [
        "# ===== Setup =====",
        "from google.colab import drive",
        "drive.mount('/content/drive')",
        "",
        "# Install dependencies",
        "!pip install -q mamba-ssm causal-conv1d einops transformers nibabel pyyaml tqdm scipy",
        "",
        "# Clone or pull repo",
        "import os",
        "REPO_DIR = '/content/TextMamba3D'",
        "DRIVE_BASE = '/content/drive/MyDrive/TextMamba3D'",
        "",
        "if os.path.isdir(REPO_DIR):",
        "    !cd {REPO_DIR} && git pull",
        "else:",
        "    !git clone https://github.com/leiyuxuan/TextMamba3D.git {REPO_DIR}",
        "",
        "# Extract BraTS data",
        "DATA_TAR = os.path.join(DRIVE_BASE, 'BraTS2023_data.tar.gz')",
        "DATA_DIR = '/content/BraTS2023'",
        "if not os.path.isdir(DATA_DIR):",
        "    !tar xzf {DATA_TAR} -C /content/",
        "",
        "# Restore ET-enriched text descriptions",
        "TEXT_TAR = os.path.join(DRIVE_BASE, 'et_enriched_text.tar.gz')",
        "if os.path.isfile(TEXT_TAR):",
        "    !tar xzf {TEXT_TAR} -C {REPO_DIR}/",
        "",
        "os.chdir(REPO_DIR)",
        "print('Setup complete. CWD:', os.getcwd())",
    ]
    return _code_cell("\n".join(lines) + "\n", experiment_id="setup")


def _summary_cell() -> dict:
    """生成最终汇总 cell: 读取结果 JSON 并展示排行榜。"""
    lines = [
        "# ===== Summary =====",
        "import json, os",
        "DRIVE_RESULTS = '/content/drive/MyDrive/TextMamba3D/autoresearch_results.json'",
        "",
        "if os.path.exists(DRIVE_RESULTS):",
        "    with open(DRIVE_RESULTS) as f:",
        "        results = json.load(f)",
        "    print(f'Total experiments: {len(results)}')",
        "    print()",
        "    # Sort by dice_mean descending",
        "    ranked = sorted(",
        "        [r for r in results if r.get('metrics', {}).get('dice_mean')],",
        "        key=lambda r: r['metrics']['dice_mean'],",
        "        reverse=True,",
        "    )",
        "    print('Top results:')",
        "    for i, r in enumerate(ranked[:10]):",
        "        m = r['metrics']",
        "        print(",
        "            f\"  {i+1}. {r['experiment_id']}: \"",
        "            f\"dice_mean={m['dice_mean']:.4f}  \"",
        "            f\"dice_ET={m.get('dice_ET', 0):.4f}\"",
        "        )",
        "else:",
        "    print('No results file found.')",
    ]
    return _code_cell("\n".join(lines) + "\n", experiment_id="summary")


def _build_eval_cells_for_sweep(exp: dict) -> list[dict]:
    """为 eval_sweep 类型实验生成 eval cells。"""
    params = exp["params"]
    sweep = params.get("sweep", {})

    # 对 sweep 参数做笛卡尔积
    sweep_keys = sorted(sweep.keys())
    sweep_values = [sweep[k] for k in sweep_keys]

    cells = []
    for combo in product(*sweep_values):
        # 构建 experiment_id 后缀
        suffix_parts = []
        extra_flags: list[str] = []
        for key, val in zip(sweep_keys, combo):
            suffix_parts.append("{k}_{v}".format(k=key, v=val))
            flag_fn = _SWEEP_FLAG_MAP.get(key)
            if flag_fn:
                extra_flags.extend(flag_fn(val))

        eid = "{base}_{suffix}".format(
            base=exp["id"],
            suffix="_".join(suffix_parts),
        )
        cell = generate_eval_cell(
            experiment_id=eid,
            checkpoint=params["checkpoint"],
            config=params["config"],
            split=params.get("split", "test"),
            use_text=params.get("use_text", True),
            tta=params.get("tta", True),
            extra_flags=extra_flags if extra_flags else None,
        )
        cells.append(cell)
    return cells


def _build_eval_cells_for_compare(exp: dict) -> list[dict]:
    """为 eval_compare 类型实验生成 eval cells。"""
    params = exp["params"]
    configs_list = params.get("configs", [])

    cells = []
    for cfg_entry in configs_list:
        name = cfg_entry["name"]
        eid = "{base}_{name}".format(base=exp["id"], name=name)

        extra_flags: list[str] = []
        if cfg_entry.get("advanced_pp"):
            extra_flags.append("--advanced-pp")

        cell = generate_eval_cell(
            experiment_id=eid,
            checkpoint=params["checkpoint"],
            config=params["config"],
            split=params.get("split", "test"),
            use_text=params.get("use_text", True),
            tta=cfg_entry.get("tta", True),
            extra_flags=extra_flags if extra_flags else None,
        )
        cells.append(cell)
    return cells


def generate_l0_notebook(experiments_path: str = EXPERIMENTS_PATH) -> str:
    """Generate the L0 notebook and save it.

    Returns the path to the saved notebook.
    """
    with open(experiments_path) as f:
        data = json.load(f)

    queue = data.get("queue", [])
    l0_experiments = [e for e in queue if e.get("layer") == 0]

    cells: list[dict] = []

    # 1. Setup cell
    cells.append(_setup_cell())

    # 2. Per-experiment cells
    for exp in l0_experiments:
        # Section header
        cells.append(_md_cell(
            "## {eid}: {name}\n{desc}".format(
                eid=exp["id"],
                name=exp["name"],
                desc=exp.get("description", ""),
            )
        ))

        exp_type = exp.get("type", "")
        if exp_type == "eval_sweep":
            cells.extend(_build_eval_cells_for_sweep(exp))
        elif exp_type == "eval_compare":
            cells.extend(_build_eval_cells_for_compare(exp))

    # 3. Summary cell
    cells.append(_md_cell("## Results Summary"))
    cells.append(_summary_cell())

    nb = create_notebook(cells, title="TextMamba3D AutoResearch - Layer 0")
    out_path = os.path.join(os.path.dirname(HERE), OUTPUT_NOTEBOOK)
    return save_notebook(nb, out_path)


if __name__ == "__main__":
    path = generate_l0_notebook()
    print("Generated:", path)
