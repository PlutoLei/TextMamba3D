"""Generate Colab notebooks for AutoResearch experiments."""
from __future__ import annotations

import json
import os
from typing import Any


def _code_cell(source: str, experiment_id: str = "") -> dict:
    """Create a notebook code cell dict."""
    metadata: dict[str, Any] = {}
    if experiment_id:
        metadata["experiment_id"] = experiment_id
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _md_cell(source: str) -> dict:
    """Create a notebook markdown cell dict."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def generate_eval_cell(
    experiment_id: str,
    checkpoint: str,
    config: str,
    split: str = "test",
    use_text: bool = True,
    tta: bool = True,
    extra_flags: list[str] | None = None,
) -> dict:
    """Generate a code cell that runs evaluate_full.py and records metrics.

    Parameters
    ----------
    experiment_id : str
        Unique experiment identifier (e.g. ``L0-1_et_wt_ratio_0.03``).
    checkpoint : str
        Checkpoint filename, looked up under DRIVE_CKPT.
    config : str
        YAML config path relative to repo root.
    split : str
        Dataset split (``val`` or ``test``).
    use_text : bool
        Whether to pass ``--use-text``.
    tta : bool
        Whether to pass ``--tta``.
    extra_flags : list[str] | None
        Additional CLI flags forwarded verbatim to evaluate_full.py.
    """
    flags_parts: list[str] = []
    if use_text:
        flags_parts.append("'--use-text'")
    if tta:
        flags_parts.append("'--tta'")
    if extra_flags:
        for f in extra_flags:
            flags_parts.append(repr(f))
    flags_literal = ", ".join(flags_parts)

    # 构建 cell source —— 用列表拼接避免 f-string 内含 \n
    lines = [
        "# --- Eval: {eid} ---".format(eid=experiment_id),
        "import subprocess, json, re, os, pathlib",
        "",
        "DRIVE_CKPT = '/content/drive/MyDrive/TextMamba3D/checkpoints'",
        "DRIVE_RESULTS = '/content/drive/MyDrive/TextMamba3D/autoresearch_results.json'",
        "REPO_DIR = '/content/TextMamba3D'",
        "",
        "ckpt_path = os.path.join(DRIVE_CKPT, '{ckpt}')".format(ckpt=checkpoint),
        "assert os.path.isfile(ckpt_path), f'Checkpoint not found: {{ckpt_path}}'",
        "",
        "cmd = [",
        "    'python', 'evaluate_full.py',",
        "    '--config', '{cfg}',".format(cfg=config),
        "    '--checkpoint', ckpt_path,",
        "    '--split', '{sp}',".format(sp=split),
        "    {flags}".format(flags=flags_literal),
        "]",
        "print('Running:', ' '.join(cmd))",
        "",
        "result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_DIR)",
        "output = result.stdout",
        "print(output[-1500:])",
        "if result.returncode != 0:",
        "    print('STDERR:', result.stderr[-1000:])",
        "",
        "# Parse metrics",
        "metrics = {}",
        "for key in ['dice_ET', 'dice_TC', 'dice_WT', 'dice_mean',",
        "            'hd95_ET', 'hd95_TC', 'hd95_WT']:",
        "    m = re.search(rf'{key}: ([\\d.]+) \\+/- ([\\d.]+)', output)",  # noqa: W605
        "    if m:",
        "        metrics[key] = float(m.group(1))",
        "        metrics[key + '_std'] = float(m.group(2))",
        "",
        "record = {",
        "    'experiment_id': '{eid}',".format(eid=experiment_id),
        "    'checkpoint': '{ckpt}',".format(ckpt=checkpoint),
        "    'config': '{cfg}',".format(cfg=config),
        "    'split': '{sp}',".format(sp=split),
        "    'metrics': metrics,",
        "    'returncode': result.returncode,",
        "}",
        "",
        "# Append to Drive results file",
        "if os.path.exists(DRIVE_RESULTS):",
        "    with open(DRIVE_RESULTS) as f:",
        "        all_results = json.load(f)",
        "else:",
        "    all_results = []",
        "all_results.append(record)",
        "with open(DRIVE_RESULTS, 'w') as f:",
        "    json.dump(all_results, f, indent=2)",
        "",
        "print()",
        "print('Metrics:', metrics)",
    ]
    source = "\n".join(lines) + "\n"
    return _code_cell(source, experiment_id)


def generate_train_cell(
    experiment_id: str,
    config_path: str,
    resume_checkpoint: str | None = None,
    extra_args: list[str] | None = None,
) -> dict:
    """Generate a code cell that trains the model via ``!python -u train.py``.

    Uses Jupyter magic ``!`` so output streams live into the notebook.
    """
    parts = ["!python -u train.py --config {cfg}".format(cfg=config_path)]
    if resume_checkpoint:
        parts.append("  --resume {ckpt}".format(ckpt=resume_checkpoint))
    if extra_args:
        for a in extra_args:
            parts.append("  " + a)
    # join with backslash-continuation for readability
    source_cmd = " \\\n".join(parts)

    lines = [
        "# --- Train: {eid} ---".format(eid=experiment_id),
        "import os",
        "os.chdir('/content/TextMamba3D')",
        "",
        source_cmd,
    ]
    source = "\n".join(lines) + "\n"
    return _code_cell(source, experiment_id)


def create_notebook(cells: list[dict], title: str = "AutoResearch") -> dict:
    """Wrap cells into a complete nbformat v4 notebook dict."""
    title_cell = _md_cell("# " + title)
    all_cells = [title_cell] + cells
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
            "colab": {
                "provenance": [],
            },
        },
        "cells": all_cells,
    }


def save_notebook(nb: dict, path: str) -> str:
    """Write notebook dict to a .ipynb file.

    Returns the absolute path of the saved file.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    return os.path.abspath(path)
