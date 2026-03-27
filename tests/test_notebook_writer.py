"""Tests for autoresearch.notebook_writer module."""
from autoresearch.notebook_writer import (
    _code_cell,
    _md_cell,
    create_notebook,
    generate_eval_cell,
    generate_train_cell,
    save_notebook,
)


# --------------- _code_cell / _md_cell ---------------

def test_code_cell_structure():
    cell = _code_cell("print('hello')", "exp-1")
    assert cell["cell_type"] == "code"
    assert cell["execution_count"] is None
    assert cell["metadata"]["experiment_id"] == "exp-1"
    assert isinstance(cell["outputs"], list)
    assert "".join(cell["source"]).strip() == "print('hello')"


def test_md_cell_structure():
    cell = _md_cell("# Title")
    assert cell["cell_type"] == "markdown"
    assert "".join(cell["source"]).strip() == "# Title"


# --------------- generate_eval_cell ---------------

def test_generate_eval_cell_returns_valid_cell():
    cell = generate_eval_cell(
        experiment_id="L0-1_et_wt_ratio_0.03",
        checkpoint="best_v5.0.pth",
        config="configs/archive/textbrats_a100_v5.yaml",
        split="test",
        use_text=True,
        tta=True,
        extra_flags=["--advanced-pp", "--et-wt-ratio", "0.03"],
    )
    assert cell["cell_type"] == "code"
    assert cell["metadata"]["experiment_id"] == "L0-1_et_wt_ratio_0.03"
    src = "".join(cell["source"])
    assert "evaluate_full.py" in src
    assert "best_v5.0.pth" in src
    assert "--advanced-pp" in src
    assert "--et-wt-ratio" in src
    assert "autoresearch_results.json" in src
    assert "L0-1_et_wt_ratio_0.03" in src


def test_generate_eval_cell_no_tta_no_text():
    cell = generate_eval_cell(
        experiment_id="test-no-flags",
        checkpoint="ckpt.pth",
        config="configs/test.yaml",
        use_text=False,
        tta=False,
    )
    src = "".join(cell["source"])
    assert "'--use-text'" not in src
    assert "'--tta'" not in src


def test_generate_eval_cell_parses_metrics():
    """Eval cell source must contain metric parsing logic."""
    cell = generate_eval_cell(
        experiment_id="parse-test",
        checkpoint="ckpt.pth",
        config="configs/test.yaml",
    )
    src = "".join(cell["source"])
    assert "dice_ET" in src
    assert "dice_mean" in src
    assert "re.search" in src


# --------------- generate_train_cell ---------------

def test_generate_train_cell():
    cell = generate_train_cell(
        experiment_id="L1-lr001",
        config_path="configs/l1_lr001.yaml",
        resume_checkpoint="best_v5.0.pth",
        extra_args=["--epochs", "200"],
    )
    assert cell["cell_type"] == "code"
    src = "".join(cell["source"])
    assert "!python -u train.py" in src
    assert "configs/l1_lr001.yaml" in src
    assert "best_v5.0.pth" in src
    assert "--epochs" in src


def test_generate_train_cell_no_resume():
    cell = generate_train_cell(
        experiment_id="L1-base",
        config_path="configs/base.yaml",
    )
    src = "".join(cell["source"])
    assert "--resume" not in src


# --------------- create_notebook ---------------

def test_create_notebook_structure():
    cells = [_code_cell("x = 1")]
    nb = create_notebook(cells, title="Test Notebook")
    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] == 5
    assert "kernelspec" in nb["metadata"]
    # title cell + 1 code cell
    assert len(nb["cells"]) == 2
    assert nb["cells"][0]["cell_type"] == "markdown"
    assert "Test Notebook" in "".join(nb["cells"][0]["source"])
    assert nb["cells"][1]["cell_type"] == "code"


def test_create_notebook_default_title():
    nb = create_notebook([])
    assert "AutoResearch" in "".join(nb["cells"][0]["source"])


# --------------- save_notebook ---------------

def test_save_notebook(tmp_path):
    nb = create_notebook([_code_cell("pass")])
    out = save_notebook(nb, str(tmp_path / "test.ipynb"))
    assert out.endswith("test.ipynb")
    import json
    with open(out) as f:
        loaded = json.load(f)
    assert loaded["nbformat"] == 4
    assert len(loaded["cells"]) == 2
