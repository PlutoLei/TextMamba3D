# TextMamba3D 目录整理设计

**日期:** 2026-03-22
**目标:** 瘦身清理 + 结构重组，同时保护同事的活跃分支

---

## 1. 约束条件

| 约束 | 说明 |
|------|------|
| `keyword-text-experiment` 分支 | 有 2 个未合并 commit，引用 `configs/textbrats.yaml`、`train.py`、`data/brats_textbrats_dataset.py`。不能重命名这些文件 |
| `feat/v4.6-attnres-skip-gate` 分支 | 已完全合并到 main，可安全删除远程分支（需跟同事确认） |
| LFS 文件 (369 个 .npy) | 全在 `data/TextBraTS/`，本次不移动，无风险 |
| `.git/` 3.4GB | Notebook 输出历史导致，本次不做 filter-repo 清洗 |
| Colab notebook 路径 | 全部 `os.chdir('/content/TextMamba3D')`，移动 notebook 不影响 Colab 执行 |

---

## 2. 删除清单

### 2.1 空文件/空目录
- `download.html` (0 字节)
- `TextMamba3D/TextMamba3D/` (嵌套空目录，含空 `.crossfire/`)
- `checkpoints/` 内容 (目录保留，已 gitignored)
- `docs/superpowers/` (整理完成后的空目录)

### 2.2 临时/缓存
- `.crossfire/` (审计遗留 + tmp_mamba3_src 克隆仓库，untracked)
- `.ruff_cache/`
- 所有 `__pycache__/` 目录 (6+ 处)

### 2.3 过时文件
- `nexus_state.json` (已 gitignored)
- `GIT_PULL_GUIDE.md` (3 行内容并入 README)
- `docs/generate_report_pdf.py` (一次性脚本)
- `scripts/generate_v43_notebook.py` (V4.3 notebook 生成器，已废弃)

### 2.4 V4.3 死代码 (losses/)

**注意：** `contrastive_loss.py` 经审计确认为**活跃代码**，被 `losses/__init__.py` 的 `CombinedLoss` 类和 `tests/test_losses.py` 直接引用，**不得删除**。

经审计确认：上述 4 个文件（`embedding_perturbation.py`、`pwam.py`、`text_necessity_loss.py`、`text_voxel_loss.py`）**不存在于磁盘上**。它们仅在 Colab 运行时由 `scripts/generate_v43_notebook.py` 动态创建。因此无需从 `losses/` 删除文件。

`losses/__init__.py` 无需修改 — 它只 import `contrastive_loss`、`dice_loss`、`edge_loss`。

需要删除的 V4.3 遗留是 `scripts/generate_v43_notebook.py`（notebook 生成脚本）。

保留：
- `contrastive_loss.py` (**活跃使用**)
- `dice_loss.py`
- `edge_loss.py`
- `__init__.py`

---

## 3. Notebook 归档

**根目录保留 3 个：**
- `TextMamba3D_A100_V5.1.ipynb` (最新训练，Mamba3)
- `TextMamba3D_eval_unified.ipynb` (统一评估)
- `TextMamba3D_Ablation_TC.ipynb` (消融实验)

**归档到 `notebooks/archive/`：**

| 源文件 | 目标 |
|--------|------|
| `TextMamba3D_A100.ipynb` | `notebooks/archive/v4.0/` |
| `TextMamba3D_A100_V4.1.ipynb` | `notebooks/archive/v4.1/` |
| `TextMamba3D_A100_V4.2.ipynb` | `notebooks/archive/v4.2/` |
| `TextMamba3D_A100_V4.3.ipynb` | `notebooks/archive/v4.3/` |
| `TextMamba3D_A100_V4.4.ipynb` | `notebooks/archive/v4.4/` |
| `TextMamba3D_A100_V5.0.ipynb` | `notebooks/archive/v5.0/` |
| `TextMamba3D_H100_V4.6.ipynb` | `notebooks/archive/v4.6/` |
| `TextMamba3D_Mamba3_Test.ipynb` | `notebooks/archive/misc/` |

**已有 notebooks/ 子目录处理：**
- `notebooks/v4.5/` → 移入 `notebooks/archive/v4.5/`
- `notebooks/v4.6/` → 移入 `notebooks/archive/v4.6/` (与根目录的 H100 合并)

**文件名冲突处理：**
根目录的 `TextMamba3D_H100_V4.6.ipynb` (64KB) 与 `notebooks/v4.6/TextMamba3D_H100_V4.6.ipynb` (20KB) 同名。根目录版本更新（含最新训练输出），保留根目录版本，删除 `notebooks/v4.6/` 下的旧副本。

**归档 notebook 说明：**
归档 notebook 内的 config 路径 (`--config configs/textbrats_a100.yaml` 等) 不做更新。归档 notebook 是历史快照，不保证可直接运行；如需复现，需手动将 config 路径更新为当前活跃 config 或 `configs/archive/` 下对应版本。

---

## 4. Config 重组

**保持原名（保护 keyword 分支）：**
- `textbrats.yaml` — 不重命名

**重命名（安全，无分支引用）：**

| 原名 | 新名 |
|------|------|
| `textbrats_a100.yaml` | `a100.yaml` |
| `textbrats_a100_v5.1.yaml` | `a100_v5.1.yaml` |
| `textbrats_v8_h100.yaml` | `h100.yaml` |

**保留不动：**
- `default.yaml`

**归档到 `configs/archive/`：**
- `textbrats_v5.yaml` (V4.3 实验)
- `textbrats_v7.yaml` (V4.5)
- `textbrats_v8.yaml` (V4.6 A100)
- `textbrats_a100_v5.yaml` (V5.0)

---

## 5. Docs 重组

**MACBOOK_GUIDE.md 处理：** 用 `git mv` 从根目录移至 `docs/guides/COLAB_GUIDE.md`（文件名更准确地反映其内容：Colab 使用指南）。根目录不保留副本。

```
docs/
├── guides/
│   ├── USAGE.md
│   ├── REPRODUCTION_GUIDE.md
│   ├── COLAB_GUIDE.md              ← 原根目录 MACBOOK_GUIDE.md
│   └── 训练指南.md
├── dev/
│   ├── architecture_evolution_v1_to_v4.md
│   ├── experiment_log.md
│   ├── v44_changelog.md
│   ├── mamba3_integration_lessons.md
│   └── archive/
│       ├── architecture_proposal_step4.md
│       ├── architecture_v2_postmortem.md
│       ├── improvement_plan.md
│       ├── text_guidance_improvement_plan.md
│       ├── research_summary_step4.md
│       └── v44_crossfire_audit.md
├── Papers/                          ← 不动
├── planning/                        ← 不动
└── architecture.png                 ← 保留
```

---

## 6. 硬编码修复

### 6.1 Config 重命名引发

| 文件 | 修改 |
|------|------|
| `TextMamba3D_A100_V5.1.ipynb` | `--config configs/textbrats_a100_v5.1.yaml` → `configs/a100_v5.1.yaml` |
| `TextMamba3D_Ablation_TC.ipynb` | 检查 config 引用，如指向已归档的 config 则更新路径为 `configs/archive/...` |
| `TextMamba3D_eval_unified.ipynb` | 同上 |
| `evaluate_full.py:25` | default 从 `configs/textbrats_a100.yaml` → `configs/a100.yaml` |
| `scripts/quick_train.sh` | 不改（仍用 `configs/textbrats.yaml`，未重命名） |
| `scripts/quick_train.bat` | 不改 |
| `scripts/quick_eval.sh` | 不改 |
| `scripts/quick_eval.bat` | 不改 |

### 6.2 losses/__init__.py

移除对已删除 V4.3 模块的 import，只保留 dice_loss 和 edge_loss。

### 6.3 .gitignore 补充

新增：
```
.ruff_cache/
.crossfire/
```

---

## 7. Git 提交策略

三个原子提交，便于独立 revert：

| 顺序 | 提交 | 内容 |
|------|------|------|
| 1 | `chore: clean up dead code, caches, and empty files` | 删除空文件、缓存、`.crossfire/`、V4.3 loss 死代码、更新 `.gitignore` 和 `losses/__init__.py` |
| 2 | `refactor: reorganize notebooks, configs, and docs` | Notebook 归档、Config 重命名+归档、Docs 分类、MACBOOK_GUIDE 移动并重命名 |
| 3 | `fix: update hardcoded config paths in notebooks and scripts` | 更新 notebook/scripts/Python 中因重命名而失效的路径引用 |

所有文件移动使用 `git mv` 保留历史追溯。

3 个 commit 本地验证后一次性 push 到 GitHub。

### 远程分支清理（push 之后，需同事确认后执行）

执行顺序：先 push 3 个整理 commit，再清理远程分支。

- `feat/v4.6-attnres-skip-gate` — 已完全合并到 main，确认后执行 `git push origin --delete feat/v4.6-attnres-skip-gate`
- `keyword-text-experiment` — 保留，有未合并 commit

---

## 8. 最终目录结构

```
TextMamba3D/
├── train.py
├── evaluate_full.py
├── inference.py
├── preprocess_data.py
├── smoke_test.py
├── README.md
├── README_CN.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .gitattributes
│
├── TextMamba3D_A100_V5.1.ipynb
├── TextMamba3D_eval_unified.ipynb
├── TextMamba3D_Ablation_TC.ipynb
│
├── configs/
│   ├── default.yaml
│   ├── textbrats.yaml               ← 保留原名
│   ├── a100.yaml
│   ├── a100_v5.1.yaml
│   ├── h100.yaml
│   └── archive/
│       ├── textbrats_v5.yaml
│       ├── textbrats_v7.yaml
│       ├── textbrats_v8.yaml
│       └── textbrats_a100_v5.yaml
│
├── models/
├── losses/
│   ├── __init__.py
│   ├── contrastive_loss.py          ← 活跃使用，保留
│   ├── dice_loss.py
│   └── edge_loss.py
│
├── data/
├── utils/
├── tests/
├── pretrained/
├── checkpoints/
│
├── scripts/
│
├── notebooks/
│   └── archive/
│       ├── v4.0/
│       ├── v4.1/
│       ├── v4.2/
│       ├── v4.3/
│       ├── v4.4/
│       ├── v4.5/
│       ├── v4.6/
│       ├── v5.0/
│       └── misc/
│
└── docs/
    ├── guides/
    ├── dev/
    │   └── archive/
    ├── Papers/
    ├── planning/
    └── architecture.png
```
