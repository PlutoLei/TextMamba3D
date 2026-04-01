# TextMamba3D 项目整理设计文档

**日期:** 2026-04-01
**目标:** 结构化归档 + 清理，兼顾开源发布、开发效率、里程碑归档三个目标

---

## 1. 整理概览

| 操作 | 预期效果 |
|------|----------|
| 删除 `archive.zip` | 释放 12GB 空间 |
| 清理缓存目录 | 移除 `__pycache__/`、`.pytest_cache/` |
| 归档 10 个 notebook | 根目录条目从 ~41 降到 ~20 |
| 重组 `docs/` | 从 6 个子目录合并为 4 个 |
| 移动临时文件 | `task_plan.md`、`progress.md`、`findings.md` 归入 docs |
| 更新 `.gitignore` | 防止大文件和缓存再次被追踪 |

## 2. 根目录清理

### 删除

- `archive.zip` — BraTS2021 原始下载包（12GB），`data/BraTS2021/` 已有解压数据
- `__pycache__/` — Python 编译缓存
- `.pytest_cache/` — pytest 缓存

### 移入 `notebooks/experiments/`

| Notebook | 状态 | 说明 |
|----------|------|------|
| TextMamba3D_A100_V5.1.ipynb | WIP-放弃 | Mamba-3 complex SSM |
| TextMamba3D_A100_V5.2.ipynb | WIP-放弃 | ET 后处理优化 |
| TextMamba3D_A100_V5.3.ipynb | WIP-放弃 | Copy-Paste 增强 |
| TextMamba3D_Ablation_TC.ipynb | 完成 | TC 回归诊断 |
| TextMamba3D_AutoResearch_L0.ipynb | 完成 | 推理时实验 |
| TextMamba3D_AutoResearch_L1.ipynb | 完成 | 训练实验 |
| TextMamba3D_Ensemble_V50_V60.ipynb | 完成 | 2模型集成 |
| TextMamba3D_Ensemble_V50_V60_V70.ipynb | 完成 | 3模型集成 (最佳 0.8511) |
| TextMamba3D_V6.0_BottleneckSeqCA.ipynb | 完成 | 瓶颈融合实验 |
| TextMamba3D_V7.0.ipynb | 完成 | Boundary + Hierarchy Loss |

### 保留在根目录

- `TextMamba3D_V8.0.ipynb` — 当前活跃实验
- `TextMamba3D_eval_unified.ipynb` — 通用评估工具
- `train.py`, `inference.py`, `evaluate_full.py`, `preprocess_data.py`, `smoke_test.py`
- `README.md`, `README_CN.md`, `LICENSE`, `requirements.txt`
- 所有子目录：`configs/`, `data/`, `docs/`, `losses/`, `models/`, `notebooks/`, `scripts/`, `tests/`, `utils/`, `autoresearch/`

### 移入 `docs/dev/`

- `task_plan.md` — 临时计划文件
- `progress.md` — 进度记录
- `findings.md` — 发现记录

## 3. docs/ 目录重组

### 整理后结构（4 个目录）

```
docs/
├── papers/       # 不变：19篇论文笔记
├── guides/       # 不变：使用指南
├── dev/          # 合并所有开发过程记录
│   ├── (原 dev/ 全部内容保留)
│   ├── task_plan.md              ← 原根目录
│   ├── progress.md               ← 原根目录
│   ├── findings.md               ← 原根目录
│   ├── mamba_text_fusion_survey_2025.md  ← 原 research/
│   └── strategy_cross_evaluation.md      ← 原 research/
├── specs/        # 合并所有规划/设计文档
│   ├── architecture_proposal.md           ← 原 planning/
│   ├── brainstorming_a100_optimization.md ← 原 planning/
│   ├── dev_plan.md                        ← 原 planning/
│   ├── progress.md                        ← 原 planning/
│   ├── research_summary.md                ← 原 planning/
│   ├── research_summary_original.md       ← 原 planning/
│   ├── V9.0_plan.md                       ← 原 planning/
│   ├── 2026-01-22-textmamba3d-implementation.md  ← 原 plans/
│   ├── (原 superpowers/specs/ 全部内容)
│   └── (原 superpowers/plans/ 全部内容)
├── architecture.png
├── TextMamba3D_Analysis_Report.pdf
└── TextMamba3D_Reading_List.pdf
```

### 删除的目录

- `docs/planning/` — 内容迁入 specs/
- `docs/plans/` — 内容迁入 specs/
- `docs/research/` — 内容迁入 dev/
- `docs/superpowers/` — 内容迁入 specs/

## 4. .gitignore 更新

```gitignore
# Python
__pycache__/
*.pyc
*.pyo

# Environment
.env
.venv/
venv/

# Data (large files)
data/BraTS*/
data/TextBraTS/
archive.zip
TextBraTS_data.zip

# Model weights
checkpoints/
pretrained/
*.pt
*.pth

# Logs
logs/
training_log*.txt
*.log

# IDE & tools
.DS_Store
.idea/
.pytest_cache/
.ruff_cache/
.crossfire/
.claude/
nexus_state.json

# Jupyter
.ipynb_checkpoints/

# Docs (generated/large)
docs/run_*/
docs/architecture_prompt.txt
docs/papers/**/*.pdf
```

变更点：
- 新增 `archive.zip`、`.pytest_cache/`、`data/TextBraTS/`、`.ipynb_checkpoints/`
- 移除 `docs/plans/`（该目录已合并到 specs/，新目录需要被追踪）

## 5. 整理后根目录预览

```
TextMamba3D/
├── .git/
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
├── README_CN.md
├── requirements.txt
├── TextMamba3D_V8.0.ipynb
├── TextMamba3D_eval_unified.ipynb
├── train.py
├── inference.py
├── evaluate_full.py
├── preprocess_data.py
├── smoke_test.py
├── autoresearch/
├── configs/
├── data/
├── docs/
├── losses/
├── models/
├── notebooks/
├── scripts/
├── tests/
└── utils/
```

## 6. 执行策略

分 4 个 commit 执行，每步可独立回滚：

1. **`chore: delete archive.zip and clean caches`** — 删除 archive.zip、__pycache__、.pytest_cache
2. **`chore: archive experiment notebooks to notebooks/experiments/`** — 移动 10 个 notebook
3. **`chore: reorganize docs/ structure`** — docs 重组 + 移动根目录临时文件
4. **`chore: update .gitignore`** — 更新忽略规则
