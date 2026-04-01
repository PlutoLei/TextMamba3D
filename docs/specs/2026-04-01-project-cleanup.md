# TextMamba3D 项目整理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 结构化整理 TextMamba3D 项目，清理 12GB 冗余文件、归档实验 notebook、重组 docs 目录、更新 .gitignore

**Architecture:** 纯文件操作（移动/删除/编辑），不涉及代码逻辑变更。分 4 个独立 commit，每步可回滚。

**Tech Stack:** git, bash (rm, mv, mkdir)

**Spec:** `docs/specs/2026-04-01-project-cleanup-design.md`

---

### Task 1: 删除大文件与清理缓存

**Files:**
- Delete: `archive.zip` (12GB)
- Delete: `__pycache__/` (根目录)
- Delete: `losses/__pycache__/`
- Delete: `data/__pycache__/`
- Delete: `models/__pycache__/`
- Delete: `utils/__pycache__/`
- Delete: `autoresearch/__pycache__/`
- Delete: `tests/__pycache__/`
- Delete: `.pytest_cache/`

- [ ] **Step 1: 确认 archive.zip 内容**

```bash
cd /Users/leiyuxuan/Documents/PyCharm_Project/TextMamba3D
unzip -l archive.zip | tail -5
```

Expected: 3 files, ~12.5GB total (BraTS2021 training data)

- [ ] **Step 2: 确认 data/BraTS2021/ 已完整解压**

```bash
ls data/BraTS2021/ | wc -l
```

Expected: 1255 (samples exist, safe to delete zip)

- [ ] **Step 3: 删除 archive.zip**

```bash
rm archive.zip
```

- [ ] **Step 4: 清理所有 __pycache__ 和 .pytest_cache**

```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
rm -rf .pytest_cache
```

- [ ] **Step 5: 验证清理结果**

```bash
find . -name "__pycache__" -o -name ".pytest_cache" | head -5
du -sh .
```

Expected: 无 pycache/pytest_cache 输出，总大小从 ~25GB 降到 ~13GB

- [ ] **Step 6: 从 git 追踪中移除缓存文件并提交**

```bash
git rm -r --cached __pycache__/ losses/__pycache__/ data/__pycache__/ models/__pycache__/ utils/__pycache__/ autoresearch/__pycache__/ tests/__pycache__/ .pytest_cache/ 2>/dev/null
git add -A __pycache__/ losses/__pycache__/ data/__pycache__/ models/__pycache__/ utils/__pycache__/ autoresearch/__pycache__/ tests/__pycache__/ .pytest_cache/
git commit -m "chore: delete archive.zip and clean caches

Remove 12GB BraTS2021 zip (data already extracted to data/BraTS2021/).
Remove all __pycache__ and .pytest_cache directories."
```

---

### Task 2: 归档实验 Notebook

**Files:**
- Create directory: `notebooks/experiments/`
- Move 10 notebooks from root → `notebooks/experiments/`

- [ ] **Step 1: 创建 experiments 目录**

```bash
cd /Users/leiyuxuan/Documents/PyCharm_Project/TextMamba3D
mkdir -p notebooks/experiments
```

- [ ] **Step 2: 移动 10 个 notebook**

```bash
git mv TextMamba3D_A100_V5.1.ipynb notebooks/experiments/
git mv TextMamba3D_A100_V5.2.ipynb notebooks/experiments/
git mv TextMamba3D_A100_V5.3.ipynb notebooks/experiments/
git mv TextMamba3D_Ablation_TC.ipynb notebooks/experiments/
git mv TextMamba3D_AutoResearch_L0.ipynb notebooks/experiments/
git mv TextMamba3D_AutoResearch_L1.ipynb notebooks/experiments/
git mv TextMamba3D_Ensemble_V50_V60.ipynb notebooks/experiments/
git mv TextMamba3D_Ensemble_V50_V60_V70.ipynb notebooks/experiments/
git mv TextMamba3D_V6.0_BottleneckSeqCA.ipynb notebooks/experiments/
git mv TextMamba3D_V7.0.ipynb notebooks/experiments/
```

- [ ] **Step 3: 验证根目录只剩 2 个 notebook**

```bash
ls *.ipynb
```

Expected: `TextMamba3D_V8.0.ipynb  TextMamba3D_eval_unified.ipynb`

- [ ] **Step 4: 验证 experiments 目录有 10 个文件**

```bash
ls notebooks/experiments/ | wc -l
```

Expected: 10

- [ ] **Step 5: 提交**

```bash
git add notebooks/experiments/
git commit -m "chore: archive 10 experiment notebooks to notebooks/experiments/

Move completed and abandoned experiment notebooks out of root.
Keep only V8.0 (active) and eval_unified (tool) in root."
```

---

### Task 3: 重组 docs/ 目录

**Files:**
- Move: `docs/research/*.md` → `docs/dev/`
- Move: `docs/planning/*.md` → `docs/specs/`
- Move: `docs/plans/*.md` → `docs/specs/`
- Move: `docs/superpowers/specs/*.md` → `docs/specs/`
- Move: `docs/superpowers/plans/*.md` → `docs/specs/`
- Move: root `task_plan.md`, `progress.md`, `findings.md` → `docs/dev/`
- Create directory: `docs/specs/`
- Delete empty directories: `docs/planning/`, `docs/plans/`, `docs/research/`, `docs/superpowers/`

- [ ] **Step 1: 创建 docs/specs/ 目录**

```bash
cd /Users/leiyuxuan/Documents/PyCharm_Project/TextMamba3D
mkdir -p docs/specs
```

- [ ] **Step 2: 移动 research/ → dev/**

```bash
git mv docs/research/mamba_text_fusion_survey_2025.md docs/dev/
git mv docs/research/strategy_cross_evaluation.md docs/dev/
```

- [ ] **Step 3: 移动 planning/ → specs/**

```bash
git mv docs/planning/architecture_proposal.md docs/specs/
git mv docs/planning/brainstorming_a100_optimization.md docs/specs/
git mv docs/planning/dev_plan.md docs/specs/
git mv docs/planning/progress.md docs/specs/planning_progress.md
git mv docs/planning/research_summary.md docs/specs/
git mv docs/planning/research_summary_original.md docs/specs/
git mv docs/planning/V9.0_plan.md docs/specs/
```

注意：`planning/progress.md` 重命名为 `planning_progress.md` 以避免与 `docs/dev/progress.md`（从根目录迁入）命名冲突。

- [ ] **Step 4: 移动 plans/ → specs/**

注意：`docs/plans/` 在旧 `.gitignore` 中被忽略，该文件可能从未被 git 追踪。使用 `git add -f` 后再移动。

```bash
cp docs/plans/2026-01-22-textmamba3d-implementation.md docs/specs/
git add -f docs/specs/2026-01-22-textmamba3d-implementation.md
rm -rf docs/plans/
```

- [ ] **Step 5: 移动 superpowers/specs/ 和 superpowers/plans/ → specs/**

```bash
git mv docs/superpowers/specs/2026-03-22-directory-reorganization-design.md docs/specs/
git mv docs/superpowers/specs/2026-03-23-et-improvement-v5.2-design.md docs/specs/
git mv docs/superpowers/specs/2026-03-27-autoresearch-design.md docs/specs/
git mv docs/superpowers/specs/2026-03-30-v7-boundary-hierarchy-design.md docs/specs/
git mv docs/superpowers/specs/2026-04-01-project-cleanup-design.md docs/specs/
git mv docs/superpowers/plans/2026-03-18-textmamba3d-v46-attnres.md docs/specs/
git mv docs/superpowers/plans/2026-03-18-v46-eval-gate-logging.md docs/specs/
git mv docs/superpowers/plans/2026-03-19-mamba3-integration.md docs/specs/
git mv docs/superpowers/plans/2026-03-22-directory-reorganization.md docs/specs/
git mv docs/superpowers/plans/2026-03-23-et-improvement-v5.2.md docs/specs/
git mv docs/superpowers/plans/2026-03-24-text-aware-copy-paste.md docs/specs/
git mv docs/superpowers/plans/2026-03-27-autoresearch-implementation.md docs/specs/
git mv docs/superpowers/plans/2026-03-30-v7-boundary-hierarchy.md docs/specs/
git mv docs/superpowers/plans/2026-03-30-v8-pretrain-and-ensemble.md docs/specs/
```

- [ ] **Step 6: 移动本计划文件到 specs/**

```bash
git mv docs/superpowers/plans/2026-04-01-project-cleanup.md docs/specs/
```

- [ ] **Step 7: 移动根目录临时文件 → dev/**

```bash
git mv task_plan.md docs/dev/
git mv progress.md docs/dev/
git mv findings.md docs/dev/
```

- [ ] **Step 8: 删除空的旧目录**

```bash
rm -rf docs/planning docs/research docs/superpowers
```

（`docs/plans/` 已在 Step 4 中删除）

- [ ] **Step 9: 修复 docs 内部交叉引用**

移动文件后，部分 markdown 文件内的相对路径引用会失效。扫描并修复：

```bash
cd /Users/leiyuxuan/Documents/PyCharm_Project/TextMamba3D
# 查找所有指向旧路径的引用
grep -rn "docs/planning\|docs/plans\|docs/research\|docs/superpowers" docs/ --include="*.md"
```

对每个命中项，更新路径映射：
- `docs/planning/` → `docs/specs/`
- `docs/plans/` → `docs/specs/`
- `docs/research/` → `docs/dev/`
- `docs/superpowers/specs/` → `docs/specs/`
- `docs/superpowers/plans/` → `docs/specs/`

如果无命中则跳过。

- [ ] **Step 10: 清理 .DS_Store 文件**

```bash
find . -name .DS_Store -delete
```

- [ ] **Step 11: 验证 docs/ 结构**

```bash
ls docs/
ls docs/dev/ | wc -l
ls docs/specs/ | wc -l
```

Expected:
- `docs/` 包含: `architecture.png`, `TextMamba3D_Analysis_Report.pdf`, `TextMamba3D_Reading_List.pdf`, `dev/`, `guides/`, `papers/`, `specs/`
- `docs/dev/` 约 14 个条目（原 7 + archive/ + 根目录 3 + research 2）
- `docs/specs/` 约 21 个文件（planning 7 + plans 1 + superpowers/specs 5 + superpowers/plans 10）-- 注意含本计划文件

- [ ] **Step 12: 提交**

```bash
git add -A docs/ task_plan.md progress.md findings.md
git commit -m "chore: reorganize docs/ — merge 6 subdirs into 4

- docs/research/ → docs/dev/ (development records)
- docs/planning/ + docs/plans/ + docs/superpowers/ → docs/specs/
- Root task_plan.md, progress.md, findings.md → docs/dev/
- Delete empty dirs: planning/, plans/, research/, superpowers/"
```

---

### Task 4: 更新 .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: 覆写 .gitignore**

将 `.gitignore` 替换为以下内容：

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

- [ ] **Step 2: 验证 .gitignore 生效**

```bash
git status --short | grep -E "(archive\.zip|__pycache__|\.pytest_cache)" | head -5
```

Expected: 无输出（这些文件/目录不应再出现在 git status 中）

- [ ] **Step 3: 提交**

```bash
git add .gitignore
git commit -m "chore: update .gitignore

Add: archive.zip, .pytest_cache/, data/TextBraTS/, .ipynb_checkpoints/
Remove: docs/plans/ (merged into tracked docs/specs/)"
```

---

### Task 5: 最终验证

- [ ] **Step 1: 验证根目录条目数**

```bash
cd /Users/leiyuxuan/Documents/PyCharm_Project/TextMamba3D
ls | wc -l
```

Expected: ~20（之前 ~41）

- [ ] **Step 2: 验证 docs 结构**

```bash
ls docs/
```

Expected: `architecture.png  TextMamba3D_Analysis_Report.pdf  TextMamba3D_Reading_List.pdf  dev  guides  papers  specs`

- [ ] **Step 3: 验证 notebooks 结构**

```bash
ls notebooks/
ls notebooks/experiments/ | wc -l
```

Expected: `archive  experiments` 目录，experiments 下 10 个文件

- [ ] **Step 4: 验证 git 状态干净**

```bash
git status
git log --oneline -4
```

Expected: clean working tree, 4 个新 commit

- [ ] **Step 5: 验证磁盘空间**

```bash
du -sh .
```

Expected: ~13GB（从 ~25GB 降下来）
