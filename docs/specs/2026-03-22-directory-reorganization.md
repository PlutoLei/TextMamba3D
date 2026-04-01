# TextMamba3D Directory Reorganization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up the 44GB TextMamba3D project — remove dead files, archive old notebooks, reorganize configs/docs, fix hardcoded references, and push to GitHub.

**Architecture:** Three atomic commits: (1) cleanup, (2) restructure, (3) fix references. All moves use `git mv` to preserve history. The `keyword-text-experiment` branch is protected — `configs/textbrats.yaml`, `train.py`, and `data/brats_textbrats_dataset.py` must not be renamed.

**Tech Stack:** Git, Bash

**Spec:** `docs/specs/2026-03-22-directory-reorganization-design.md`

---

## Chunk 1: Cleanup (Commit 1)

### Task 1: Delete untracked temporary files

**Files:**
- Delete: `.crossfire/` (untracked, ~audit files + tmp git repo)
- Delete: `.ruff_cache/`
- Delete: `TextMamba3D/TextMamba3D/` (nested empty dir)
- Delete: all `__pycache__/` directories

- [ ] **Step 1: Remove untracked temporary directories**

```bash
cd E:/VSCode_Project/TextMamba3D
rm -rf .crossfire/
rm -rf .ruff_cache/
rm -rf TextMamba3D/TextMamba3D/
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
```

- [ ] **Step 2: Verify cleanup**

```bash
ls -d .crossfire/ .ruff_cache/ TextMamba3D/TextMamba3D/ 2>&1
# Expected: all "No such file or directory"
find . -type d -name __pycache__ | wc -l
# Expected: 0
```

### Task 2: Delete tracked dead files

**Files:**
- Delete (tracked): `download.html`, `GIT_PULL_GUIDE.md`
- Delete (tracked): `docs/generate_report_pdf.py`
- Delete (tracked): `scripts/generate_v43_notebook.py`
- Delete (gitignored): `nexus_state.json`

- [ ] **Step 1: Remove gitignored file**

```bash
cd E:/VSCode_Project/TextMamba3D
rm -f nexus_state.json
```

- [ ] **Step 2: Remove tracked files via git rm**

```bash
git rm download.html
git rm GIT_PULL_GUIDE.md
git rm docs/generate_report_pdf.py
git rm scripts/generate_v43_notebook.py
```

- [ ] **Step 3: Verify removals**

```bash
git status --short
# Expected: 4 lines starting with "D " for the removed files
ls download.html GIT_PULL_GUIDE.md docs/generate_report_pdf.py scripts/generate_v43_notebook.py 2>&1
# Expected: all "No such file or directory"
```

### Task 3: Update .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Append new ignore patterns**

Add these lines to the end of `.gitignore`:

```
.ruff_cache/
.crossfire/
```

- [ ] **Step 2: Stage the change**

```bash
git add .gitignore
```

### Task 4: Commit cleanup

- [ ] **Step 1: Review staged changes**

```bash
git status
git diff --cached --stat
```

Expected: 4 deleted files + 1 modified (.gitignore). No unexpected changes.

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: clean up dead files, caches, and obsolete scripts

Remove empty download.html, GIT_PULL_GUIDE.md (content in README),
V4.3 notebook generator script, and one-off report generator.
Add .ruff_cache/ and .crossfire/ to .gitignore.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 2: Restructure (Commit 2)

### Task 5: Create target directories

- [ ] **Step 1: Create all new directories**

```bash
cd E:/VSCode_Project/TextMamba3D
mkdir -p notebooks/archive/v4.0
mkdir -p notebooks/archive/v4.1
mkdir -p notebooks/archive/v4.2
mkdir -p notebooks/archive/v4.3
mkdir -p notebooks/archive/v4.4
mkdir -p notebooks/archive/v5.0
mkdir -p notebooks/archive/misc
mkdir -p configs/archive
mkdir -p docs/guides
mkdir -p docs/dev/archive
```

Note: `notebooks/archive/v4.5/` and `notebooks/archive/v4.6/` will be created by moving existing `notebooks/v4.5/` and `notebooks/v4.6/`.

### Task 6: Archive notebooks from root

**Files:**
- Move: 8 notebooks from root → `notebooks/archive/`

- [ ] **Step 1: Move versioned training notebooks**

```bash
cd E:/VSCode_Project/TextMamba3D
git mv TextMamba3D_A100.ipynb notebooks/archive/v4.0/
git mv TextMamba3D_A100_V4.1.ipynb notebooks/archive/v4.1/
git mv TextMamba3D_A100_V4.2.ipynb notebooks/archive/v4.2/
git mv TextMamba3D_A100_V4.3.ipynb notebooks/archive/v4.3/
git mv TextMamba3D_A100_V4.4.ipynb notebooks/archive/v4.4/
git mv TextMamba3D_A100_V5.0.ipynb notebooks/archive/v5.0/
```

- [ ] **Step 2: Move H100 and misc notebooks**

```bash
git mv TextMamba3D_H100_V4.6.ipynb notebooks/archive/v4.6/
git mv TextMamba3D_Mamba3_Test.ipynb notebooks/archive/misc/
```

- [ ] **Step 3: Verify root only has 3 notebooks**

```bash
ls *.ipynb
# Expected: TextMamba3D_A100_V5.1.ipynb  TextMamba3D_Ablation_TC.ipynb  TextMamba3D_eval_unified.ipynb
```

### Task 7: Archive existing notebook subdirectories

**Files:**
- Move: `notebooks/v4.5/` → `notebooks/archive/v4.5/`
- Move: `notebooks/v4.6/` contents → `notebooks/archive/v4.6/` (handle filename collision)

- [ ] **Step 1: Move v4.5 directory**

```bash
cd E:/VSCode_Project/TextMamba3D
git mv notebooks/v4.5 notebooks/archive/v4.5
```

- [ ] **Step 2: Handle v4.6 with filename collision**

Root's `TextMamba3D_H100_V4.6.ipynb` (64KB, newer) was already moved to `notebooks/archive/v4.6/` in Task 6.
Now move the remaining files from `notebooks/v4.6/` that don't collide:

```bash
# The duplicate TextMamba3D_H100_V4.6.ipynb (20KB, older) — remove it
git rm notebooks/v4.6/TextMamba3D_H100_V4.6.ipynb
# Move the eval notebook (no collision)
git mv notebooks/v4.6/TextMamba3D_H100_V4.6_eval.ipynb notebooks/archive/v4.6/
```

- [ ] **Step 3: Clean up empty v4.6 directory**

```bash
rmdir notebooks/v4.6 2>/dev/null || true
```

- [ ] **Step 4: Verify archive structure**

```bash
find notebooks/archive -type f -name "*.ipynb" | sort
# Expected: 12 notebooks across v4.0-v5.0 + misc
```

### Task 8: Rename and archive configs

**Files:**
- Rename: `textbrats_a100.yaml` → `a100.yaml`
- Rename: `textbrats_a100_v5.1.yaml` → `a100_v5.1.yaml`
- Rename: `textbrats_v8_h100.yaml` → `h100.yaml`
- Archive: 4 old configs → `configs/archive/`

- [ ] **Step 1: Rename active configs**

```bash
cd E:/VSCode_Project/TextMamba3D
git mv configs/textbrats_a100.yaml configs/a100.yaml
git mv configs/textbrats_a100_v5.1.yaml configs/a100_v5.1.yaml
git mv configs/textbrats_v8_h100.yaml configs/h100.yaml
```

- [ ] **Step 2: Archive old configs**

```bash
git mv configs/textbrats_v5.yaml configs/archive/
git mv configs/textbrats_v7.yaml configs/archive/
git mv configs/textbrats_v8.yaml configs/archive/
git mv configs/textbrats_a100_v5.yaml configs/archive/
```

- [ ] **Step 3: Verify config structure**

```bash
ls configs/
# Expected: a100.yaml  a100_v5.1.yaml  archive  default.yaml  h100.yaml  textbrats.yaml
ls configs/archive/
# Expected: textbrats_a100_v5.yaml  textbrats_v5.yaml  textbrats_v7.yaml  textbrats_v8.yaml
```

### Task 9: Reorganize docs

**Files:**
- Move: 4 files → `docs/guides/`
- Move: 4 files → `docs/dev/`
- Move: 6 files → `docs/dev/archive/`

- [ ] **Step 1: Move guides**

```bash
cd E:/VSCode_Project/TextMamba3D
git mv docs/USAGE.md docs/guides/
git mv docs/REPRODUCTION_GUIDE.md docs/guides/
git mv docs/训练指南.md docs/guides/
git mv MACBOOK_GUIDE.md docs/guides/COLAB_GUIDE.md
```

- [ ] **Step 2: Move active dev docs**

```bash
git mv docs/architecture_evolution_v1_to_v4.md docs/dev/
git mv docs/experiment_log.md docs/dev/
git mv docs/v44_changelog.md docs/dev/
git mv docs/mamba3_integration_lessons.md docs/dev/
```

- [ ] **Step 3: Move archived dev docs**

```bash
git mv docs/architecture_proposal_step4.md docs/dev/archive/
git mv docs/architecture_v2_postmortem.md docs/dev/archive/
git mv docs/improvement_plan.md docs/dev/archive/
git mv docs/text_guidance_improvement_plan.md docs/dev/archive/
git mv docs/research_summary_step4.md docs/dev/archive/
git mv docs/v44_crossfire_audit.md docs/dev/archive/
```

- [ ] **Step 4: Clean up empty docs/superpowers if needed**

The specs/ and plans/ we created stay. No cleanup needed.

- [ ] **Step 5: Verify docs structure**

```bash
echo "=== guides ===" && ls docs/guides/
echo "=== dev ===" && ls docs/dev/
echo "=== dev/archive ===" && ls docs/dev/archive/
echo "=== Papers ===" && ls docs/Papers/ | head -5
echo "=== planning ===" && ls docs/planning/
echo "=== root docs ===" && ls docs/*.png docs/*.md 2>/dev/null
```

### Task 10: Commit restructure

- [ ] **Step 1: Review all staged changes**

```bash
git status
git diff --cached --stat
```

Expected: ~25 renames/moves, 1 deletion (duplicate v4.6 notebook). No content modifications.

- [ ] **Step 2: Commit**

```bash
git commit -m "refactor: reorganize notebooks, configs, and docs

Notebooks: archive 8 old versions to notebooks/archive/{version}/,
keep V5.1 + eval_unified + Ablation_TC in root.
Configs: rename active configs for clarity (a100, h100, a100_v5.1),
archive 4 old experiment configs. textbrats.yaml kept as-is to
protect keyword-text-experiment branch.
Docs: split into guides/ (user-facing) and dev/ (development history),
move MACBOOK_GUIDE.md to docs/guides/COLAB_GUIDE.md.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 3: Fix References (Commit 3)

### Task 11: Update evaluate_full.py default config path

**Files:**
- Modify: `evaluate_full.py:25`

- [ ] **Step 1: Update the default config argument**

In `evaluate_full.py` line 25, change:
```python
parser.add_argument('--config', type=str, default='configs/textbrats_a100.yaml')
```
to:
```python
parser.add_argument('--config', type=str, default='configs/a100.yaml')
```

- [ ] **Step 2: Verify no other references to old config names in .py files**

```bash
cd E:/VSCode_Project/TextMamba3D
grep -rn "textbrats_a100\." *.py scripts/*.py 2>/dev/null
# Expected: no output (all references updated)
```

### Task 12: Update notebook config references

**Files:**
- Modify: `TextMamba3D_A100_V5.1.ipynb` (config path in training cell)
- Check: `TextMamba3D_eval_unified.ipynb` and `TextMamba3D_Ablation_TC.ipynb`

- [ ] **Step 1: Check what configs the 3 active notebooks reference**

```bash
cd E:/VSCode_Project/TextMamba3D
grep -o "configs/[a-zA-Z0-9_]*\.yaml" TextMamba3D_A100_V5.1.ipynb | sort -u
grep -o "configs/[a-zA-Z0-9_]*\.yaml" TextMamba3D_eval_unified.ipynb | sort -u
grep -o "configs/[a-zA-Z0-9_]*\.yaml" TextMamba3D_Ablation_TC.ipynb | sort -u
```

- [ ] **Step 2: Update V5.1 notebook**

If it references `configs/textbrats_a100_v5.1.yaml`, replace with `configs/a100_v5.1.yaml`.
Use `sed` or manual edit on the JSON cell contents.

```bash
sed -i 's|configs/textbrats_a100_v5.1.yaml|configs/a100_v5.1.yaml|g' TextMamba3D_A100_V5.1.ipynb
```

- [ ] **Step 3: Update eval_unified notebook**

If it references `configs/textbrats_a100.yaml`, replace with `configs/a100.yaml`.
If it references `configs/textbrats_v8_h100.yaml`, replace with `configs/h100.yaml`.

```bash
sed -i 's|configs/textbrats_a100\.yaml|configs/a100.yaml|g' TextMamba3D_eval_unified.ipynb
sed -i 's|configs/textbrats_v8_h100\.yaml|configs/h100.yaml|g' TextMamba3D_eval_unified.ipynb
```

- [ ] **Step 4: Update Ablation notebook**

Check what it references. If it uses configs that were archived (e.g., `textbrats_v7.yaml`), update to `configs/archive/textbrats_v7.yaml`.

```bash
sed -i 's|configs/textbrats_v7\.yaml|configs/archive/textbrats_v7.yaml|g' TextMamba3D_Ablation_TC.ipynb
```

- [ ] **Step 5: Verify no stale config references in active notebooks**

```bash
for nb in TextMamba3D_A100_V5.1.ipynb TextMamba3D_eval_unified.ipynb TextMamba3D_Ablation_TC.ipynb; do
    echo "=== $nb ==="
    grep -o "configs/textbrats[a-zA-Z0-9_]*\.yaml" "$nb" || echo "(clean)"
done
# Expected: all "(clean)" — no references to old textbrats_a100/v5/v7/v8 configs
# Note: configs/textbrats.yaml references are OK (kept as-is)
```

### Task 13: Stage and commit reference fixes

- [ ] **Step 1: Stage all changes**

```bash
cd E:/VSCode_Project/TextMamba3D
git add evaluate_full.py
git add TextMamba3D_A100_V5.1.ipynb TextMamba3D_eval_unified.ipynb TextMamba3D_Ablation_TC.ipynb
```

- [ ] **Step 2: Review diff**

```bash
git diff --cached
```

Expected: only config path strings changed, no logic changes.

- [ ] **Step 3: Commit**

```bash
git commit -m "fix: update config paths after reorganization

Update evaluate_full.py default config and notebook references
to match renamed configs (a100.yaml, a100_v5.1.yaml, h100.yaml).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 4: Push and Verify

### Task 14: Final verification

- [ ] **Step 1: Verify git log**

```bash
cd E:/VSCode_Project/TextMamba3D
git log --oneline -5
# Expected: 3 new commits on top
```

- [ ] **Step 2: Verify final directory structure**

```bash
echo "=== root ===" && ls *.py *.ipynb *.md
echo "=== configs ===" && ls configs/
echo "=== configs/archive ===" && ls configs/archive/
echo "=== notebooks/archive ===" && find notebooks/archive -type f -name "*.ipynb" | wc -l
echo "=== docs ===" && ls docs/
echo "=== docs/guides ===" && ls docs/guides/
echo "=== docs/dev ===" && ls docs/dev/
echo "=== losses ===" && ls losses/*.py
```

- [ ] **Step 3: Run smoke test to verify imports still work**

```bash
cd E:/VSCode_Project/TextMamba3D
python -c "from losses import CombinedLoss; from models import TextMamba3D; print('imports OK')"
```

### Task 15: Push to GitHub

- [ ] **Step 1: Push all commits**

```bash
cd E:/VSCode_Project/TextMamba3D
git push origin main
```

- [ ] **Step 2: Verify on remote**

```bash
git log --oneline origin/main -5
# Expected: 3 new commits visible
```

### Task 16: Clean up merged remote branch (after colleague confirmation)

**This task requires user confirmation before execution.**

- [ ] **Step 1: Confirm with user that colleague approves deletion**

Ask: "Can I delete the remote branch `feat/v4.6-attnres-skip-gate`? It's fully merged into main."

- [ ] **Step 2: Delete remote branch (only after approval)**

```bash
git push origin --delete feat/v4.6-attnres-skip-gate
```
