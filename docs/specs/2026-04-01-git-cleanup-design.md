# TextMamba3D Git 整理设计文档

**日期:** 2026-04-01
**目标:** 清理陈旧分支、标记版本里程碑、建立 commit 规范，不重写历史

---

## 1. 整理概览

| 操作 | 预期效果 |
|------|----------|
| 删除已合入的远程分支 | 从 3 个远程分支减到 2 个 |
| 打 5 个版本 tag | 关键里程碑可快速定位 |
| 添加 commit 模板 | 未来 commit message 格式统一 |

**不做的事：** 不重写历史、不 squash、不 force push。

## 2. 分支清理

### 删除

| 分支 | 原因 |
|------|------|
| `origin/feat/v4.6-attnres-skip-gate` | 已完全合入 main（0 个独有 commit） |

### 保留

| 分支 | 原因 |
|------|------|
| `origin/main` | 主分支 |
| `origin/keyword-text-experiment` | 2 个独有 commit，供将来参考 |

## 3. 版本 Tag

使用 annotated tag（`git tag -a`），包含版本说明。

| Tag | Commit | 说明 |
|-----|--------|------|
| `v5.0` | `d6aee89` | 基线模型确立 (Mamba2 + SeqCA, Mean Dice=0.8479) |
| `v6.0` | `7ea4368` | BottleneckSeqCA 实验 (TC+0.0086, ET-0.0140) |
| `v7.0` | `5881bea` | Boundary + Hierarchy Loss (WT best-ever 0.9032) |
| `v8.0-wip` | `d2e0a26` | 两阶段预训练 BraTS2021 (进行中) |
| `ensemble-best` | `8e1f4c7` | 3模型集成最佳结果 (Mean Dice=0.8511) |

## 4. Commit 规范

### .gitmessage 模板

```
<type>(<scope>): <subject>

# type: feat | fix | docs | test | chore | refactor | perf
# scope: 可选，如 v8, model, loss, data, notebook
# subject: 简明描述，不超过 72 字符
#
# 示例:
#   feat(v8): two-stage training configs and notebook
#   fix(model): correct decoder skip connection dimensions
#   docs: update experiment log with V7.0 results
```

### 配置方式

```bash
git config --local commit.template .gitmessage
```

使用模板而非 hook，柔性引导而非强制拦截。

## 5. 执行策略

分 3 个步骤，每步一个 commit（或独立操作）：

1. **删除远程分支** — `git push origin --delete feat/v4.6-attnres-skip-gate`
2. **打 5 个 annotated tag 并推送** — `git tag -a` + `git push --tags`
3. **添加 .gitmessage 模板并配置** — commit 模板文件 + `git config --local`
