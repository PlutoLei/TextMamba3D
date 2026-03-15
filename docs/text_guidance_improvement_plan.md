# TextMamba3D 文本引导改进计划

## 问题

v3 baseline (A100, patch 128³):
- With text: **88.15%** Dice
- Without text: **88.51%** Dice
- 文本引导净负面 (**-0.36%**)

## 根因

| # | 根因 | 严重性 |
|---|------|--------|
| 1 | 融合仅在 bottleneck 一层（4×4×4=64 tokens） | 高 |
| 2 | 零对齐信号（contrastive=0） | 高 |
| 3 | PubMedBERT 完全冻结 | 中 |
| 4 | zero-init out_proj 可能永远不激活 | 中 |
| 5 | BraTS 文本描述同质化 | 低-中 |

## 实施步骤

### Step 1: Multi-Scale Fusion + Unfreeze BERT ✅ 完成

**变更：**
- `models/textmamba3d.py`: bottleneck-only → MultiScalePixelTextAttention (stages 1,2,3)
- `configs/textbrats_a100.yaml`: `unfreeze_text_layers: 2`
- Stage 0 (32K tokens) 排除，避免 VRAM 问题

**Crossfire 审查：** ✅ Claude + Codex 双 PASS，4 tests passed

**新参数量：** ~38M trainable (vs 24M baseline)

**训练结果（2026-03-12）：**
- 166 epochs, early stopping
- Best val_dice: with-text 0.8847, no-text 0.8823
- Full-volume evaluation (95 cases, sliding window):
  - With text: ET=0.7663, TC=0.8411, WT=0.8875, Mean=0.8316
  - No text:   ET=0.7679, TC=0.8415, WT=0.8860, Mean=0.8318
- **结论：** 文本引导从净负面（v3: -0.36%）改善至接近中性（-0.02%），但仍无显著贡献
- **Checkpoint：** `Drive/TextMamba3D/checkpoints/best_v4.pth`

### Step 2: Pixel Contrastive + Semantic Matching Loss（待实施）

- 用 text-to-pixel contrastive 替代全局 contrastive（CRIS 启发）
- 用 MedCLIP 风格的 semantic matching loss 替代 InfoNCE
- `contrastive_weight` 从 0.0 调到 0.05-0.1

### Step 3: Class-Conditional Text Guidance（待实施）

- 将文本拆分为 ET/TC/WT 区域相关片段
- 分别引导对应分割通道
- GTGM 论文证实：文本对小目标（ET）帮助最大

### Step 4: Text Augmentation（待实施）

- LLM paraphrase augmentation 增加文本多样性
- 对无文本样本生成模板化描述

## 消融实验表（每步一行）

| 实验 | 变更 | Dice (text) | Dice (no-text) | Delta | 状态 |
|------|------|-------------|----------------|-------|------|
| v3 Baseline | — | 88.15% | 88.51% | -0.36% | ✅ 完成 |
| v4 Multi-scale + Unfreeze | Step 1 | 83.16% | 83.18% | -0.02% | ✅ 完成 |
| v5 + Pixel Contrastive | Step 1+2 | ? | ? | ? | 待实施 |
| v6 + Class-Conditional | Step 1+2+3 | ? | ? | ? | 待实施 |
| v7 + Text Augmentation | All | ? | ? | ? | 待实施 |

## 相关文献

- **CRIS** (CVPR 2022): text-to-pixel contrastive learning
- **GLoRIA** (ICCV 2021): global-local contrastive for medical VLP
- **MedCLIP** (arXiv 2210.10163): semantic matching loss for medical data
- **GTGM** (arXiv 2306.04811): text guidance 对小目标最有效
