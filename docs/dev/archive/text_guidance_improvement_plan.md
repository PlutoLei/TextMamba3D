# TextMamba3D 文本引导改进计划

> 最后更新：2026-03-15 | v4.4 训练完成后修订

## 问题演进

| 版本 | Delta (text vs no-text) | 诊断 |
|------|------------------------|------|
| v3 baseline | **-0.36%** | 文本净负面 — bottleneck 太窄 + BERT 冻结 + 无对齐信号 |
| v4.1 | **-0.02%** | 接近中性 — 多尺度融合+解冻BERT方向正确，但 Q/KV 方向错误 |
| v4.3 (PWAM) | **-0.26%** | 退步 — 乘法融合放大低质量文本噪声，过度复杂化 |
| **v4.4 (SeqCA)** | **+0.55%** | **首次正向！** Text=Q 方向翻转有效，但 ET 仍为短板 |

## 根因分析（基于 v4.4 结果更新）

| # | 根因 | 严重性 | 状态 |
|---|------|--------|------|
| 1 | 融合仅在 bottleneck 一层 | 高 | ✅ Step 1 已解决（多尺度 stages 1,2,3） |
| 2 | Q/KV 方向错误（Image=Q 让模型可忽略文本） | 高 | ✅ v4.4 SeqCA 已解决（Text=Q） |
| 3 | PubMedBERT 完全冻结 | 中 | ✅ Step 1 已解决（unfreeze last 2） |
| 4 | **文本缺乏 ET 特异性语义** | **高** | ❌ 未解决 — 当前最大瓶颈 |
| 5 | Stage 0 (32K tokens) 无文本融合，ET 依赖的高分辨率层缺失 | 中 | ❌ 计算约束，暂无方案 |
| 6 | 文本描述同质化（369 份结构相似） | 低-中 | ❌ 待 Step 2 解决 |

## 实施路线（修订版）

### Step 1: Multi-Scale Fusion + Unfreeze BERT ✅ 完成

**变更：** bottleneck-only → MultiScalePixelTextAttention (stages 1,2,3) + `unfreeze_text_layers: 2`

**训练结果（2026-03-12）：** 166 epochs, delta = **-0.02%**（从 -0.36% 改善 94%）

**Checkpoint：** `Drive/TextMamba3D/checkpoints/best_v4.pth`

### Step 1.5: SeqCA Q/KV 方向翻转 ✅ 完成（v4.4）

**变更：** PixelTextCrossAttention → SequentialCrossAttention（Text=Q, Image=KV → Image=Q, Refined=KV）

**训练结果（2026-03-15）：** 200 epochs

| 指标 | With Text | No Text | Delta |
|------|-----------|---------|-------|
| ET | 0.7630 ± 0.2164 | 0.7646 ± 0.2127 | **-0.16%** |
| TC | 0.8512 ± 0.1494 | 0.8407 ± 0.1635 | **+1.05%** |
| WT | 0.8887 ± 0.0722 | 0.8813 ± 0.0781 | **+0.74%** |
| **Mean** | **0.8343** | **0.8288** | **+0.55%** |

**分析：** TC/WT 显著受益于文本引导，但 **ET 仍轻微负面**。原因：
1. 文本描述中缺乏 ET 特异性术语（无 "enhancing rim"、"contrast enhancement"）
2. Stage 0（ET 最依赖的高分辨率层）无文本融合
3. SeqCA 的 attention 对 ET 这种小目标容易分散

**Checkpoint：** `Drive/TextMamba3D/checkpoints/best_v4.4.pth`

### ~~Step 2: Pixel Contrastive + Semantic Matching Loss~~ ⏸️ 降低优先级

原计划用 CRIS 风格 text-to-pixel contrastive + MedCLIP semantic matching。但 v4.4 已证明 SeqCA 可实现正向 delta（无需对比学习），且 v4.3 的辅助损失（T2VLoss + NecessityLoss）实测无效。

**新判断：** 对比学习不是当前瓶颈。如果 Step 2（ET-Enriched Text）成功提升 ET，可考虑在 Step 3 中作为锦上添花加入。

### Step 2（新）: ET-Enriched Text Generation（下一步）

**目标：** 为 369 个样本补充 ET 特异性文本描述，解决根因 #4。

**方案：** 从 segmentation mask 自动提取 ET 定量特征，模板化拼接到现有专家文本后面。

从 mask 提取的 ET 特征：
- `enhancing_ratio`：ET 体积 / TC 体积
- `enhancement_pattern`：环形 vs 实性 vs 斑片状（从 mask 形态计算）
- `boundary_regularity`：表面积 / 体积比

生成示例：
```
原文本 + " Enhancing component: rim-enhancing, occupying 12% of
tumor core, with irregular margins."
```

**零标注成本**（纯代码实现），~1-2 天工作量。

**训练策略：** LaCLIP 的 stochastic selection——训练时 50% 概率用原文本，50% 用 enriched 文本，增加鲁棒性。

**文献支撑：**
- **LaCLIP** (NeurIPS 2023): LLM rewrite + stochastic selection
- **MIMIC-R3G** (ICLR 2025): LLM 驱动的放射学文本生成 pipeline
- **StructTuning** (ICLR 2025): 结构化知识注入，极端数据效率

### Step 3（新）: LaPael 式 Embedding 扰动

**目标：** 在 PubMedBERT **早期层（layer 1-3）** 施加 input-dependent noise，增加文本表示多样性。

**与原 Module E 的区别：** 原方案在最终输出加 Gaussian noise；LaPael (NeurIPS 2024) 发现**扰动早期层效果更好**，因为早期层控制表层表达，后期层控制语义。

**前置条件：** Step 2 先验证 ET-enriched text 是否有效。

### Step 4（新）: Contrastive Loss（可选锦上添花）

仅当 Step 2-3 使 ET delta 转正后考虑。用 v4.2 已设计的 ForegroundContrastiveLoss + warmup 调度。

## 已验证无效的方案（红线清单）

| 方案 | 版本 | 结果 | 教训 |
|------|------|------|------|
| 浅层 FiLM 注入 | v2 | -0.19% | 低级特征层不应注入高维语义 |
| 因果 MambaFusion | v2 | -0.19% | 单向 SSM 不适合跨模态融合 |
| default_text_embed | v2 | — | 污染 text-free 基线，永远不要用 |
| PWAM 乘法融合 | v4.3 | -0.26% | 低质量文本 + 乘法 = 放大噪声 |
| TextToVoxelLoss | v4.3 | -0.26% | T2V 权重 0.1 太弱，且 3D kernel 参数膨胀 |
| TextNecessityLoss | v4.3 | -0.26% | 双前向开销大，margin hinge 在小 batch 下不稳定 |
| EmbeddingPerturbation (输出层) | v4.3 | -0.26% | 输出层扰动破坏语义；应扰动早期层 (LaPael) |

## 消融实验表

| 实验 | 变更 | Dice (text) | Dice (no-text) | Delta | 状态 |
|------|------|-------------|----------------|-------|------|
| v3 Baseline | — | 88.15% | 88.51% | -0.36% | ✅ 完成 |
| v4.1 Multi-scale + Unfreeze | Step 1 | 83.16% | 83.18% | -0.02% | ✅ 完成 |
| v4.3 PWAM + 辅助损失 | PWAM+T2V+Nec+EmbPerturb | 83.19% | 83.45% | -0.26% | ❌ 失败 |
| **v4.4 SeqCA** | **Text=Q 方向翻转** | **83.43%** | **82.88%** | **+0.55%** | **✅ 首次正向** |
| v4.5 + ET-Enriched Text | Step 2 (新) | ? | ? | ? | 下一步 |
| v4.6 + LaPael Perturbation | Step 2+3 (新) | ? | ? | ? | 待定 |
| v4.7 + Contrastive | Step 2+3+4 (新) | ? | ? | ? | 可选 |

## BraTS2020 竞争力定位

| 方法 | ET | TC | WT | Mean | 对比 |
|------|----|----|----|----- |------|
| nnU-Net (冠军) | 79.89 | 85.06 | 91.24 | **85.40** | +1.97% |
| TextBraTS (MICCAI'25) | 83.3 | 82.8 | 89.9 | **85.3** | +1.87% |
| **TextMamba3D v4.4** | **76.30** | **85.12** | **88.87** | **83.43** | — |
| SwinUNETR | 81.0 | 80.8 | 89.5 | 83.8 | +0.37% |
| TransBTS | 78.73 | 81.73 | 90.09 | 83.52 | +0.09% |

**最大差距在 ET**（76.30 vs TextBraTS 83.3，差 7 个点）。TC 85.12 已超过所有基线。

## 相关文献

- **TextBraTS** (MICCAI 2025): SeqCA 启发源，同数据集 +1.5% Dice
- **LaCLIP** (NeurIPS 2023): LLM text rewrite + stochastic selection
- **LaPael** (NeurIPS 2024): 早期层 latent perturbation
- **MIMIC-R3G** (ICLR 2025): LLM 驱动放射学文本生成
- **StructTuning** (ICLR 2025): 结构化知识注入
- **CRIS** (CVPR 2022): text-to-pixel contrastive learning
- **GLoRIA** (ICCV 2021): global-local contrastive for medical VLP
- **SimGRACE** (ACM 2022): encoder perturbation 替代数据增强
