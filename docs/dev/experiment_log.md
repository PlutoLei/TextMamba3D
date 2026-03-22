# TextMamba3D 实验日志

> 记录各版本的实验设计、训练结果与失败教训。前身为 `text_guidance_improvement_plan.md` 的原始版本。

## 原始 4-Step 改进计划（2026-03-12 制定）

基于 v3 baseline delta = -0.36% 的根因分析，设计了 4 步递进路线：

| Step | 内容 | 解决的根因 |
|------|------|-----------|
| Step 1 | Multi-Scale Fusion + Unfreeze BERT | 根因 #1（bottleneck 太窄）+ #3（BERT 冻结） |
| Step 2 | Pixel Contrastive + Semantic Matching Loss | 根因 #2（无对齐信号） |
| Step 3 | Class-Conditional Text Guidance | 根因 #5（文本同质化） |
| Step 4 | Text Augmentation | 根因 #5（文本同质化） |

---

## 实验 1: v3 Baseline（A100, embed_dim=96）

**配置：** patch 128³, batch 1-4, bottleneck-only CrossAttn, BERT 全冻结, contrastive=0

**结果：** Dice 88.15% (text) / 88.51% (no-text), **delta = -0.36%**

**诊断：** 绝对性能大幅提升（v2 68% → 88%），但文本仍为负面。bottleneck 64 tokens 太窄，BERT 冻结无法适应分割任务。

---

## 实验 2: v4.1 Multi-Scale + Unfreeze（Step 1）

**变更：**
- `models/textmamba3d.py`: bottleneck-only → MultiScalePixelTextAttention (stages 1,2,3)
- `configs/textbrats_a100.yaml`: `unfreeze_text_layers: 2`, embed_dim: 48
- Stage 0 (32K tokens) 排除，避免 VRAM 问题

**Crossfire 审查：** ✅ Claude + Codex 双 PASS，4 tests passed

**训练：** 166 epochs, early stopping, ~38M trainable params

**Full-volume 评估（95 test cases, sliding window）：**

| 指标 | With Text | No Text | Delta |
|------|-----------|---------|-------|
| ET | 0.7663 | 0.7679 | -0.16% |
| TC | 0.8411 | 0.8415 | -0.04% |
| WT | 0.8875 | 0.8860 | +0.15% |
| **Mean** | **0.8316** | **0.8318** | **-0.02%** |

**结论：** delta 从 -0.36% 改善至 -0.02%（改善 94%），但绝对 Dice 从 88% 降至 83%（embed_dim 96→48）。方向正确但文本贡献仍不显著。

**Checkpoint：** `Drive/TextMamba3D/checkpoints/best_v4.pth`

---

## 实验 3: v4.2 ForegroundContrastiveLoss（设计未训练）

**设计：** 前景加权 InfoNCE + warmup 调度（epoch 30 后线性引入）

经 Codex DEBATE 质询修订了 5 个关键问题（4³ 分辨率下 pixel-level contrastive 是伪命题、cosine→BCE 数学不兼容等）。

**状态：** 未训练。v4.3 先行测试了更激进的方案。

---

## 实验 4: v4.3 PWAM + 多辅助损失（Step 4 Proposal 实验）

**变更：** 基于 `architecture_proposal_step4.md` 的 Module B-E：
- PWAM3D 乘法融合替换 additive CrossAttn (Module B)
- TextToVoxelLoss: CRIS 风格 per-voxel BCE, weight=0.1 (Module C)
- TextNecessityLoss: hinge loss on dice delta, weight=0.05, warmup 30 epochs (Module D)
- EmbeddingPerturbation: Gaussian + learned directions on PubMedBERT output (Module E)

**Codex DEBATE：** 10 issues resolved（维度不匹配、mask shape、OOM 风险等）

**训练：** 300 epochs config, A100

**Full-volume 评估（94 test cases）：**

| 指标 | With Text | No Text | Delta |
|------|-----------|---------|-------|
| ET | 0.7574 | 0.7657 | **-0.83%** |
| TC | 0.8409 | 0.8413 | -0.04% |
| WT | 0.8975 | 0.8965 | +0.10% |
| **Mean** | **0.8319** | **0.8345** | **-0.26%** |

**结论：** 相比 v4.1 的 -0.02%，**退步至 -0.26%**。ET 受损最严重 (-0.83%)。

**失败根因：**
1. **PWAM 乘法融合放大噪声** — `gate * (vis × lang) + vis` 在低质量文本上放大噪声。V4.1 的 `residual + out_proj(attn)` 更安全（zero-init → identity start）
2. **辅助损失太弱** — T2VLoss (0.1) + NecessityLoss (0.05) 仅占总 loss 的 ~5%
3. **复杂度爆炸** — 4 个新模块 + 双阶段训练 + dual forward，无法隔离改善来源

**教训：** 确立第 5 条红线——乘法融合 (PWAM) 在当前 BraTS 文本质量下不如加法残差。

---

## 实验 5: v4.4 SeqCA 两步交叉注意力

**变更：** 仅 1 处——PixelTextCrossAttention → SequentialCrossAttention
- Step 1 (T2I): Text=Q, Image=KV（文本主动定位自己描述的区域）
- Step 2 (I2T): Image=Q, Refined=KV（图像用文本过滤后的信息增强自身）
- 理论依据：TextBraTS (MICCAI 2025) 同数据集 +1.5% Dice

**训练：** 200 epochs (0-150 + 151-199 resume), contrastive_weight=0.0（隔离 SeqCA 效果）

**Full-volume 评估（95 test cases, sliding window）：**

| 指标 | With Text | No Text | Delta |
|------|-----------|---------|-------|
| ET | 0.7630 ± 0.2164 | 0.7646 ± 0.2127 | -0.16% |
| TC | 0.8512 ± 0.1494 | 0.8407 ± 0.1635 | **+1.05%** |
| WT | 0.8887 ± 0.0722 | 0.8813 ± 0.0781 | **+0.74%** |
| **Mean** | **0.8343** | **0.8288** | **+0.55%** |

**结论：** **首次实现正向 text guidance delta！** TC/WT 显著受益，但 ET 仍轻微负面。

**ET 负面原因分析：**
1. 文本描述缺乏 ET 特异性语义（无 "enhancing rim"、"contrast enhancement"）
2. Stage 0（ET 最依赖的高分辨率层，32K tokens）无文本融合
3. SeqCA attention 对小目标（ET 通常只有几个 voxel 厚）容易分散

**BraTS2020 竞争力：** Mean Dice 83.43% 处于中游（TransBTS 83.52, SwinUNETR 83.8），TC 85.12% 超过所有基线。ET 76.30% 是最大短板（TextBraTS 83.3%，差 7 个点）。

**Checkpoint：** `Drive/TextMamba3D/checkpoints/best_v4.4.pth`

---

## 消融实验总表

| 版本 | 融合机制 | 辅助损失 | Dice (text) | Dice (no-text) | Delta | 结论 |
|------|---------|---------|-------------|----------------|-------|------|
| v2 | FiLM+MambaFusion+CA | — | 68.48% | 68.67% | -0.19% | 浅层 FiLM 有害 |
| v3 | Bottleneck CA | — | 88.15% | 88.51% | -0.36% | 太窄 + BERT 冻结 |
| v4.1 | MultiScale CA (Image=Q) | — | 83.16% | 83.18% | -0.02% | 方向正确但 Q/KV 错 |
| v4.3 | PWAM (乘法) | T2V+Nec+EmbPerturb | 83.19% | 83.45% | -0.26% | 乘法放大噪声 |
| **v4.4** | **SeqCA (Text=Q)** | **—** | **83.43%** | **82.88%** | **+0.55%** | **✅ 首次正向** |

## 已确认的红线

1. 不用 CausalMambaFusion（单向偏差）
2. 不在浅层注入文本（Stage 0 排除）
3. 不用 null/default text embed（污染基线）
4. Bottleneck-only 不够（需多尺度）
5. PWAM 乘法融合在低质量文本下不如加法残差
6. 复杂度爆炸（一次改多个变量）无法隔离改善来源
