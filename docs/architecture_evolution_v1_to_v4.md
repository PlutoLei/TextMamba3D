# TextMamba3D 架构变迁：V1 → V4

> 文档版本：2026-03-13 | 项目：TextMamba3D — 文本引导 3D 脑瘤分割

## 1. 核心问题

TextMamba3D 要解决的核心问题是：**如何让文本引导（放射学报告）真正提升 3D 脑瘤分割性能？**

BraTS2020 数据集包含 369 例多模态 MRI（T1, T1ce, T2, FLAIR）和对应的专家报告，任务是分割三类肿瘤亚区（ET 增强型肿瘤、TC 肿瘤核心、WT 整体肿瘤）。四个版本的迭代围绕"文本信号怎么注入才有效"展开，每一步都是对上一版失败原因的定向修复。

## 2. 版本总览

| 版本 | 硬件 | Patch | embed_dim | Mean Dice (text) | Delta (text vs no-text) | 状态 |
|------|------|-------|-----------|-----------------|------------------------|------|
| V2 | RTX 4060 8GB | 64³ | 48 | 68.48% | -0.19% | 失败 |
| V3 | A100 40GB | 128³ | 96 | 88.15% | -0.36% | 基线 |
| V4.1 | A100 40GB | 128³ | 48 | 83.16% | -0.02% | 历史最佳 |
| V4.2 | A100 40GB | 128³ | 48 | — | — | 未训练 |
| V4.3 | A100 40GB | 128³ | 48 | 83.19% | -0.26% | 退步 |
| V4.4 | A100 40GB | 128³ | 48 | — | — | 待训练 |

演进主线：

```
V2: 到处注入文本 → 文本有害 (-0.19%)
 ↓ 教训：浅层不能注入语义信号；FiLM 不适合高维文本
V3: 只在 bottleneck 注入 → 文本仍有害 (-0.36%)，但绝对性能大幅提升
 ↓ 教训：注入点正确但太窄；缺少对齐信号；BERT 冻结无法适应
V4.1: 扩展到 3 个 stage + 解冻 BERT → 近乎中性 (-0.02%) ← 历史最佳
 ↓ 教训：融合面足够了，但缺少显式的文本-图像对齐损失
V4.2: 加入前景对比损失 + warmup → 未训练
 ↓
V4.3: PWAM 乘法融合 + 4 个辅助模块 → 退步 (-0.26%)
 ↓ 教训：乘法融合在低质量文本上放大噪声；过度复杂化
V4.4: SeqCA (Text=Q) 翻转 Q/KV 方向 → 待训练
 ↓ 基于 TextBraTS MICCAI'25 同数据集 +1.5% 验证
```

## 3. V2：FiLM + CausalMambaFusion

### 3.1 架构设计

V2 采用三层文本注入机制，文本参数量 ~3.3M，占模型总参数（24M）的 13.8%。

```
Encoder (4 stages) ──┬── MultiScaleFiLM (γ·x + β 调制所有 4 层)
                     │        ↓
                     └── CausalMambaFusion (文本序列拼接在图像前，因果 SSM)
                              ↓
                           Decoder
                     ↑
              PixelTextCrossAttention (多尺度交叉注意力，所有 4 层)
```

| 注入层 | 机制 | 位置 | 参数量 |
|--------|------|------|--------|
| MultiScaleFiLM | γ·x + β 仿射调制 | Encoder 所有 4 个 stage | ~0.3M |
| CausalMambaFusion | 文本 token 拼接在图像 token 前，因果 SSM | Bottleneck | ~2M |
| PixelTextCrossAttention | 多尺度交叉注意力 | 所有 4 个 stage | ~1M |

### 3.2 训练配置

- **硬件：** RTX 4060 8GB（WSL2），patch 64³，batch_size 1
- **训练：** 300 epochs，lr=1e-4，weight_decay=0.01
- **BERT：** 完全冻结（unfreeze_text_layers=0）
- **Contrastive：** weight=0.0（禁用）
- **no_text_ratio：** 0.15（15% 批次不使用文本）

### 3.3 结果

Dice 68.48%（text）/ 68.67%（no-text），**delta = -0.19%**，文本不仅无益反而有害。

### 3.4 失败根因分析

**问题一：浅层 FiLM 破坏视觉基础特征。** Encoder 的 Stage 1-2 提取的是边缘、纹理等低级视觉特征。FiLM 在这些层注入 256 维 BERT 嵌入，等于往低级特征里掺入高维语义噪声。TextBraTS 论文（Li et al., 2025）已证明 bottleneck-only 注入比 multi-scale FiLM 高 +1.5% Dice。更深层的原因是 FiLM 本身是为低维信号设计的（如 5-10 维的分类标签），256 维 BERT 嵌入通过 γ·x+β 变换退化为全局通道缩放，无法选择性路由空间相关信息。

**问题二：因果 MambaFusion 的单向偏差。** 文本 token 拼接在图像序列前面，因果 SSM 只有 text→image 的单向信息流，图像无法反向影响文本理解，文本信号占主导，图像特征被压缩。正确做法应使用标准双向 cross-attention（Q=image, K/V=text）。

**问题三：default_text_embed 污染 text-free 基线。** 当 `use_text=False` 时，模型并非跳过融合层，而是生成一个随机初始化的 `default_text_embed`（加 LayerNorm）送入 MambaFusion + FiLM。这意味着对比实验中的"无文本"组实际是"伪文本"组，with-text vs without-text 的结论不可信。

**问题四：No-text 批次的 contrastive loss 计算错误。** 15% 的训练批次没有真实文本，但仍计算图像特征和 `default_text_embed` 之间的对比损失——等于训练图像编码器去对齐随机噪声，扭曲了图像表示空间。

**问题五：参数-样本比严重失衡。** 24M 参数 / 369 训练样本 = 65,000:1，额外 3.3M 文本参数加剧过拟合风险。

## 4. V3：Bottleneck-Only Cross-Attention

### 4.1 架构设计

V3 完全推翻 V2 的三层注入策略，改为 bottleneck 单点注入，融合参数量从 3.3M 降至 ~0.2M。

```
Encoder (4 stages) ── 原始 skip connections ──→ Decoder
         │
         └── PixelTextCrossAttention（仅 bottleneck）
                 Q=image, K/V=text
                 8 heads, feat_dim=384, text_dim=256
```

### 4.2 V2 → V3 关键变更

| 组件 | V2 | V3 | 变更原因 |
|------|----|----|----------|
| 注入位置 | 所有 4 个 stage | 仅 bottleneck | 浅层不适合语义注入，文献证据支持 |
| 融合类型 | 因果 SSM + FiLM + CrossAttn | 标准双向 CrossAttn | 消除单向偏差，Q=image/K,V=text |
| text-free 路径 | default_text_embed → 融合 | 完全跳过融合层 | 保证对比实验公平 |
| 融合参数量 | ~3.3M | ~0.2M | 小数据集需要轻量融合 |
| Skip connections | FiLM 调制后 | 原始 encoder 特征 | 保护低级特征不被污染 |
| Patch size | 64³ | 128³ | A100 显存支持更大 patch（覆盖原图 23% vs 3%） |
| embed_dim | 48 | 96 | 更大容量 |

### 4.3 训练配置

- **硬件：** A100 40GB，patch 128³，batch_size 1-4
- **训练：** 150-200 epochs，lr=1e-4，weight_decay=1e-5
- **BERT：** 完全冻结
- **Contrastive：** weight=0.0（禁用）
- **Deep supervision：** 禁用

### 4.4 结果

Dice 88.15%（text）/ 88.51%（no-text），**delta = -0.36%**。

绝对 Dice 从 V2 的 68% 大幅提升至 88%，主要归功于：128³ patch 覆盖率提升（23% vs 3%）、A100 大显存支持更大模型（embed_dim 96）、更大 batch。但文本贡献仍为负，说明 bottleneck-only 的融合架构方向正确，但仅靠融合机制不足以让文本产生正向效果。

### 4.5 遗留问题诊断

| # | Root Cause | 严重性 |
|---|-----------|--------|
| 1 | 融合仅在 bottleneck（4³ = 64 tokens），信息瓶颈太窄 | 高 |
| 2 | contrastive_weight=0，文本无显式对齐信号 | 高 |
| 3 | PubMedBERT 完全冻结，无法适应分割任务 | 中 |
| 4 | Cross-attention 的 out_proj 零初始化导致初期不激活 | 中 |
| 5 | BraTS 文本描述同质化（报告间区分度低） | 低-中 |

## 5. V4 / V4.1：Multi-Scale Fusion + Unfreeze BERT

### 5.1 架构设计

V4 在 V3 的基础上将融合点从 bottleneck 扩展到 Stage 1/2/3 三个尺度，同时解冻 PubMedBERT 最后 2 层。

```
Encoder 4 stages
    Stage 0 (32K tokens) → 直接送 Decoder（VRAM 限制，不做文本融合）
    Stage 1 (4K tokens)  ──┐
    Stage 2 (512 tokens)   ├─→ MultiScalePixelTextAttention ──→ Decoder
    Stage 3 (64 tokens)    │   （3 路并行双向 cross-attention）
                           └── PubMedBERT（最后 2 层解冻）
```

MultiScalePixelTextAttention 为每个 stage 独立创建一个 PixelTextCrossAttention 模块，stage 维度分别对应 encoder 各层的特征维度。Stage 0 因 32K tokens 的 cross-attention 计算量过大而被排除。

### 5.2 V3 → V4 关键变更

| 变更 | V3 → V4 | 变更原因 |
|------|---------|----------|
| 融合位置 | 仅 bottleneck (64 tokens) | Stage 1,2,3 (4K+512+64 tokens) | 64 tokens 太窄，扩展交互面 |
| 融合类型 | 单个 CrossAttn | MultiScalePixelTextAttention（3 路并行） | 多尺度特征都能接收文本信号 |
| BERT 状态 | 全冻结 | 最后 2 层解冻 | 适应分割任务的语义需求 |
| embed_dim | 96 | 48 | 多尺度融合增加计算，降维控制 VRAM |
| Deep supervision | 禁用 | 启用 [0.2, 0.1, 0.05] | 中间 decoder 层也提供梯度信号 |
| batch_size | 1-4（不固定） | 4（固定） | A100 显存充足，梯度更稳定 |
| grad_accumulation | 4 | 1 | 真实 batch=4 不再需要累积 |

### 5.3 训练配置

- **硬件：** A100 40GB，patch 128³，batch_size 4
- **训练：** 200 epochs（early stopping at 166），lr=1e-4，weight_decay=0.01
- **BERT：** 最后 2 层解冻（unfreeze_text_layers=2）
- **Contrastive：** weight=0.0（仍禁用，isolate fusion variable first）
- **Deep supervision：** weights [0.2, 0.1, 0.05]
- **AMP：** bf16 on A100

### 5.4 结果

**Full-volume evaluation（95 test cases，sliding window）：**

| 指标 | With Text | No Text | Delta |
|------|-----------|---------|-------|
| ET | 0.7663 | 0.7679 | -0.16% |
| TC | 0.8411 | 0.8415 | -0.04% |
| WT | 0.8875 | 0.8860 | +0.15% |
| **Mean** | **0.8316** | **0.8318** | **-0.02%** |

文本的负面影响从 V3 的 -0.36% 缩小到 -0.02%，改善了 94%，几乎达到中性。绝对 Dice 从 88% 降至 83% 主要是 embed_dim 96→48 带来的模型容量下降。V4.1 证明：多尺度融合+解冻 BERT 方向正确，但还缺少显式的文本-图像对齐损失来让文本产生正向贡献。

## 6. V4.2：ForegroundContrastiveLoss + Warmup

### 6.1 设计动机

V4.1 的 contrastive_weight=0 意味着训练过程没有任何显式信号告诉模型"文本嵌入应该和图像嵌入对齐"。Cross-attention 虽然提供了融合通道，但缺少对齐损失，模型可能学会忽略文本输入（因为分割 loss 本身不依赖文本质量）。

V4.2 的设计经过 Codex (GPT-5.4) 的对抗性质询（DEBATE），原始蓝图被 Codex 指出多处严重问题后大幅修订。

### 6.2 Codex DEBATE 关键质询与修订

| Codex 质询 | 原蓝图问题 | 修订方案 |
|------------|-----------|----------|
| 4³ 分辨率下 pixel-level contrastive 是伪命题 | 每个 token 对应 32³ 体素块，不是像素级 | 改为前景加权全局 contrastive |
| cosine→BCE 数学不兼容 | cosine 值域 [-1,1]，BCE 期望 [0,1] | 改用标准 symmetric InfoNCE |
| B=4 时 semantic matching 太弱 | 仅 3 个负样本，batch-level KL 太嘈杂 | 砍掉 SemanticMatchingLoss |
| Case-level text vs patch crop = 标签噪声 | 文本描述整个病例，但训练用 128³ 裁剪 | GT mask 加权池化 + 无前景 fallback |
| 梯度干扰风险 | 从 epoch 1 启用两个新 loss | warmup 30 epochs + text features detach |

### 6.3 ForegroundContrastiveLoss 设计

传统 InfoNCE 对全部 bottleneck tokens 做 mean pooling，大量背景 token 稀释肿瘤信号。ForegroundContrastiveLoss 用 GT mask 下采样到 4³ 分辨率，给前景 token 赋予更高权重再池化，让图像嵌入聚焦于肿瘤区域。

```
pixel_feat [B, 64, 384]  ──→  前景加权池化  ──→  img_proj  ──→  L2 normalize
                               ↑                                      │
GT mask [B,128,128,128]  ──→  adaptive_max_pool3d(4³)                 │
                               → fg_weights [B, 64]                   │
                                                                      ↓
text_feat [B, 256]  ──→  L2 normalize (.detach())  ──→  symmetric InfoNCE
```

| 设计选择 | 原因 |
|----------|------|
| GT mask 加权池化 | 聚焦肿瘤区域，避免背景 token 稀释 |
| adaptive_max_pool3d | 保留前景存在性（比 avg_pool 更适合稀疏 mask） |
| 无前景时 fallback 到 uniform | 50% 裁剪为随机位置，可能不含肿瘤 |
| text_feat.detach() | 防止对比损失梯度回流干扰 BERT 训练 |
| symmetric InfoNCE | 数学上比 BCE 更适合嵌入空间对齐 |
| img_proj (Linear + LayerNorm) | 将 384 维图像特征投影到 256 维文本空间 |

### 6.4 Contrastive Warmup 调度

| Epoch 区间 | contrastive_weight | 设计意图 |
|------------|-------------------|----------|
| 0–30 | 0.0 | 让分割 loss 先稳定 bottleneck 特征表示 |
| 30–60 | 线性 0→0.05 | 渐进引入对比信号，避免梯度干扰 |
| 60+ | 0.05 | 全强度对齐 |

### 6.5 模型接口变更

V4.1 前向传播返回 3-tuple `(seg_output, img_global, text_global)`；V4.2 扩展为 4-tuple `(seg_output, img_global, text_global, pixel_feat)`，其中 `pixel_feat = decoder_features[-1]` 是 bottleneck tokens `[B, 64, 384]`，供 ForegroundContrastiveLoss 使用。

### 6.6 文件变更清单

| 文件 | 操作 | 变更内容 |
|------|------|----------|
| losses/contrastive_loss.py | 新增类 | ForegroundContrastiveLoss (~45 行) |
| losses/__init__.py | 修改 | 导入 ForegroundContrastiveLoss，forward 接收 pixel_feat/mask |
| models/textmamba3d.py | 修改 | return_features 时额外返回 pixel_feat |
| train.py | 修改 | 解包 4-tuple，传 mask 给 loss，warmup 逻辑 |
| configs/textbrats_a100.yaml | 修改 | contrastive_weight: 0.05, warmup: 30 |

### 6.7 预期效果

- 文本 delta 从 -0.02% 转正（期望 +0.5% ~ +1.5%）
- VRAM 增量 < 50MB（仅多返回一个 [B, 64, 384] tensor + 一个小投影层）
- 训练时间增量可忽略

## 7. 超参数演进对比

| 超参数 | V2 | V3 | V4/V4.1 | V4.2 |
|--------|----|----|---------|------|
| embed_dim | 48 | 96 | 48 | 48 |
| patch_size | 64³ | 128³ | 128³ | 128³ |
| batch_size | 1 | 1-4 | 4 | 4 |
| lr | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| weight_decay | 0.01 | 1e-5 | 0.01 | 0.01 |
| epochs | 300 | 150-200 | 200 | 200 |
| warmup_epochs | 10 | 5 | 10 | 10 |
| contrastive_weight | 0.0 | 0.0 | 0.0 | 0.05 (warmup) |
| unfreeze_text_layers | 0 | 0 | 2 | 2 |
| deep_supervision | No | No | Yes | Yes |
| no_text_ratio | 0.15 | — | 0.15 | 0.15 |
| gradient_checkpointing | Yes | Yes | Yes | Yes |

## 8. 损失函数演进

| 组件 | V2 | V3 | V4/V4.1 | V4.2 |
|------|----|----|---------|------|
| Dice Loss | 类别加权 | 类别加权 | 类别加权 | 类别加权 |
| CE Loss | 类别加权 | 标准 | 类别加权 | 类别加权 |
| Edge Loss | 3D Sobel | 3D Sobel | 3D Sobel | 3D Sobel |
| Contrastive | 禁用 | 禁用 | 禁用 | ForegroundContrastiveLoss + warmup |
| Deep Supervision | — | — | Dice + CE (3 aux heads) | Dice + CE (3 aux heads) |

类别权重 `[0.25, 3.0, 1.0, 4.0]` 对应 [背景, ET, TC, WT]，反映 BraTS 类别不平衡：背景占绝大多数体素，ET 和 WT 体积小但临床重要。

## 9. 关键文献支撑

| 文献 | 发现 | 对 TextMamba3D 的影响 |
|------|------|----------------------|
| TextBraTS (Li et al., 2025) | Bottleneck-only 双向 cross-attention +1.5% Dice | V3 架构依据 |
| Lemay et al. (2021) | FiLM 适合低维分类元数据 (+5.1% Dice) | 解释 V2 FiLM 失败原因 |
| Neural Field Conditioning (2023) | Cross-Attention > FiLM > Concatenation | 融合机制选择依据 |
| CLIP (Radford et al., 2021) | InfoNCE 在小 batch 下仍可工作 | V4.2 B=4 对比学习可行性 |

## 10. 附录：文件索引

| 文件 | 内容 |
|------|------|
| docs/architecture_v2_postmortem.md | V2 失败详细分析 |
| docs/text_guidance_improvement_plan.md | 4 步改进计划及 V4 结果 |
| docs/planning/research_summary.md | 文献调研总结 |
| models/textmamba3d.py | 主模型（当前 V4.2） |
| models/fusion.py | MultiScalePixelTextAttention |
| losses/contrastive_loss.py | ContrastiveLoss + ForegroundContrastiveLoss |
| losses/__init__.py | CombinedLoss 组合损失 |
| configs/textbrats_a100.yaml | A100 训练配置（当前 V4.2） |
| TextMamba3D_A100_V4.1.ipynb | V4.1 baseline notebook |
| TextMamba3D_A100_V4.2.ipynb | V4.2 训练 notebook |
| TextMamba3D_A100_V4.3.ipynb | V4.3 训练 notebook |
| TextMamba3D_A100_V4.4.ipynb | V4.4 训练 notebook |
| docs/v44_changelog.md | V4.4 详细变更日志 |
| docs/papers/18.TextBraTS_MICCAI2025/ | TextBraTS 论文阅读笔记 |

---

## 11. V4.3：PWAM 乘法融合 + 多辅助损失（退步）

### 11.1 设计动机

受 LAVT (CVPR 2022) 启发，V4.3 将 V4.1 的加法 cross-attention 替换为乘法 PWAM (Pixel-Word Attention Module)，并加入三个辅助损失来强化文本引导：

| 新模块 | 来源 | 作用 |
|--------|------|------|
| PWAM3D | LAVT | 乘法融合：vis × lang + Tanh 门控 |
| TextToVoxelLoss | CRIS | 文本条件的逐体素 BCE（3 region targets） |
| TextNecessityLoss | 自创 | 强制 Dice(text) > Dice(no-text) + margin |
| EmbeddingPerturbation | RLEG | 嵌入空间的方向扰动增强 |

### 11.2 Codex DEBATE（10 issues resolved）

蓝图经过 Codex (GPT-5.4) 对抗性质询，修正了 10 个关键问题，包括：768 vs 256 维度不匹配、mask shape、T2VLoss 用错 tensor、OOM 风险等。详见 `docs/plans/v43_blueprint_locked.md`。

### 11.3 结果

**Full-volume evaluation（94 test cases）：**

| 指标 | With Text | No Text | Delta |
|------|-----------|---------|-------|
| ET | 0.7574 | 0.7657 | **-0.83%** |
| TC | 0.8409 | 0.8413 | -0.04% |
| WT | 0.8975 | 0.8965 | +0.10% |
| **Mean** | **0.8319** | **0.8345** | **-0.26%** |

相比 V4.1 的 -0.02%，V4.3 退步至 -0.26%。ET 受损最严重 (-0.83%)。HD95 边界指标有微弱改善。Val dice 0.89 vs test dice 0.83 有过拟合迹象。

### 11.4 失败根因

1. **PWAM 的乘法融合放大噪声**：V4.1 的加法残差 `residual + out_proj(attn)` 天然安全（zero-init → 初始为 identity）。PWAM 的 `gate * (vis × lang) + vis` 是乘法交互，如果 lang 信号质量不高（ET 仅 3/368 报告提及），乘法会放大噪声而非信号。
2. **辅助损失太弱**：T2VLoss (0.1) + NecessityLoss (0.05) 合计仅占总 loss 的 ~5%，不足以有效约束模型使用文本。
3. **复杂度爆炸**：4 个新模块 + Phase A/B 双阶段训练 + dual forward（null text），相对 V4.1 引入过多变量，无法隔离改善来源。

### 11.5 教训

**第 5 条红线确认：乘法融合 (PWAM) 在当前 BraTS 文本质量下不如加法残差。**

---

## 12. V4.4：SeqCA 两步交叉注意力

### 12.1 设计动机

TextBraTS (MICCAI 2025) 在同一 BraTS2020 数据集上用 SeqCA 实现 +1.5% Dice 正向提升。核心发现：**Q/KV 方向比融合形式（加法/乘法）更重要**。

V4.1 → V4.4 的改动极小（仅 2 行 import + 148 行新类），遵循"一次改一个变量"原则。

### 12.2 SeqCA vs PixelTextCrossAttention

| 设计选择 | V4.1 (PixelTextCA) | V4.4 (SeqCA) |
|----------|-------------------|--------------|
| **Step 1 Q/KV** | Image=Q, Text=KV | **Text=Q, Image=KV** |
| Step 2 | 无 | Image=Q, Refined=KV |
| 输出形状 | [B, N, D] | [B, N, D]（相同） |
| 初始化 | out_proj zero-init | Step 2 i2t_out zero-init |
| 残差连接 | residual + out_proj(attn) | residual + joint |
| 融合位置 | stages 1,2,3 | stages 1,2,3（不变） |

### 12.3 文件变更

| 文件 | 操作 | 改动量 |
|------|------|--------|
| models/fusion.py | 新增 SequentialCrossAttention + MultiScaleSeqCA | +148 行 |
| models/textmamba3d.py | 替换 import（2 行） | 2 行 |
| configs/textbrats_v6.yaml | 新建 | 新文件 |
| TextMamba3D_A100_V4.4.ipynb | 新建 Colab notebook | 18 cells |

### 12.4 预期效果

- Text delta: -0.02% → 正向（基于 TextBraTS 同数据集验证 +1.5%）
- ET Dice 改善（V4.3 退步最大的指标）
- VRAM 增量 ~73MB（attention maps），A100 40GB 无压力

### 12.5 训练配置

与 V4.1 相同（textbrats_v6.yaml 基于 textbrats_a100.yaml），200 epochs, lr=1e-4, bs=4, contrastive_weight=0.0（Phase 1 隔离 SeqCA 效果）。

### 12.6 结果

待训练。

---

## 13. 超参数演进对比（完整版）

| 超参数 | V2 | V3 | V4.1 | V4.2 | V4.3 | V4.4 |
|--------|----|----|------|------|------|------|
| embed_dim | 48 | 96 | 48 | 48 | 48 | 48 |
| patch_size | 64³ | 128³ | 128³ | 128³ | 128³ | 128³ |
| batch_size | 1 | 1-4 | 4 | 4 | 4 | 4 |
| fusion | FiLM+Mamba+CA | BottleneckCA | MultiScaleCA | MultiScaleCA | PWAM3D | **SeqCA** |
| contrastive | 0.0 | 0.0 | 0.0 | 0.05 | 0.0 | 0.0 |
| epochs | 300 | 200 | 200 | 200 | 300 | 200 |
| unfreeze_text | 0 | 0 | 2 | 2 | 2 | 2 |
| **delta** | **-0.19%** | **-0.36%** | **-0.02%** | **待训练** | **-0.26%** | **待训练** |
