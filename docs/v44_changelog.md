# TextMamba3D V4.4 — SeqCA 变更日志

> 日期：2026-03-14 | 基于 TextBraTS (MICCAI 2025) 启发

## 1. 变更摘要

V4.4 是对 V4.1（历史最佳 delta = -0.02%）的定向改进，核心变更只有一处：**将 PixelTextCrossAttention 替换为 SequentialCrossAttention (SeqCA)**，翻转 Q/KV 方向。

| 维度 | V4.1 (当前) | V4.4 (新) |
|------|-------------|-----------|
| 融合模块 | PixelTextCrossAttention | SequentialCrossAttention |
| Step 1 | Image=Q, Text=KV | **Text=Q, Image=KV** |
| Step 2 | 无（单步） | **Image=Q, Refined=KV** |
| 融合位置 | Stages 1,2,3 | Stages 1,2,3（不变） |
| 损失函数 | Dice + CE + Edge | Dice + CE + Edge（不变） |
| 初始化策略 | out_proj zero-init | Step 2 i2t_out zero-init（不变） |

## 2. 理论依据

TextBraTS (MICCAI 2025) 在同一 BraTS2020 数据集上，用 SwinUNETR + SeqCA 实现了 **+1.5% Dice** 正向提升（83.8% → 85.3%）。核心洞察：

**Text=Query 比 Image=Query 更有效。**

- V4.1 (Image=Q): 图像在文本中寻找信息。如果文本信息贫瘠（ET 仅 3/368 报告提及），图像只能找到噪声。
- V4.4 (Text=Q): 文本主动定位自己描述的区域。即使文本简短，也能精准指向对应区域。

SeqCA 两步机制：
1. **Step 1 (T2I)**: Text=Q, Image=KV — 文本"询问"图像：我描述的位置/特征在图像的哪里？
2. **Step 2 (I2T)**: Image=Q, Refined=KV — 图像用文本过滤后的视觉信息增强自身

输出形状与 V4.1 完全一致 [B, N, D]，decoder 无需任何修改。

## 3. 修改文件清单

| 文件 | 操作 | 改动量 |
|------|------|--------|
| `models/fusion.py` | 新增 SeqCA + MultiScaleSeqCA | +148 行 |
| `models/textmamba3d.py` | 替换 import（2 行） | 2 行 |
| `configs/textbrats_v6.yaml` | 新建 V4.4 config | 新文件 |
| `TextMamba3D_A100_V4.4.ipynb` | 新建 Colab notebook | 18 cells |

**未修改**：train.py, losses/, encoder_3d.py, decoder_3d.py, text_encoder.py, mamba_block.py

## 4. SeqCA 实现细节

### 4.1 模块结构

```
SequentialCrossAttention(feat_dim, text_dim=256, num_heads=4)
├── text_proj: Linear(256 → feat_dim) + LayerNorm
├── Step 1 (T2I):
│   ├── t2i_norm_q, t2i_norm_kv: LayerNorm (pre-norm)
│   ├── t2i_q, t2i_k, t2i_v: Linear(feat_dim → feat_dim)
│   └── t2i_out: Linear + LayerNorm (正常初始化)
└── Step 2 (I2T):
    ├── i2t_norm_q, i2t_norm_kv: LayerNorm (pre-norm)
    ├── i2t_q, i2t_k, i2t_v: Linear(feat_dim → feat_dim)
    └── i2t_out: Linear (zero-init → identity start)
```

### 4.2 维度验证 (embed_dim=48, img_size=128³)

| Stage | Image tokens | Image dim | Text tokens | Text proj | Step1 attn | Step2 attn |
|-------|-------------|-----------|-------------|-----------|------------|------------|
| 1 | 4096 (16³) | 96 | 256 | 256→96 | [B,4,256,4096] | [B,4,4096,256] |
| 2 | 512 (8³) | 192 | 256 | 256→192 | [B,4,256,512] | [B,4,512,256] |
| 3 | 64 (4³) | 384 | 256 | 256→384 | [B,4,256,64] | [B,4,64,256] |

### 4.3 VRAM 影响

attention map 额外占用 ~73MB（bs=4, bf16），A100 40GB 完全可行。

### 4.4 Mask 传递策略

- **Step 1 (T2I)**: 不传 mask。Text=Q 查询 Image=KV，image tokens 来自 Mamba encoder 固定输出，无 padding。
- **Step 2 (I2T)**: 传 text_mask。Image=Q 查询 Refined=KV，refined 长度 = M（text tokens），pad 位置对应无效 refined tokens。

### 4.5 Zero-Init 策略

仅 Step 2 的 `i2t_out` 做零初始化。训练初期：
- Step 1 正常运作（产出 refined features ≠ 0）
- Step 2 输出 = i2t_out(attn) = 0（因为权重全零）
- 最终输出 = residual + 0 = residual（identity）

这确保 SeqCA 在训练初期不干扰视觉特征，逐步学习有意义的文本引导。

## 5. 历史教训对照

| # | 红线 | V4.4 是否遵守 |
|---|------|--------------|
| 1 | 不用 CausalMambaFusion | 是 — 使用 cross-attention |
| 2 | 不在浅层注入文本 | 是 — Stage 0 (32K tokens) 不做融合 |
| 3 | 不用 null/default text embed | 是 — text-free = bypass fusion |
| 4 | Bottleneck-only 不够 | 是 — 保留多尺度 stages 1,2,3 |
| 5 | 乘法融合(PWAM)比加法差 | 是 — 使用加法残差（residual + joint） |
| 6 | V4.2 ForegroundContrastiveLoss 未测试 | Phase 2 中测试 |

## 6. Config 变更 (textbrats_v6.yaml)

基于 textbrats_a100.yaml (V4.2)，移除 V4.3 特有字段：

| 字段 | V4.3 (textbrats_v5) | V4.4 (textbrats_v6) |
|------|---------------------|---------------------|
| use_pwam | true | 删除 |
| emb_perturbation | true | 删除 |
| t2v_weight | 0.1 | 删除 |
| necessity_weight | 0.05 | 删除 |
| necessity_warmup | 30 | 删除 |
| contrastive_weight | 0.0 | 0.0（Phase 1 隔离 SeqCA 效果） |
| epochs | 300 | 200 |
| patience | 50 | 40 |

## 7. 后续计划（基于 v4.4 训练结果更新）

v4.4 已实现 +0.55% 正向 delta（TC +1.05%, WT +0.74%），**ET -0.16% 是当前唯一短板**。

| Phase | 变更 | 预期效果 | 状态 |
|-------|------|---------|------|
| 1 | SeqCA (Text=Q) | delta: +0.55% ✅ | ✅ 完成 |
| 2 | ET-Enriched Text（从 mask 提取 ET 特征 → 模板化文本） | ET delta 转正 | 下一步 |
| 3 | LaPael 式早期层 embedding 扰动 | 文本多样性 | 待定 |
| 4 | ForegroundContrastiveLoss（锦上添花） | 额外 +0.2~0.5% | 可选 |

**已放弃：** PWAM（v4.3 退步）、TextNecessityLoss（v4.3 无效）、TextToVoxelLoss（v4.3 无效）。详见 `experiment_log.md`。

## 8. Checkpoint 不兼容

V4.1/V4.2 checkpoint 不兼容（MultiScalePixelTextAttention → MultiScaleSeqCA 参数名不同），必须从头训练。
