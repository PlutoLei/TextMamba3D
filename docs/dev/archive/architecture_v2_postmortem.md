# TextMamba3D v2 架构复盘：多尺度文本注入的失败教训

**日期：** 2026-03-08
**训练环境：** RTX 4060 8GB (WSL2), embed_dim=48, 300 epochs
**最终性能：** Dice 68.48% (with text) / 68.67% (no text)
**结论：** 文本注入净负面，no-text 反超 with-text 0.19%

---

## 失败架构概述

```
Encoder (4 stages) ──┬── MultiScaleFiLM (text 调制所有 4 层)
                     │       ↓
                     └── MambaFusion (causal text-first SSM)
                             ↓
                          Decoder
```

### 三层文本注入机制

| 层 | 机制 | 参数 | 问题 |
|----|------|------|------|
| MultiScaleFiLM | γ/β 仿射变换调制 4 个 encoder stage | ~0.3M | 浅层特征被文本信号污染 |
| MambaFusion | 文本序列拼在图像前，causal SSM 处理 | ~2M | 文本主导注意力，图像特征被压缩 |
| PixelTextAttn (multi-scale) | 4 层 cross-attention | ~1M | 与 FiLM 重复，计算浪费 |

**总文本注入参数：** ~3.3M / 24M total = 13.8% 参数用于文本注入

---

## 五个具体失败点

### 1. 浅层 FiLM 破坏视觉基础特征

**现象：** Stage 1-2 的 encoder 提取边缘、纹理等低级特征，这些特征是空间基元，与语义文本无关。FiLM 在这些层施加 γ·x + β 调制，引入了与空间结构不匹配的噪声。

**证据：** TextBraTS 论文实验显示 bottleneck-only 注入比多尺度注入高 +1.5% Dice。

**教训：** 文本调制应仅作用于语义丰富的高层特征（bottleneck），而非全部 encoder 层。

### 2. Causal MambaFusion 的文本优先偏差

**现象：** MambaFusion 将文本 token 拼接在图像 token 前面，通过 causal SSM 处理。由于 SSM 的因果性质，文本信息单向流入图像但图像无法反向影响文本，造成文本信号主导融合输出。

**教训：** 用标准双向 cross-attention（Q=image, K/V=text）替代因果融合。

### 3. default_text_embed 污染 text-free 基线

**现象：** 当 `use_text=False` 时，模型生成一个学习到的 `default_text_embed`（随机初始化 + LayerNorm），将其送入 MambaFusion 和 FiLM。这意味着"无文本"模式实际上是"有一个学习到的伪文本"模式，而非真正的纯视觉基线。

**后果：** 无法公平对比 with-text vs without-text 性能，实验结论不可信。

**教训：** text-free 路径必须完全绕过融合模块（bypass），不能用默认嵌入替代。

### 4. Contrastive Loss 在无文本 batch 上的错误监督

**现象：** `no_text_ratio=0.15` 意味着 15% 的 batch 不使用真实文本。但训练代码仍然对这些 batch 计算 contrastive loss（img_feat vs default_text_feat），导致图像编码器被训练去对齐一个随机的默认嵌入，而非有意义的文本语义。

**后果：** 图像表征空间被扭曲，contrastive loss 成为一个混淆信号而非有用的正则化。

**教训：** 无文本 batch 应跳过 contrastive loss，`return_features=True` 应仅在 `use_text=True` 时启用。

### 5. 过参数化的融合 vs 数据量不匹配

**现象：** 24M 可训练参数 / 369 个训练样本 = 65,000:1 的参数-样本比。即使纯视觉部分也过参数化，加上 3.3M 的文本注入参数进一步恶化了过拟合风险。

**教训：** 在小数据集上，融合机制应尽可能轻量。单个 cross-attention 层（~0.2M）足够。

---

## 修复方案（v3 架构）

```
Encoder (4 stages) ── raw skip connections ──→ Decoder
                     │
                     └── PixelTextCrossAttention (bottleneck only)
                             Q=image, K/V=text
                             8 heads, feat_dim=384, text_dim=256
```

| 对比 | v2 (失败) | v3 (修复) |
|------|----------|----------|
| 文本注入层数 | 4 (全部 encoder stage) | 1 (仅 bottleneck) |
| 融合机制 | Causal SSM + FiLM + CrossAttn | 标准 CrossAttn |
| text-free 路径 | default_text_embed → 融合 | 完全 bypass |
| 无文本 batch 的 contrastive | 错误计算 | 跳过 |
| 融合参数量 | ~3.3M | ~0.2M |
| Skip connections | FiLM 调制后 | 原始 encoder 特征 |

---

## Checkpoint 存档

| 文件 | Epoch | Dice (text) | Dice (no-text) | 大小 |
|------|-------|-------------|----------------|------|
| best.pth | 84 | 68.48% | 68.67% | 693M |
| best_no_text.pth | 84 | 67.17% | 68.67% | 693M |
| last.pth | 85 | 68.48% | 68.67% | 693M |

**注意：** 这些 checkpoint 包含 MambaFusion + MultiScaleFiLM 权重，与 v3 架构不兼容，无法 `load_state_dict`。

---

## 核心教训总结

1. **文本注入不是越多越好** — 多尺度 FiLM 在 4 层全部注入文本，反而破坏了低级视觉特征。Bottleneck-only 是正确粒度。
2. **因果融合不适合多模态** — SSM 的因果性质造成信息流不对称。Cross-attention 是多模态融合的标准选择。
3. **评估完整性是实验的前提** — default_text_embed 让 text-free 基线失去意义，所有消融实验的结论都不可信。
4. **训练信号的一致性** — Contrastive loss 在无文本 batch 上会成为有害信号。每个 loss 组件都需要确认其监督目标在当前 batch 条件下是有效的。
5. **过参数化 + 小数据 = 融合层无法学习有效表征** — 369 个样本不足以训练 3.3M 的多层融合机制。
