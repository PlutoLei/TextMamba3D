# RLEG: Vision-Language Representation Learning with Diffusion-based Embedding Generation

> ICML 2023 | Zhao et al. (Alibaba / Ant Group) | PMLR 202:42247-42258
> PDF: https://proceedings.mlr.press/v202/zhao23l/zhao23l.pdf

## 一句话总结

用预训练的冻结 diffusion model 在 embedding 空间双向生成合成表示，作为额外正样本增强 contrastive learning，小数据集相对提升最高达 +48.2%。

## 核心贡献

1. 提出 **embedding-level augmentation** — 不在输入空间（文本/图像）做增强，而是在 embedding 空间用 diffusion 生成合成表示
2. 双向生成：image→text embedding + text→image embedding
3. 生成的 embedding 作为额外正样本参与 contrastive loss
4. **训练时在线生成，推理零开销**

## 核心机制

### Diffusion Embedding Generator

**架构：** DALL-E 2 的 prior model（12 层 decoder-only Transformer），预训练在 LAION-400M 上。

**前向扩散：**

```
q(v_t | v_0) = N(v_t; √ᾱ_t · v_0, (1 - ᾱ_t) · I)
v_t = √ᾱ_t · v_0 + √(1 - ᾱ_t) · ε,    ε ~ N(0, I)
```

**反向生成（Classifier-Free Guidance）：**

```
ε̃(v_t, t) = w · ε_θ(v_t, t) + (1 - w) · ε_θ(v_t)

v_{t-1} = (1/√α_t) · [v_t - (1-α_t)/√(1-ᾱ_t) · ε̃] + σ_t · ε
```

w = 2.0（guidance weight）

**生成流程：**

```
Text embedding t → Image-to-Text Generator → 合成 text embeddings {t'_1, ..., t'_K}
Image embedding v → Text-to-Image Generator → 合成 image embeddings {v'_1, ..., v'_K}
```

两个方向**同时**进行，每个样本生成 K=4 个合成 embedding。

### 训练目标

**Loss = 标准 contrastive + 生成引导 contrastive**

```
L = (L_i2t + L_t2i) + λ · (L_i2i + L_t2t)
```

**标准 contrastive (image-to-text, 含多正样本)：**

```
L_i2t = -Σ_i (1/|R(i)|) · Σ_{r∈R(i)} log[ exp(v_i^T · t_r / τ) / Σ_j exp(v_i^T · t_j / τ) ]
```

**生成引导 contrastive（合成 embedding = 额外正样本）：**

```
L_i2i = -Σ_i Σ_{k=1}^{K} (1/|R(i)|) · Σ_{r∈R(i)} log[ exp(v_i^T · v'_{rk} / τ) / Σ_j exp(v_i^T · v'_{jk} / τ) ]
```

L_t2t 对称定义。

**关键：** 合成 embedding `v'_{rk}` 是由匹配的 text embedding `t_r` 条件生成的，因此与 `v_i` 构成正样本对。batch 内其他样本的合成 embedding 是负样本。

### 在线增强流程

```
每个 batch:
  1. 编码: image → v, text → t  (可学习 encoder)
  2. 投影: v, t → MLP → embedding space  (可学习 projector)
  3. 生成: v → Generator → {v'_k}  (冻结, K=4, 10 DDIM steps)
           t → Generator → {t'_k}  (冻结, K=4, 10 DDIM steps)
  4. 计算 loss: L_standard + 0.1 × L_generated
  5. 反向传播 (只更新 encoder + projector，不更新 generator)
```

## 关键实验结果

### vs 无增强基线（ImageNet zero-shot）

| 配置 | Top-1 | Top-5 | 提升 |
|------|-------|-------|------|
| CLIP (无增强) | 30.1% | 53.5% | — |
| RLEG K=1 | 36.7% | 61.3% | +6.6% |
| **RLEG K=4** | **39.1%** | **63.8%** | **+9.0%** |

### vs 其他增强方法

| 方法 | 增强类型 | ImageNet Top-1 |
|------|---------|---------------|
| CLIP | 无 | 30.1% |
| SLIP | 自监督多视角 | 32.5% |
| MS-CLIP | 模态共享 contrastive | 34.3% |
| MaskCLIP | Patch-masked image | 36.0% |
| DeCLIP | 多源监督 | 36.2% |
| **RLEG** | **Embedding-level diffusion** | **39.1%** |

RLEG 超越所有输入级增强方法 +2.9~9.0%。

### 小数据集收益（最关键发现）

| 数据集大小 | 无增强 | RLEG K=4 | 相对提升 |
|-----------|--------|----------|---------|
| 3M | 16.4% | 24.3% | **+48.2%** |
| 3M (CC3M) | 18.1% | 25.8% | **+42.5%** |
| 12M | 26.8% | 35.6% | **+32.3%** |
| 15M | 30.1% | 39.1% | **+29.9%** |

**越小的数据集，相对提升越大。** 369 样本的 TextMamba3D 理论上收益应该更大。

### DDIM 步数消融

| 步数 | ImageNet Top-1 |
|------|---------------|
| 5 | 38.7% |
| **10** | **39.1%** |
| 50 | 39.2% |

5 步已经足够，10 步最优，50 步无显著提升。

### Guidance weight 消融

| w | Top-1 |
|---|-------|
| 0.1 | 32.4% |
| 1.0 | 38.3% |
| **2.0** | **39.1%** |
| 5.0 | 38.8% |

### Loss 权重 λ 消融

| λ | Top-1 |
|---|-------|
| 0.01 | 37.9% |
| **0.1** | **39.1%** |
| 1.0 | 38.5% |

## 实践参数总结

| 参数 | 值 |
|------|-----|
| Diffusion 架构 | 12 层 decoder-only Transformer (DALL-E 2 prior) |
| 预训练数据 | LAION-400M |
| Embedding 维度 | 512 (CLIP ViT-B/32) |
| DDIM 步数 | 10 |
| 噪声调度 | 标准 DDPM β_t schedule |
| Guidance weight w | 2.0 |
| 每样本生成数 K | 4 |
| Loss 权重 λ | 0.1 |
| Projector | 2 层 MLP (去掉损失 0.7%) |
| 温度 τ | 可学习 |
| Dynamic thresholding | 是 (following Imagen) |
| Generator 可学习参数 | **0（完全冻结）** |
| 训练开销 | 1.83× CLIP (仅训练时) |
| 推理开销 | **零**（generator 不参与推理） |

## 对 TextMamba3D 的适配要点

### 核心问题：369 样本能否训练 diffusion generator？

**不能直接用原版方案。** RLEG 的 generator 预训练在 LAION-400M (4 亿样本) 上。369 样本无法训练 12 层 Transformer diffusion model。

### 简化方案：Embedding Perturbation

保留 RLEG 的核心思想（embedding 空间增强），但简化实现：

```python
# 方案 1: Gaussian noise (最简单)
perturbed = text_emb + α · N(0, I)    # α = 0.1

# 方案 2: Learned directions (推荐)
directions = nn.Parameter(randn(8, 768))   # 8 个学习方向
coeffs = randn(B, 8)                       # 随机组合
perturbation = coeffs @ directions          # [B, 768]
perturbed = text_emb + perturbation.unsqueeze(1)

# 方案 3: Mixup in embedding space
λ = Beta(0.2, 0.2).sample()
mixed = λ · text_emb_i + (1-λ) · text_emb_j
```

### 推荐优先级

| 方案 | 复杂度 | 预期效果 | 推荐 |
|------|--------|---------|------|
| Gaussian noise | 零参数 | 基本正则化 | 作为 baseline |
| **Learned directions** | ~6K 参数 | 有意义的多样性 | **首选** |
| Embedding mixup | 零参数 | 跨样本信息交换 | 可与其他组合 |
| 完整 RLEG | ~10M+ 参数 | 最高但可能过拟合 | 不推荐(数据不足) |

### 关键 takeaway

1. **Embedding 空间增强 > 输入空间增强** — RLEG 超越所有 input-level 方法
2. **小数据集收益最大** — 直接支持在 TextMamba3D 上的应用
3. **推理零开销** — 增强只在训练时使用
4. **λ=0.1** — 增强 loss 应该是辅助的，不能主导训练
