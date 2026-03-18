# LAVT: Language-Aware Vision Transformer for Referring Image Segmentation

> CVPR 2022 | Yang et al. | 452+ citations
> GitHub: https://github.com/yz93/LAVT-RIS
> PDF: https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Language-Aware_Vision_Transformer_for_Referring_Image_Segmentation_CVPR_2022_paper.pdf

## 一句话总结

在 vision encoder（Swin Transformer）的每一层**内部**注入语言特征，通过乘性融合 + 门控残差实现不可绕过的文本引导，比所有 late-fusion 方法高 +1.84 oIoU。

## 核心贡献

1. 提出 **early fusion** 范式：语言特征在 encoder 内部与视觉特征交互，而非 encoder 之后
2. 设计 **PWAM (Pixel-Word Attention Module)**：跨模态注意力 + 乘性融合 + 门控残差
3. 在 RefCOCO/RefCOCO+/G-Ref 上取得 SOTA，比之前最佳方法高 +6.6~8.6%

## 核心机制：PWAM

### 数学公式

**Step 1 — 跨模态注意力（Q=视觉, K/V=语言）：**

```
Q = InstanceNorm(Conv1d(V_visual))           # [B, C, N]  N=H×W
K = Conv1d(L_tokens)                          # [B, C, T]  T=词数
V = Conv1d(L_tokens)                          # [B, C, T]

A = softmax(Q^T · K / √C) · V^T              # [B, N, C]  逐像素的语言特征
G = InstanceNorm(Conv1d(A))                   # output projection
```

**Step 2 — 乘性融合（关键机制）：**

```
V_proj = GELU(Conv1d(V_visual))               # 视觉投影
F = GELU(Conv1d(V_proj × G))                  # element-wise 乘法 ← 不可忽略
```

**Step 3 — 门控残差：**

```
S = Tanh(Linear(ReLU(Linear(F))))             # gate ∈ [-1, +1]
Output = S × F + V_visual                     # 门控残差连接
```

### 为什么是乘性而非加性

| 方式 | 公式 | 模型能否忽略文本 |
|------|------|----------------|
| Additive (TextMamba3D v4) | `out = vis + attn(vis, text)` | 能：attention weight → 0 |
| **Multiplicative (LAVT)** | `out = gate × (vis_proj × lang_feat) + vis` | **不能**：vis × lang 为零意味着整个分支为零，gate 需要学到恒等变换才能绕过，但 Tanh 限制了范围 |

### 插入位置

在 Swin Transformer 的 **4 个 stage 各 1 个 PWAM**，位于每个 stage 的最后（所有 Swin block 之后、下采样之前）。

## 关键消融实验

### Early Fusion vs Late Fusion（同 Swin-B + BERT 骨架）

| 方法 | 融合策略 | oIoU | mIoU |
|------|---------|------|------|
| LTS | Late (locate-then-segment) | 69.94 | 70.56 |
| EFN | Encoder-level (不同实现) | 70.76 | 72.95 |
| VLT | Late decoder (cross-modal Transformer) | 70.89 | 71.98 |
| **LAVT** | **Early encoder fusion** | **72.73** | **74.46** |

**结论：** Early fusion > Late fusion，差距 +1.84 oIoU。

### 融合方式对比

| 方式 | oIoU | 备注 |
|------|------|------|
| Replacement (无门控) | 训练发散 | 破坏预训练权重 |
| Concatenation (无门控) | 60.52 | 严重退化 |
| Sum (无门控) | 72.24 | 可用但不最优 |
| **Sum + Language Gate** | **72.73** | 最佳 |

**门控贡献：+0.49 oIoU**；更关键的是门控让训练稳定。

### InstanceNorm 的重要性

| 配置 | oIoU |
|------|------|
| 无 normalization | 70.66 |
| BatchNorm | 71.50 |
| LayerNorm | 71.24 |
| **InstanceNorm** | **72.73** |

**InstanceNorm 贡献 +2.07 oIoU**，必须保留。

## 训练细节

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW, weight_decay=0.01 |
| 学习率 | 5e-5, polynomial decay (power=0.9) |
| Epochs | 40 |
| Batch size | 32 (8/GPU × 4 GPU) |
| 图像尺寸 | 480 × 480 |
| Vision backbone | Swin-Base (ImageNet-22K pretrained) |
| BERT | bert-base-uncased, **fine-tuned** (非冻结) |
| Loss | Weighted CE, weights [0.9, 1.1] |
| Data augmentation | 无 |

## 参数量

PWAM 新增参数估算（Swin-Base, C_i = 128/256/512/1024）：

| 组件 | 每 stage 参数 |
|------|-------------|
| q_proj + InstanceNorm | C² |
| k_proj | 768 × C |
| v_proj | 768 × C |
| out_proj + InstanceNorm | C² |
| vis_proj | C² |
| mm_proj | C² |
| gate (2 × Linear) | 2C² |
| **合计** | **~8C² + 2×768×C** |

4 个 stage 总计 **~14M**，占总模型 (<7%) 的极小比例。

## 对 TextMamba3D 的适配要点

### 直接可迁移

1. **PWAM 是架构无关的** — 操作在 flattened [B, N, C] 特征上，不依赖 self-attention
2. **门控残差连接** — 保护预训练 SSM 权重不被破坏
3. **InstanceNorm** — 必须保留

### 需要注意的差异

| 方面 | LAVT (Swin) | TextMamba3D (Mamba) |
|------|------------|-------------------|
| 语言信号传播 | Self-attention 自然传播到全局 | SSM 按序列方向传播，可能不足 |
| 建议 | 每个 stage 1 个 PWAM | **考虑每个 block 后都注入**，弥补 SSM 缺乏全局 attention |
| 维度 | 2D: [B, H×W, C] | 3D: [B, D×H×W, C]，cross-attention 计算量随 N 线性增长 |
| Stage 0 | 56×56=3136 tokens | 32×32×32=32768 tokens，**必须排除** |

### embed_dim=48 的参数估算

| Stage | vis_dim | PWAM 参数 |
|-------|---------|----------|
| 1 | 96 | ~150K |
| 2 | 192 | ~450K |
| 3 | 384 | ~1.5M |
| **合计** | | **~2.1M** |
