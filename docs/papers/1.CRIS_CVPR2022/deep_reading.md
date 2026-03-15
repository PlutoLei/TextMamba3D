# CRIS: CLIP-Driven Referring Image Segmentation

> CVPR 2022 | Wang et al.
> GitHub: https://github.com/DerrickWang005/CRIS.pytorch
> PDF: https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_CRIS_CLIP-Driven_Referring_Image_Segmentation_CVPR_2022_paper.pdf
> arXiv: https://arxiv.org/abs/2111.15174

## 一句话总结

利用 CLIP 的多模态知识，通过 vision-language decoder 和 text-to-pixel contrastive learning 实现细粒度文本-像素对齐，其"contrastive loss"本质上是 **per-pixel BCE with text-conditioned dynamic convolution**。

## 核心贡献

1. 提出 **text-to-pixel contrastive learning** — 文本全局特征投影为动态卷积核，在像素特征上做卷积产生逐像素预测
2. 设计 **Vision-Language Decoder** — 3 层 Transformer decoder，视觉 self-attention + 文本 cross-attention
3. 基于 CLIP 的 referring image segmentation SOTA

## 核心机制：Text-to-Pixel Contrastive Loss

### 论文公式 (Eq. 8-10)

```
z_v = Upsample(F_c) · W_v + b_v      # 视觉投影: [N, D]，N=H/4 × W/4
z_t = F_s · W_t + b_t                  # 文本投影: [D]，F_s = CLIP [EOS] token

L_con(i) = {
    -log σ(z_t · z_v^i),           if i ∈ P (前景)
    -log(1 - σ(z_t · z_v^i)),      if i ∈ N (背景)
}

L_con = (1/|P ∪ N|) × Σ L_con(i)
```

### 代码实际实现（关键发现）

**论文的 "contrastive loss" 和 per-pixel BCE 数学上完全等价：**

```python
# 论文公式展开:
# -y·log(σ(s)) - (1-y)·log(1-σ(s))  ← 标准 BCE

# 代码实现 (segmenter.py):
loss = F.binary_cross_entropy_with_logits(pred, mask)
```

**动态卷积替代点积：**

```python
class Projector(nn.Module):
    def forward(self, x, word):
        # 视觉: upsample + projection
        x = self.vis(x)              # [B, 256, 104, 104]

        # 文本 → 动态 3×3 卷积核
        word = self.txt(word)         # [B, 256*3*3 + 1]
        weight = word[:, :-1].reshape(B, C, 3, 3)   # 动态 kernel
        bias = word[:, -1]                            # 动态 bias

        # 逐样本卷积 = 带空间上下文的点积
        out = F.conv2d(x, weight, padding=1, groups=B, bias=bias)
        # [B, 1, 104, 104] ← 逐像素 logits
        return out
```

### 正负样本定义

| | 定义 | 采样策略 |
|---|------|---------|
| 正样本 (P) | GT mask 中的前景像素 | **全部使用**，无采样 |
| 负样本 (N) | GT mask 中的背景像素 | **全部使用**，无采样 |

没有 InfoNCE、没有温度参数、没有 batch 内负样本。纯粹的 per-pixel BCE。

## Vision-Language Decoder

### 架构（3 层 Transformer decoder）

每层包含：

```
Step 1: Self-Attention on visual features
  F_v' = MHSA(LN(F_v)) + F_v                    # 8 heads

Step 2: Cross-Attention with text tokens
  vis2 = MHCA(Q=F_v'+pos_2d, K=F_t+pos_1d, V=F_t) + F_v'
  # Q: 视觉特征 + 2D 正弦位置编码
  # K/V: 词级文本特征 + 1D 位置编码
  # padding mask 屏蔽 [PAD] tokens

Step 3: Feed-Forward Network
  F_c = MLP(LN(vis2)) + vis2
  # Linear(512, 2048) → ReLU → Dropout → LN → Linear(2048, 512)
```

| 参数 | 值 |
|------|-----|
| 层数 n | 3 (最优; tested 1,2,3,4) |
| 注意力头数 | 8 |
| FFN 维度 | 2048 |
| Dropout | 0.1 |
| 特征维度 | 512 |
| 输入分辨率 | 26×26 (从 CLIP ResNet-50) |

### 输入

- **视觉：** Cross-modal Neck 输出 F_v ∈ R^{(26×26) × 512}
- **文本：** CLIP text encoder 词级特征 F_t ∈ R^{L × 512}（L=17 for RefCOCO）

## 关键消融实验

### Contrastive vs Decoder（RefCOCO+ val oIoU）

| 配置 | oIoU | 增量 |
|------|------|------|
| Baseline (无 decoder, 无 contrastive) | 50.17 | — |
| + Contrastive only | 53.15 | +2.98 |
| + Decoder only (n=1) | 54.73 | +4.56 |
| **+ Both (n=3)** | **61.39** | **+11.22** |

**超线性协同效应：** +11.22 > 2.98 + 4.56 = 7.54。两者配合时效果远超简单相加。

**解释：** Decoder 学习精细的词-像素对应关系，contrastive loss 在全局层面强化文本-前景对齐。Decoder 提供局部精度，contrastive 提供全局一致性。

### Decoder 层数

| n | oIoU |
|---|------|
| 1 | 58.08 |
| 2 | 60.25 |
| **3** | **61.39** |
| 4 | 60.91 |

n=3 最优，n=4 过拟合。

## Loss 函数

**全模型只有一个 loss：**

```
L_total = L_con = BCE(σ(z_t · z_v), mask)
```

没有额外的 Dice loss、CE loss、或辅助 loss。没有 loss 权重超参数。

## 训练细节

| 参数 | 值 |
|------|-----|
| 优化器 | Adam |
| 学习率 | 1e-4, 在 epoch 35 衰减 ×0.1 |
| Backbone LR | ×0.1（10 倍低于其他模块） |
| Epochs | 50 |
| Batch size | 64 (8/GPU × 8 V100) |
| 图像尺寸 | 416 × 416 |
| Text max length | 17 tokens (RefCOCO), 22 (G-Ref) |
| Backbone | CLIP ResNet-50 (fine-tuned at 0.1× LR) |
| CLIP encoders | **Fine-tuned**（非冻结） |
| 混合精度 | 是 (torch.cuda.amp) |
| 预测阈值 | 0.35 |

## 对 TextMamba3D 的适配要点

### 3D 适配方案

```python
# 2D → 3D 的核心改动：

# 1. 动态卷积核: 3×3 → 1×1×1 (避免参数膨胀)
#    参数量: C×9 → C×1 (降低 9 倍)
#    或 3×3×3 → C×27 (膨胀 3 倍，不推荐)
kernel = text_to_kernel(text_global)  # [B, C*1*1*1 + 1]
logits = F.conv3d(voxel_features, kernel.reshape(B,1,C,1,1,1), ...)

# 2. Upsample: bilinear → trilinear
vis_up = F.interpolate(features, scale_factor=2, mode='trilinear')

# 3. Loss 不变
loss = F.binary_cross_entropy_with_logits(logits_3d, mask_3d)
```

### 关键决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Kernel size | 1×1×1 | 3D 中 3×3×3=27 倍参数膨胀过大 |
| 文本特征 | Global ([CLS] token) | CRIS 用 global 做 contrastive，词级在 decoder 里处理 |
| Loss 权重 | 0.1 × L_t2v | 辅助 loss，不应主导训练 |
| GT mask 处理 | 二值化 (>0 为前景) | CRIS 也是二值前景/背景 |
