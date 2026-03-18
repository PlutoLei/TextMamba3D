# TextBraTS: Text-Guided Volumetric Brain Tumor Segmentation (MICCAI 2025)

**Paper**: TextBraTS: Text-Guided Volumetric Brain Tumor Segmentation with Innovative Dataset Development and Fusion Module Exploration
**Authors**: Xiaoyu Shi, Rahul Kumar Jain, Yinhao Li, Ruibo Hou, et al.
**Affiliations**: Ritsumeikan University (Japan), Zhengzhou University (China), Zhejiang University, Dalian University of Technology
**Venue**: MICCAI 2025
**Code**: https://github.com/Jupitern52/TextBraTS

---

## 1. Core Contribution

首个公开的 **volume-level** 文本-图像脑肿瘤分割数据集，基于 BraTS2020 构建。提出 Sequential Cross-Attention (SeqCA) 融合机制，在 BraTS2020 上实现了 **文本正向引导** 分割性能的显著提升。

---

## 2. TextBraTS Dataset

### 构建流程

1. 从 BraTS2020 的 FLAIR 模态切片转视频
2. GPT-4o 根据 template prompt 生成 pseudo-report
3. 自动质控 pipeline 检查模板一致性
4. 两位放射科医生独立审核，第三位仲裁
5. 最终产出 369 例 volume-level 文本标注

### 文本结构（4 部分模板）

| 部分 | 内容 | 作用 |
|------|------|------|
| Overall lesion | 位置 + 信号特征 | 定位肿瘤区域 |
| Edema regions | 水肿范围 + 特征 | WT 分割引导 |
| Necrotic regions | 坏死区域 + 混合信号 | TC/NCR 分割引导 |
| Ventricular compression | 中线偏移 + 脑室受压 | 全局形态学信息 |

### 数据集划分

- Train: 220, Validation: 55, Test: 94
- 每例包含 4 个影像模态 (T1, T1Gd, T2, FLAIR) + 1 个文本模态

### 与 TextMamba3D V4.3 数据的关键差异

| 维度 | TextBraTS | TextMamba3D V4.3 |
|------|-----------|------------------|
| 文本来源 | GPT-4o + 放射科医生审核 | BraTS2020 原始 FLAIR report (简短) |
| 文本质量 | 结构化模板，4 部分详细描述 | 原始临床报告，长度/质量不一 |
| 文本覆盖 | 所有子区域均有描述 | ET 仅 3/368 报告提及 "enhancing" |
| 模板化 | Yes (location + features) | No (raw FLAIR reports) |

---

## 3. 模型架构

### Backbone

- **Image Encoder**: SwinUNETR (Transformer-based, pretrained weights from NVIDIA)
- **Text Encoder**: BioBERT (frozen, pretrained on biomedical文献)
- **Feature alignment**: MLP 将 BioBERT 768-dim 映射到 image feature space

### Sequential Cross-Attention (SeqCA) — 核心创新

**两步顺序交叉注意力，在 bottleneck 层融合**：

```
Step 1 (Text-to-Image, T2I):
    Q = f_t (text features)     ← TEXT 是 Query!
    K = f_i (image features)
    V = f_i (image features)
    f_i' = Norm-Linear-Norm(Softmax(QK^T/√d) V)

    → 文本"询问"图像：我描述的位置/特征在图像的哪里？
    → 产出 refined image features f_i' (token_num × 768)

Step 2 (Image-to-Text, I2T):
    Q' = f_i (original image features)  ← IMAGE 是 Query
    K' = f_i' (refined features)
    V' = f_i' (refined features)
    f_joint = Norm-Linear-Norm(Softmax(Q'K'^T/√d) V')

    → 图像用 refined features 增强自身
    → 产出 f_joint (H/32 × W/32 × D/32 × 768)
```

### 融合位置

- **仅在 bottleneck 层**（encoder 最深层输出）
- 原因：初始层空间分辨率太高（64×），不适合与 token-level 文本特征融合

### 与 TextMamba3D V4.3 PWAM 的关键对比

| 设计选择 | TextBraTS SeqCA | TextMamba3D PWAM3D |
|----------|----------------|-------------------|
| **Step 1 Q/KV** | **Text=Q, Image=KV** | Image=Q, Text=KV |
| Step 2 | Image=Q, Refined=KV | N/A (单步) |
| 融合方式 | Additive (cross-attn output) | **Multiplicative** (element-wise ×) |
| 门控 | 无显式门控 | Tanh gate + residual |
| 融合位置 | Bottleneck only | Multi-scale (每个 decoder block) |
| Normalization | Layer between cross-attns | InstanceNorm1d on q_proj/out_proj |
| 文本编码器 | BioBERT (frozen, 110M params) | Lightweight text encoder |

---

## 4. 实验结果

### Main Results (Table 1)

| Method | ET | WT | TC | **Avg Dice** | Avg HD95 |
|--------|-----|-----|-----|------------|----------|
| SwinUNETR (no text) | 81.0 | 89.5 | 80.8 | 83.8 | 7.07 |
| **TextBraTS (SeqCA)** | **83.3** | **89.9** | **82.8** | **85.3** | **5.13** |
| **Improvement** | **+2.3** | **+0.4** | **+2.0** | **+1.5** | **-1.94** |

文本引导在 **所有子区域** 都产生了正向提升，ET 改善最大 (+2.3%)。

### Ablation: 文本格式 (Table 2)

| 输入类型 | Location | Feature | Avg Dice | Avg HD95 |
|----------|----------|---------|----------|----------|
| Raw text | - | - | 84.6 | 6.46 |
| Location only | Yes | No | 84.6 | 6.81 |
| Feature only | No | Yes | 84.6 | 5.78 |
| **Full template** | **Yes** | **Yes** | **85.3** | **5.13** |

- Location → 提升整体肿瘤定位 (Dice)
- Feature → 提升边界精度 (HD95)
- 两者结合效果最佳

### Ablation: 融合策略 (Table 3)

| Method | T2I | I2T | Avg Dice | Avg HD95 |
|--------|-----|-----|----------|----------|
| Dot Sum | - | - | 81.6 | 8.57 |
| Cross-attention (单步) | Yes | No | 84.8 | 6.26 |
| **SeqCA (双步)** | **Yes** | **Yes** | **85.3** | **5.13** |

- Dot Sum 甚至比 no-text baseline (83.8) 还差 → **简单融合会伤害性能**
- 单步 cross-attention 已有效 (+1.0% vs baseline)
- 双步 SeqCA 进一步提升 (+0.5%)

### 训练配置

| 参数 | 值 |
|------|-----|
| GPU | 2 × NVIDIA RTX A6000 |
| Batch size | 2/GPU |
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Epochs | 200 (50 warmup) |
| Input size | 128×128×128 (fixed, train=test) |
| Text token length | 128 |
| Text encoder | BioBERT (frozen) |
| Image encoder | SwinUNETR (pretrained) |

---

## 5. 对 TextMamba3D V4.4 的关键启示

### 5.1 Text=Query 是正确方向

TextBraTS 的核心洞察：**文本应该作为 Query 去"询问"图像**，而不是图像去查询文本。

- V4.3 (PWAM): Image=Q, Text=KV → 图像在文本中寻找信息 → 如果文本信息贫瘠（ET 仅 3/368 报告提及），图像只能找到噪声
- TextBraTS: Text=Q, Image=KV → 文本主动定位自己描述的区域 → 即使文本简短，也能精准指向对应区域

### 5.2 Sequential 比 Single-step 更有效

两步融合的逻辑：
1. 第一步：文本过滤图像特征，提取与文本相关的视觉信息
2. 第二步：原始图像用 refined features 增强自身

这比 V4.3 的单步 PWAM 乘法门控更能对齐两个模态。

### 5.3 文本质量决定上限

TextBraTS 的模板化文本 (location + features) 确保每个子区域都有描述。V4.3 使用原始 FLAIR report，ET 几乎无文本覆盖，这是 ET dice 下降 -0.83% 的直接原因。

**V4.4 必须做的事**：
- 为每个 BraTS case 生成结构化文本模板（可参考 TextBraTS 的 GPT-4o + 专家审核流程）
- 或直接使用 TextBraTS 数据集（已公开）

### 5.4 Bottleneck-only 融合足够

TextBraTS 仅在 bottleneck 层融合就获得了 +1.5% 提升。V4.3 在每个 decoder block 都做 PWAM 融合，可能引入了过多噪声。

### 5.5 不需要复杂辅助损失

TextBraTS **没有** T2V Loss、TextNecessityLoss、EmbeddingPerturbation 等复杂机制，仅用标准分割损失就实现了正向引导。V4.3 的辅助损失加起来仅占总 loss 的 0.3%，作用极其有限。

### 5.6 训练/测试一致性

TextBraTS 训练和测试都用 128×128×128 固定尺寸。V4.3 训练用 center-crop 128³ patches，测试用 full-volume sliding window，存在 protocol mismatch。

---

## 6. V4.4 具体改进建议（基于 TextBraTS 启发）

| 优先级 | 改进项 | 来源 | 预期效果 |
|--------|--------|------|----------|
| P0 | 反转 Q/KV：Text=Q, Image=KV | SeqCA Step 1 | 修复 PWAM 方向错误 |
| P0 | 使用 TextBraTS 数据集或生成结构化文本 | Dataset Creation | 解决 ET 文本覆盖问题 |
| P1 | 改 SeqCA 双步融合替代 PWAM 单步 | SeqCA architecture | 更好的模态对齐 |
| P1 | Bottleneck-only 融合（去除 multi-scale PWAM） | Fusion position | 减少低层噪声注入 |
| P2 | 移除 T2V/L_nec/L_contrastive | Simplicity principle | 减少训练复杂度 |
| P2 | 统一 train/test 为 128³ fixed size | Protocol consistency | 消除 val/test gap |
| P3 | 换用 BioBERT 替代 lightweight encoder | Text encoder | 更强文本表征能力 |

---

## 7. 统计显著性

- TextBraTS 报告了 10 次独立训练的 t-test，p=0.0077 (<0.05)，确认文本引导提升显著
- V4.3 仅单次训练，无法判断 -0.26% 是否显著（可能在噪声范围内）

---

## 8. 局限性

- 数据集仅 369 例（小规模）
- 文本生成依赖 GPT-4o（可能引入 hallucination）
- 仅在 BraTS2020 上验证，未测试其他数据集
- 未探索 Mamba/SSM 架构
- 未与 LViT、CRIS 等其他融合方法直接对比
