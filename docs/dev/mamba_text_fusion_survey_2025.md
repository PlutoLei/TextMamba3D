# Mamba + 文本引导分割：融合方法综述 (2024-2026)

> 调研日期: 2026-04-01
> 目标: 为 TextMamba3D V9.0+ 寻找比 SeqCA 更适合 Mamba 的文本-图像融合方法

---

## 1. 核心问题诊断

TextMamba3D 当前使用 Sequential Cross-Attention (SeqCA) 做文本融合，text delta 仅 +0.01% (ET) / +0.67% (Mean)。三个方向的文献调研一致指向同一个根因：

**Cross-attention 与 Mamba 特征存在结构性不匹配 (architectural mismatch)。**

| 特征属性 | Transformer | Mamba/SSM |
|----------|------------|-----------|
| Token 表示 | 位置无关（self-attention 提供全局上下文） | 位置依赖（每个 token 是扫描链上所有前驱的函数） |
| 跨模态交互 | Cross-attention 天然适配（Q/K/V 语义独立） | Cross-attention Q/K 相似度被扫描顺序伪影污染 |
| 融合时机 | 编码后融合（后处理） | 需要在 SSM 计算内部融合（参与状态转移） |

**理论支撑:**
- **The Hidden Attention of Mamba** (ACL 2025): Mamba 可被重新表述为隐式因果自注意力，产生的注意力矩阵数量是 Transformer 的三个数量级。标准 cross-attention 的显式 Q/K/V 分解与 Mamba 的隐式注意力机制不兼容。
- **Mamba-2 SSD** (Dao & Gu, ICML 2024): SSM 与注意力是同一计算的对偶表示，scalar SSM 等价于带 1-semiseparable 因果掩码的自注意力。
- **CrossMamba** (ICASSP 2025): 明确指出 "Mamba's applicability in cross-modal tasks is limited due to its inability to capture dependencies between different sequences."

---

## 2. 关键论文索引

### 2.1 Mamba 多模态融合 (SSM-native 方法)

| 论文 | 会议 | 融合策略 | 核心贡献 |
|------|------|---------|---------|
| **Coupled Mamba** | NeurIPS 2024 | 状态链耦合 | 修改状态转移方程，h_t^A = f(A·h_{t-1}^A + coupling·h_{t-1}^B + B·x_t^A)。+0.4-2.3% F1，-49% 推理时间，-83.7% 显存 |
| **COMO** | Info Fusion 2025 | Cross-Mamba 交互 | 一个模态的特征作为另一个模态 SSM 的 B/C 矩阵条件 |
| **Mixture-of-Mamba** | ICLR 2025 | 模态感知稀疏 | 不同模态使用不同的 B/C/delta 投影矩阵，共享 SSM 块 |
| **Fusion-Mamba** | IEEE TIP 2025 | 隐状态空间交互 | SSCS (通道交换) + DSSF (双向隐状态融合)。+5.9% mAP |
| **MMMamba** | AAAI 2025 | 交错扫描 | 丢弃传统融合模块，将多模态 token 交错排列后统一扫描 |
| **MoMa** | arXiv 2025 | 序列调制 | 用一个 SSM 层生成逐位置的 scale/bias 序列来调制另一个 SSM（FiLM 的 SSM 原生版） |

### 2.2 对齐优先 (Align-then-Fuse)

| 论文 | 会议 | 融合策略 | 核心贡献 |
|------|------|---------|---------|
| **AlignMamba** | CVPR 2025 | OT 对齐 + MMD | 用最优传输做 token 级局部对齐，MMD 做分布级全局对齐，对齐后再进 Mamba。+0.9% acc，-20.3% 显存 |
| **EMMA** | ICLR 2025 | 结构+层次对齐 | 像素级自回归对齐 + 多尺度特征融合 (MFF)。4x 推理加速，幻觉更少 |

### 2.3 医学分割 + Mamba + 文本

| 论文 | 会议 | 融合策略 | 核心贡献 |
|------|------|---------|---------|
| **TextBraTS** | MICCAI 2025 | 双向 cross-attention | SwinUNETR + BioBERT，双向 CA (85.3%) > 单向 T2I CA (84.8%) > Dot Sum (81.6%)。BraTS2020 同数据集 |
| **TIFC-Mamba** | MICCAI 2025 | 对比预对齐 + Mamba 融合 | 首个 Mamba + 文本医学分割，CLIP 对齐 + Bi-Dimension Fusion |
| **IM-Fuse** | MICCAI 2025 | 交错 Mamba 融合 | BraTS2023 多模态融合，交错式 Mamba 融合块 |
| **TGSAM-2** | MICCAI 2025 | 记忆编码器条件化 | 文本条件化 SAM-2 的记忆编码器（类似 SSM 隐状态条件化） |
| **TGCAM** | MICCAI 2024 | 迭代文本增强 | 解码过程中渐进细化文本特征（非静态嵌入） |

### 2.4 Mamba 视觉-语言架构

| 论文 | 会议 | 融合策略 | 核心贡献 |
|------|------|---------|---------|
| **VL-Mamba** | NeurIPS 2024 | VSS 连接器 + 拼接 | 图像 token 通过 VSS 投影后与文本 token 拼接，无 cross-attention |
| **Cobra** | arXiv 2024 | 线性投影 + 拼接 | 简单拼接即可竞争性工作，SSM 的 B/C 矩阵提供隐式跨模态选择 |
| **MFuser** | CVPR 2025 Highlight | 混合 attn-Mamba 协适配器 | Mamba 做主干，注意力仅用于文本精炼 |
| **Sigma** | WACV 2025 Oral | Siamese Mamba | 每个模态独立 Mamba 编码器 + 隐状态空间融合 |
| **OpenMamba** | Applied Sciences 2025 | Mamba-2 SSD 原生 cross-attention | 利用 Mamba-2 的 SSD 对偶性在 SSM 框架内实现 cross-attention |

### 2.5 特殊发现

| 论文 | 会议 | 发现 |
|------|------|------|
| **Hybrid Mamba (HMNet)** | NeurIPS 2024 | **Support Forgetting 问题**: Mamba 扫描时，外部注入的支持特征（文本）在隐状态中逐渐衰减。提出 Support Recapped Mamba (SRM) 定期刷新 |
| **MambaVision** | CVPR 2025 | 注意力仅在最深层有效，浅层用 Mamba 即可 |
| **Jamba** | ICLR 2025 | 1:7 注意力:Mamba 比例已足够 |
| **Mamba-3** | ICLR 2026 | MIMO 公式可让不同 "head" 处理不同模态信息 |

---

## 3. 根因分析：为什么 SeqCA 对 ET 几乎无效

综合文献，SeqCA 失败的原因不止一个：

### 3.1 Support Forgetting (HMNet, NeurIPS 2024)
文本特征通过 cross-attention 注入后，在 Mamba 的后续扫描中逐渐衰减。对于 ET（增强肿瘤），文本描述本身就稀疏（仅 3/368 报告提到 ET 特征），衰减后信号近乎为零。

### 3.2 扫描顺序伪影 (CrossMamba, ICASSP 2025)
Mamba 的 Q（如果将 cross-attention 的 image feature 视为 Q）是扫描链上的路径函数。扫描早期 token 上下文不足、晚期 token 上下文过载，导致 cross-attention 的 Q/K 匹配信号不一致。

### 3.3 特征空间不对齐 (AlignMamba, CVPR 2025)
PubMedBERT 的文本嵌入空间与 Mamba2 的图像特征空间距离过远，标准 cross-attention 的 learned projection 不足以桥接。需要显式预对齐。

### 3.4 单向融合不足 (TextBraTS, MICCAI 2025)
SeqCA 本质是单向 T2I cross-attention，TextBraTS 实验表明双向 CA (+0.5% Dice) 优于单向。

---

## 4. 推荐策略 (按优先级排序)

### Strategy A: SSM 参数调制 (最高优先级)

**灵感:** Coupled Mamba (NeurIPS 2024), COMO (Info Fusion 2025)

**原理:** 不在 Mamba 外部做 cross-attention，而是让文本直接调制 Mamba2 的输入依赖参数 (B, C, delta)：

```python
# 标准 Mamba2 selective scan
B = Linear_B(x)       # input-dependent
C = Linear_C(x)       # input-dependent

# 文本调制版
text_mod = MLP(text_global)  # [B, D]
B = Linear_B(x) + text_mod_B(text_mod).unsqueeze(1)  # text 影响输入矩阵
C = Linear_C(x) + text_mod_C(text_mod).unsqueeze(1)  # text 影响输出矩阵
```

**优点:**
- SSM 原生，文本参与状态转移而非后处理
- 避免 O(N*M) 注意力开销
- 零初始化保证训练安全（起始等效无文本）
- 新颖性高，可作为论文核心贡献

**实现复杂度:** 中等（需修改 Mamba2 块内部）

---

### Strategy B: AdaLN 条件化 (低实现成本)

**灵感:** DiT (ICCV 2023), 2024-2025 广泛采用

**原理:** 将 Mamba 块内的 LayerNorm 替换为文本条件化的 Adaptive LayerNorm：

```python
# 标准 LayerNorm
x = LayerNorm(x)

# AdaLN (文本条件化)
gamma, beta = MLP(text_global).chunk(2, dim=-1)  # from text
x = gamma * LayerNorm(x) + beta  # 文本调制归一化后的特征
```

**优点:**
- 实现最简单（替换 nn.LayerNorm）
- 文本在 SSM 计算之前影响输入分布
- 零初始化 (AdaLN-Zero) 保证安全
- 无额外注意力开销

**实现复杂度:** 低

---

### Strategy C: 交错扫描融合

**灵感:** MMMamba (AAAI 2025), IM-Fuse (MICCAI 2025)

**原理:** 将压缩后的文本 token (4-8 个) 交错插入图像 token 序列，Mamba 统一扫描：

```python
# 压缩文本为 K 个 prompt token
text_prompts = PromptProjector(text_embedding)  # [B, K, D], K=4-8

# 每隔 N/K 个图像 token 插入一个文本 token
interleaved = interleave(image_tokens, text_prompts, stride=N//K)
output = mamba_block(interleaved)
image_out = remove_text_positions(output)
```

**优点:**
- 解决 Support Forgetting（文本定期出现在扫描序列中）
- SSM 原生（统一扫描，无额外模块）
- IM-Fuse 已在 BraTS 上验证有效

**实现复杂度:** 中等

---

### Strategy D: 预对齐 + 拼接

**灵感:** AlignMamba (CVPR 2025), TIFC-Mamba (MICCAI 2025)

**原理:** 在融合前用对比学习/最优传输对齐文本和图像的特征空间：

```python
# 训练时加 alignment loss
ot_loss = optimal_transport_loss(image_features, text_features)
mmd_loss = mmd_distance(image_features, text_features)
total_loss = seg_loss + 0.1 * ot_loss + 0.1 * mmd_loss

# 对齐后简单拼接进 Mamba
aligned_seq = concat([text_tokens, image_tokens])
output = mamba_block(aligned_seq)
```

**优点:**
- 解决特征空间距离问题
- AlignMamba 报告 +0.9% 提升
- 可与 Strategy A/B 组合

**实现复杂度:** 中等

---

### Strategy E: 定期文本刷新 (Support Recapped)

**灵感:** HMNet (NeurIPS 2024)

**原理:** 在 Mamba 扫描过程中定期将文本信息注入隐状态，防止衰减：

```python
# 每 L 步刷新一次文本
for t in range(seq_len):
    h[t] = A * h[t-1] + B * x[t]
    if t % L == 0:
        h[t] = h[t] + alpha * text_state  # 定期刷新
    y[t] = C * h[t]
```

**优点:**
- 直接解决 Support Forgetting
- HMNet 已验证有效性
- 可叠加在 Strategy A 之上

**实现复杂度:** 低-中

---

## 5. 组合推荐

对 TextMamba3D V9.0，建议组合使用：

| 阶段 | 融合方式 | 原因 |
|------|---------|------|
| Stage 0 (32K tokens) | 无融合 | 序列太长，任何融合都太贵 |
| Stage 1 (4K tokens) | **AdaLN** (Strategy B) | 轻量级，替代当前 SeqCA |
| Stage 2 (512 tokens) | **SSM 参数调制** (Strategy A) | 核心融合层，文本直接参与状态转移 |
| Stage 3 (64 tokens) | **SSM 参数调制** (Strategy A) + 交错扫描 (Strategy C) | 最深层，序列短，可叠加多种策略 |
| 全局 | **预对齐 loss** (Strategy D) | 训练时辅助对齐文本-图像特征空间 |

---

## 6. 对标论文

以上策略的新颖性分析：

- **SSM 参数调制用于医学文本引导分割**: 无直接先例。Coupled Mamba 做多模态情感分析，COMO 做遥感。将此方法引入 3D 医学分割 + Mamba2 是新颖组合。
- **与 TextBraTS (MICCAI 2025) 的区别**: TextBraTS 用 SwinUNETR (Transformer)，我们用 Mamba2。TextBraTS 用标准 cross-attention，我们用 SSM-native 融合。
- **与 IM-Fuse (MICCAI 2025) 的区别**: IM-Fuse 做多 MRI 模态融合（T1/T2/FLAIR），我们做文本-图像融合。

---

## 7. 参考文献

1. Coupled Mamba (NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024/hash/6e09c213ac18d6375704a4f3ea75c4f8-Abstract-Conference.html
2. AlignMamba (CVPR 2025): https://arxiv.org/abs/2412.00833
3. EMMA (ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/hash/3760dbb5835bf0b771c3f83cb27ef2c0-Abstract-Conference.html
4. HMNet / Hybrid Mamba (NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024/hash/86fe62e3b315d2578721562d9fd1a433-Abstract-Conference.html
5. MMMamba (AAAI 2025): https://arxiv.org/abs/2512.15261
6. Fusion-Mamba (IEEE TIP 2025): https://arxiv.org/abs/2404.09146
7. COMO (Info Fusion 2025): https://arxiv.org/abs/2412.18076
8. Mixture-of-Mamba (ICLR 2025): https://arxiv.org/abs/2501.16295
9. MoMa (arXiv 2025): https://arxiv.org/pdf/2506.23283
10. TextBraTS (MICCAI 2025): https://papers.miccai.org/miccai-2025/0918-Paper2164.html
11. TIFC-Mamba (MICCAI 2025): https://papers.miccai.org/miccai-2025/paper/0648_paper.pdf
12. IM-Fuse (MICCAI 2025): https://papers.miccai.org/miccai-2025/0437-Paper0747.html
13. MFuser (CVPR 2025 Highlight): https://arxiv.org/abs/2504.03193
14. Sigma (WACV 2025 Oral): https://arxiv.org/abs/2404.04256
15. VL-Mamba (NeurIPS 2024): https://proceedings.mlr.press/v262/qiao24a.html
16. OpenMamba (Applied Sciences 2025): https://www.mdpi.com/2076-3417/15/16/9087
17. TGSAM-2 (MICCAI 2025): https://papers.miccai.org/miccai-2025/0921-Paper0846.html
18. TGCAM (MICCAI 2024): https://papers.miccai.org/miccai-2024/145-Paper2599.html
19. Hidden Attention of Mamba (ACL 2025): https://aclanthology.org/2025.acl-long.76/
20. Mamba-2 SSD (ICML 2024): https://arxiv.org/abs/2405.21060
21. MambaVision (CVPR 2025): https://arxiv.org/abs/2407.08083
22. Jamba (ICLR 2025): https://arxiv.org/abs/2403.19887
23. Mamba-3 (ICLR 2026): https://openreview.net/forum?id=HwCvaJOiCj
24. BiomedParse (Nature Methods 2024)
25. Cobra (arXiv 2024): https://arxiv.org/abs/2403.14520
26. CrossMamba (ICASSP 2025): https://arxiv.org/abs/2409.04803
27. Interactive Text-Guided Segmentation (Sci Reports 2026): https://www.nature.com/articles/s41598-026-43841-w
