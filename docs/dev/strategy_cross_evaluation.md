# 融合策略交叉评测报告

> 评测框架: Scientific Critical Thinking + Scientific Brainstorming
> 日期: 2026-04-01
> 评测对象: 4 个候选策略 (A/B/C/D) + 已建立的 7 条红线

---

## 评测方法论

每个策略从 5 个维度打分 (1-5):

| 维度 | 含义 |
|------|------|
| **可行性** | 在当前代码架构下的工程实现难度 |
| **安全性** | 与 7 条红线的冲突程度，以及训练崩溃风险 |
| **新颖性** | 是否与已失败方法存在根本区别 |
| **证据强度** | 文献支撑力度 (会议级别 × 任务相似度) |
| **预期收益** | 预估 text delta 改善幅度 |

---

## Strategy A: SSM 参数调制 (文本调制 B/C 矩阵)

### 关键发现: 实现路径受阻

**Mamba2 内部结构分析:**

```
Mamba2.forward(x):
    x_proj = self.in_proj(x)           # [B, L, 2*d_inner + 2*d_state + nheads]
    x, z, B, C, dt = x_proj.split(...)  # B/C 从 in_proj 分裂出来
    y = ssd(x, dt, A, B, C, ...)        # Triton kernel, 不可插入
```

B 和 C 是 `in_proj` (单个 fused linear) 的输出切片，由 Triton 编译的 `ssd` 内核消费。**无法从外部注入文本调制到 B/C**，除非:

| 实现路径 | 可行性 | 风险 |
|---------|--------|------|
| (1) Fork `mamba_ssm`，修改 `ssd` kernel | 极高难度，需修改 Triton CUDA | 破坏上游兼容性 |
| (2) 修改 `in_proj` 的权重/偏置 | 中等，但 `in_proj` 是 fused proj | B/C 与 x/z/dt 共享同一 Linear，偏置修改影响所有分量 |
| (3) 调制 `in_proj` 的输入 x | **可行** | 但这本质上等于 AdaLN/FiLM |

**批判性结论:** 路径 (3) 是唯一实际可行的方案。但 `gamma * LN(x) + beta → in_proj → B, C` 意味着文本通过 x 间接影响 B/C。这与 Coupled Mamba 论文中直接修改状态转移方程 `h_t = f(A·h_{t-1} + coupling·h_{t-1}^text)` 是**完全不同的机制**。Coupled Mamba 修改的是状态传播（A 矩阵侧），而路径 (3) 只影响输入投影，是一个弱得多的干预。

**与 FiLM 的真实区别:**

| 属性 | FiLM (V2, 失败) | Strategy A 路径(3) |
|------|----------------|-------------------|
| 调制位置 | Skip connection (编码后) | Mamba block 内部 (编码中) |
| 调制对象 | 原始特征 x | LayerNorm(x) → SSM 输入 |
| 间接效果 | 无 (不影响 SSM 计算) | 通过 in_proj 间接影响 B/C/dt |
| 区别程度 | — | **中等**，位置不同但机制相似 |

### 评分

| 维度 | 分数 | 理由 |
|------|------|------|
| 可行性 | 2/5 | 真正的 B/C 调制需 fork mamba_ssm；退化版本等同 AdaLN |
| 安全性 | 3/5 | 零初始化可保安全，但修改 SSM 内部有训练不稳定风险 |
| 新颖性 | 2/5 | 可行版本 (路径3) 与 AdaLN/FiLM 区别不够大 |
| 证据强度 | 4/5 | Coupled Mamba (NeurIPS 2024)、COMO (Info Fusion 2025) |
| 预期收益 | 2/5 | 如退化为路径(3)，与 AdaLN 收益相同 |

**修正判定: 原报告高估了此策略。真正的 SSM 参数调制在 Mamba2 闭源 Triton kernel 下不可行，退化版本与 Strategy B (AdaLN) 合并。**

---

## Strategy B: AdaLN 条件化

### 与 FiLM 的严格对比

**代码层面的差异:**

```python
# FiLM (V2, 失败) — 应用在 skip connection 上
# fusion.py:32-43
gamma = 2.0 * sigmoid(Linear(text_cond))  # (0, 2)
beta = Linear(text_cond)
return gamma * x + beta  # x 是 encoder 输出的 skip feature

# AdaLN (提议) — 应用在 MambaBlock 内部
# 替换 mamba_block.py:155 的 self.norm(x)
gamma, beta = MLP(text_global).chunk(2, dim=-1)
x = gamma * LayerNorm(x) + beta  # x 是 SSM 的输入
x = self.mamba(x)  # 文本间接影响 SSM 计算
```

| 失败因素 | FiLM 中是否存在 | AdaLN 中是否存在 | 分析 |
|---------|---------------|-----------------|------|
| 浅层污染 (Red Line #2) | 是 (所有 4 个 stage) | **取决于配置** — 如限 Stage 2-3 可避免 | 可控 |
| 全局通道调制对高维 BERT 无效 | 是 (256-dim 直接投影) | **仍存在** — CLS token 仍是高维 | 核心风险 |
| 过参数化 (65000:1) | 是 (所有 4 stage × 2 proj) | **显著降低** — 仅 Stage 2-3, 且 zero-init | 改善 |
| 与 SSM 无交互 | 是 (post-encoding) | **有间接交互** — 调制 SSM 输入 | 改善 |
| 调制范围不可控 | 部分 (gamma 限 0-2) | **zero-init 更安全** — gamma=0 起始 | 改善 |

**批判性结论:** AdaLN 比 FiLM 有 3 处结构性改善 (位置、参数量、初始化)，但共享同一核心限制——**全局通道调制对稀疏文本 (ET 仅 3/368 提及) 可能仍然无效**。对 TC (+1.05%) 和 WT (+0.74%) 可能有额外提升，但对 ET 改善概率低。

**潜在价值:** 如果 V8.0 预训练显著提升了 backbone 特征质量，AdaLN 的条件信号可能变得更有效。这是一个需要在 V8.0 完成后验证的条件性假设。

### 评分

| 维度 | 分数 | 理由 |
|------|------|------|
| 可行性 | **5/5** | 替换 `nn.LayerNorm` → `AdaLN`，~30 行代码 |
| 安全性 | 4/5 | AdaLN-Zero 初始化，限 Stage 2-3，不违反红线 |
| 新颖性 | 2/5 | 与 FiLM 共享全局通道调制核心机制 |
| 证据强度 | 3/5 | DiT (ICCV 2023) 在图像生成中成功，但无医学分割证据 |
| 预期收益 | 2/5 | TC/WT 可能 +0.3-0.5%，ET 大概率无效 |

---

## Strategy C: 交错扫描融合 + 定期文本刷新

### 实现可行性的关键障碍

**CrossScanBiMamba3DBlock 的空间约束:**

```python
# mamba_block.py:405-439
def forward(self, x):  # x: [B, D*H*W, C]
    # DWConv3D: 假设 x 可以 reshape 为 (B, D, H, W, C)
    x_3d = x.reshape(B, D, H, W, C)
    x = x + self.dwconv(x_3d)  # ← 插入文本 token 后这里会崩

    # 空间重排: 假设 L = D*H*W
    x_hwd = self._reorder(x, 'd h w', 'h w d')  # ← 预计算的 index 不含文本位置
```

**问题:** DWConv3D 和 `_reorder` 都假设序列长度 = D×H×W。插入文本 token 后:
- DWConv3D 的 reshape 维度不匹配
- 预计算的重排 index 不含文本位置
- uncertainty_head 的门控也作用于文本 token

**可行的变通方案:**

| 方案 | 描述 | 可行性 | 代价 |
|------|------|--------|------|
| (a) CrossScan 后独立 Mamba 层 | 先 CrossScan 编码，再用独立 Mamba 处理 [text, image] | 可行 | 等同于 MambaFusion (V2 失败) |
| (b) 重写 CrossScanBiMamba3DBlock | 让 DWConv/reorder 感知文本 token 位置 | 大重构 | 工程量巨大，引入新 bug |
| (c) 跳过 CrossScan，在 PatchMerging 前/后交错 | 在 encoder stage 之间而非之内插入 | 可行 | 文本不参与 SSM 扫描，效果弱 |
| **(d) 仅在 Stage 3 (64 tokens) 实现** | Stage 3 无后续 downsample，可单独处理 | **最可行** | 仅限最深层 |

**方案 (d) 详细分析:**
- Stage 3: 64 image tokens + 4-8 text prompt tokens = 68-72 tokens
- 无 DWConv 问题 (可跳过或用 1D conv 替代)
- 无 reorder 问题 (64 tokens 已是最小尺度)
- 但: 仅在最深层交错 ≈ 改良版 MambaFusion (限于 bottleneck)
- **与 Red Line #4 矛盾**: "单 bottleneck 太窄"

**Support Forgetting 是否真实存在?**

HMNet (NeurIPS 2024) 发现此问题，但其任务是 few-shot segmentation (support set 衰减)。TextMamba3D 的情况不同:
- SeqCA 是在 skip connection 上融合，不在 SSM 扫描内部
- 文本特征通过 cross-attention 注入到图像特征后，图像特征再进入 decoder
- **文本信号根本不经过 Mamba 扫描**——它在编码完成后注入

**批判性结论:** Support Forgetting 在当前架构下**不是问题**，因为文本融合发生在编码后的 skip connection 上。如果改为在 SSM 内部融合 (Strategy A/C)，才会触发此问题。这是一个有条件的问题，不是当前架构的瓶颈。

### 评分

| 维度 | 分数 | 理由 |
|------|------|------|
| 可行性 | 2/5 | DWConv3D + spatial reorder 假定固定空间网格，全面实现需大重构 |
| 安全性 | 2/5 | 触碰 encoder 核心模块；仅 Stage 3 可行但违反 Red Line #4 |
| 新颖性 | 4/5 | 与 MambaFusion (V2) 有本质区别 (交错 vs 前置) |
| 证据强度 | 4/5 | MMMamba (AAAI 2025), IM-Fuse (MICCAI 2025, BraTS 上验证) |
| 预期收益 | 3/5 | 如能全面实现预期 +1-2%，但工程可行性限制了实际效果 |

---

## Strategy D: 预对齐 Loss

### 实现分析

**已有基础设施 (textmamba3d.py:169-173):**

```python
if has_text:
    pixel_feat = decoder_features[-1]
    img_global = self.img_proj(pixel_feat.mean(dim=1))  # [B, 256]
    text_global = self.text_encoder.get_global_feature(text_features)  # [B, 256]
    return seg_output, img_global, text_global, pixel_feat
```

`img_global` 和 `text_global` 已经投影到同一维度 (256)。添加对齐 loss 仅需在 train.py 中加 ~10 行代码。

**对齐方法选择:**

| 方法 | 计算量 | 实现难度 | 预期效果 |
|------|--------|---------|---------|
| **Cosine similarity loss** | O(B×D) | 极低 (3行) | 弱，仅全局对齐 |
| **NT-Xent (InfoNCE)** | O(B²×D) | 低 (20行) | 中，batch 内对比 |
| **Optimal Transport (Sinkhorn)** | O(B×N×M×iter) | 中 (50行+依赖) | 强，token 级对齐 |
| **MMD** | O(B²×D) | 低 (15行) | 中，分布级对齐 |

**批判性评估:**

1. **正面:** 这是唯一不修改推理架构的策略。最坏情况是浪费训练时间但不影响推理。
2. **质疑:** 如果文本和图像的特征空间已经通过 SeqCA 的 text_proj (Linear + LayerNorm) 隐式对齐，额外的显式对齐 loss 可能是冗余的。
3. **关键假设检验:** AlignMamba (CVPR 2025) 报告 +0.9% 是在情感分析任务上，数据量大且文本密度高。BraTS 仅 295 样本 + 稀疏文本，对齐效果可能大幅缩水。
4. **潜在负面效果:** 对齐 loss 可能导致图像特征向文本特征空间偏移，损害不使用文本时的分割性能 (no-text baseline 下降)。

### 评分

| 维度 | 分数 | 理由 |
|------|------|------|
| 可行性 | **5/5** | 基础设施已就绪，10-20 行代码 |
| 安全性 | **5/5** | 仅影响训练，不改推理架构，loss 权重可随时降为 0 |
| 新颖性 | 3/5 | 对比学习在医学分割中已有大量先例 |
| 证据强度 | 2/5 | 相关文献 (AlignMamba) 任务差异大，BraTS 上无直接验证 |
| 预期收益 | 2/5 | 单独使用可能仅 +0.1-0.3%，需与其他策略组合 |

---

## 综合评分矩阵

| 策略 | 可行性 | 安全性 | 新颖性 | 证据 | 收益 | **总分** | **修正判定** |
|------|--------|--------|--------|------|------|---------|-------------|
| A. SSM 参数调制 | 2 | 3 | 2 | 4 | 2 | **13/25** | 不可行 (Triton 闭源)，退化为 B |
| B. AdaLN | 5 | 4 | 2 | 3 | 2 | **16/25** | 低成本试探，但可能重蹈 FiLM |
| C. 交错扫描 | 2 | 2 | 4 | 4 | 3 | **15/25** | 工程可行性严重受限 |
| D. 预对齐 loss | 5 | 5 | 3 | 2 | 2 | **17/25** | 最安全的增量改善 |

---

## 盲点分析 (Brainstorming 框架)

### 被忽略的关键问题

**1. 当前融合位置可能就是最佳位置**

所有 4 个策略都试图将文本更深地注入编码器。但 SeqCA 在 skip connection 上已经是 V4.4 的成功要素。问题可能不在于**文本在哪里融合**，而在于**文本本身的质量**。

证据:
- ET 失败的直接原因是仅 3/368 报告提到 ET 特征
- TC +1.05% 因为几乎所有报告都描述 tumor core
- 换融合位置不会改变文本内容的稀疏性

**2. 文本增强 > 融合方法改变**

ET_improvement_methods_survey.md 中的 "ET-Enriched Text generation" (N23 in 原文档, P3 级别) 可能被低估了。如果用 LLM 为每个 case 生成 ET 专用描述 ("The enhancing rim shows irregular margins at the left temporal lobe, measuring approximately 2cm..."), 即使用现有 SeqCA 也可能显著提升 ET delta。

**3. V8.0 预训练可能已解决核心问题**

V8.0 的核心假设: 文本 delta 小是因为 backbone 太弱。如果 Stage 1 预训练 (epoch 77 时 val ET=0.8828) 能大幅提升基线:
- 更强的特征 → SeqCA 的 Q/K 匹配更准确 → text delta 自然提升
- 此时换融合方法可能是过度工程化

**4. 应先验证 V8.0 text delta 再决策**

在 V8.0 Stage 2 完成前，任何融合方法的改变都缺乏决策依据。如果 V8.0 的 text delta 从 +0.55% 跳到 +2%，说明预训练解决了根因，不需要换融合方法。

### 逆向思维: 什么条件下不应该换融合方法?

- 如果 V8.0 text delta > +1.5%: 预训练已解决，SeqCA 足够
- 如果 V8.0 no-text baseline Mean > 0.88: backbone 已足够强，瓶颈转移到文本质量
- 如果 V8.0 ET text delta > +0.5%: ET 的文本信号在强 backbone 下被激活

---

## 最终推荐 (修正版)

### 短期: 等待 V8.0 结果

**不要在 V8.0 Stage 2 完成前实现任何新融合方法。** V8.0 的 text delta 是判断下一步的关键证据。

### V8.0 text delta 决策树

```
V8.0 text delta:
├── > +1.5% Mean → 预训练解决了问题
│   └── 行动: 论文 story = "预训练释放文本引导"，不换融合方法
│
├── +0.5% ~ +1.5% Mean, ET > +0.3% → 有改善但未达预期
│   └── 行动: 
│       (1) 先加 预对齐 loss (Strategy D, 零风险)
│       (2) 再试 AdaLN Stage 2-3 (Strategy B, 低风险)
│       (3) 配合文本增强 (ET-enriched text generation)
│
├── < +0.5% Mean, ET 仍负 → 预训练未解决融合问题
│   └── 行动:
│       (1) 文本增强是必要条件 (不改文本，换融合方法也无用)
│       (2) 考虑 Strategy C 在 Stage 3 的简化版
│       (3) 考虑放弃文本融合，转向 ensemble 路线
│
└── V8.0 no-text baseline Mean > 0.87
    └── 行动: 直接对标 SOTA，论文 story 不依赖文本
```

### 如果决定实施新融合方法

优先级:
1. **Strategy D (预对齐 loss)** — 零风险，10 行代码，先试
2. **Strategy B (AdaLN, 仅 Stage 2-3)** — 低风险，~30 行代码
3. **文本增强 (不在原 4 策略中)** — 可能是真正的关键
4. **Strategy C (交错扫描, 仅 Stage 3 简化版)** — 需大量工程，最后考虑
5. ~~Strategy A (SSM 参数调制)~~ — 在 Mamba2 Triton 内核下不可行，降级为 B

---

## 方法论声明

本评测可能存在的偏差:
1. **保守偏差:** 因 5 次失败历史，评测可能过度规避风险
2. **局部知识限制:** Mamba2 内核分析基于公开文档推断，Colab 环境可能有不同版本
3. **任务迁移不确定性:** 所有引用论文的效果在 BraTS 数据集上未直接验证
