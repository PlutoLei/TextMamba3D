# TextMamba3D 过拟合分析与改进计划

> 日期: 2026-03-04
> 目标: 将 Val Dice 从 0.63 提升至 0.80+，对标 nnU-Net (0.841) 和 TextBraTS 原论文 (0.853)

---

## 一、核心问题诊断

### 1.1 过拟合风险（严重）

| 对比维度 | TextMamba3D | SegMamba | TextBraTS 原论文 |
|----------|-------------|---------|-----------------|
| 总参数 | ~180M | ~23M | 未公开 |
| 可训练参数 | ~70M | ~23M | 未公开 |
| 训练样本 | 295 例 | ~877 例 | 220 例 |
| 参数/样本比 | **237K:1** | 26K:1 | — |
| 数据集 | TextBraTS 369例 | BraTS2023 1251例 | TextBraTS 369例 |
| 数据划分 | 295/74 (train/val) | 877/125/250 (t/v/t) | 220/55/94 (t/v/t) |

TextMamba3D 用 3 倍的可训练参数去拟合 1/3 的数据（相比 SegMamba），参数/样本比是 SegMamba 的 ~9 倍。

### 1.2 实验设计缺陷

**val == test（致命问题）**：`_split_cases()` 中 `split='val'` 和 `split='test'` 返回同一组 74 例数据。best checkpoint 选择依据就是这 74 例，所有报告指标都是选择偏倚后的乐观估计。

```python
# 当前代码（有缺陷）
if split == 'train':
    selected_indices = indices[:train_size]
else:  # val or test — 返回相同数据！
    selected_indices = indices[train_size:]
```

其他问题：
- 无交叉验证：单一 seed=42 的 80/20 划分，结果对划分敏感
- TextBraTS 原论文使用 220/55/94 三划分，有独立 94 例测试集

### 1.3 正则化不足

| 正则化手段 | 当前值 | 问题 |
|-----------|--------|------|
| Dropout | **0.0**（全模型） | 70M 参数无随机正则化 |
| Weight Decay | 1e-5 | 过小，几乎无效 |
| PubMedBERT | 解冻最后 2 层 (+14M) | 14M 文本参数在 295 例上难以有效学习 |
| no-text ratio | 0.15 | 偏低 |

仅有的正则化：数据增强（7 种）、deep supervision、梯度裁剪。

### 1.4 硬件限制 (RTX 4060 Laptop 8GB)

| 约束 | 当前值 | 理想值 | 影响 |
|------|--------|--------|------|
| Batch size | 1 | 4~8 | 梯度噪声大，训练不稳定 |
| Patch size | 64³ | 96³~128³ | 64³ 仅覆盖原图 (240×240×155) 的 ~3%，丢失大量上下文 |
| 对比损失 | 无效（需 batch>1） | batch=4+ | 设计了但实际没用上 |
| 梯度累积 | 4（模拟 batch=4） | — | 只解决梯度均值，不解决对比学习 |
| 训练速度 | 极慢 | — | 超参搜索、消融实验不可行 |

**patch_size=64³ 是最致命的硬件约束**。脑肿瘤分割需要全局上下文（肿瘤位置、与脑室关系、水肿范围），64³ 视野太小。这部分解释了 Dice 停在 0.63 的原因——不完全是过拟合，也是欠拟合（信息不足）。

---

## 二、当前性能 vs 基准

### 2.1 同数据集 (TextBraTS 369 例) 性能对比

| 方法 | ET Dice | WT Dice | TC Dice | Avg Dice | 来源 |
|------|---------|---------|---------|----------|------|
| 3D-UNet | 80.4% | 87.3% | 81.6% | 83.1% | TextBraTS 原论文 |
| nnU-Net | 82.2% | 87.5% | 82.6% | **84.1%** | TextBraTS 原论文 |
| SegResNet | 80.9% | 88.4% | 82.3% | 83.8% | TextBraTS 原论文 |
| SwinUNETR | 81.0% | 89.5% | 80.8% | 83.8% | TextBraTS 原论文 |
| **TextBraTS (原论文)** | **83.3%** | **89.9%** | **82.8%** | **85.3%** | TextBraTS 原论文 |
| TextMamba3D (epoch 11/150) | — | — | — | 62.3% | 当前 |
| TextMamba3D (历史最佳 epoch 62) | — | — | — | 63.7% | 历史 |

注意：TextBraTS 原论文用 220 例训练，TextMamba3D 用 295 例训练——更多数据反而更差，说明过拟合+硬件限制正在起作用。

### 2.2 文本引导增益对比

| 方法 | 文本增益 | 可靠性 |
|------|---------|--------|
| TextMamba3D | +2.7% (0.5959 → 0.6233) | 不可靠：未收敛、无统计检验、可能是参数容量贡献 |
| TextBraTS 原论文 | +1.2% (84.1% → 85.3%) | 可靠：收敛后、独立测试集、有显著性检验 (p=0.0077) |

### 2.3 Dice 预期

| 场景 | 预期 Avg Dice | 依据 |
|------|--------------|------|
| 当前配置不改 | 0.65~0.70 | 历史最佳 0.6366，且 val=test 有偏倚 |
| 修复正则化 (Phase 2) | 0.75~0.80 | dropout + weight decay + 冻结 BERT |
| 修复正则化 + 模型瘦身 + 更大 GPU | 0.80~0.85 | 接近 nnU-Net，合理上限 |
| 理论上限（此数据集） | ~0.85 | TextBraTS 原论文 SOTA |

---

## 三、改进计划（三阶段）

### Phase 1: 修复实验设计（不改模型代码）

**目标**：让评估体系可信，否则后续所有改动都无法判断效果。

- [ ] **分离 val/test**：修改 `_split_cases()`，参考原论文 220/55/94 划分
  - 或实现 5-fold CV（369 例全利用，每折 ~74 例测试，报告均值±标准差）
- [ ] **添加 train/val loss 曲线**：在 TensorBoard 中记录，监测过拟合拐点
- [ ] **用当前模型跑一次完整 baseline**：作为后续改动的对照

关键代码修改位置：`data/brats_textbrats_dataset.py` → `_split_cases()`

### Phase 2: 正则化增强（改配置，不改架构）

**目标**：减少可训练参数，增强正则化，预期 Dice 0.75~0.80。

| 改动 | 修改位置 | 当前值 → 目标值 |
|------|---------|----------------|
| Dropout | 所有 Mamba blocks | 0.0 → 0.1~0.2 |
| Weight Decay | `textbrats.yaml` | 1e-5 → 1e-2~5e-3 |
| 冻结 PubMedBERT | `textbrats.yaml` | unfreeze_text_layers: 2 → 0 |
| no-text ratio | `textbrats.yaml` | 0.15 → 0.25~0.30 |

可训练参数：~70M → ~56M（冻结 BERT 后）

### Phase 3: 模型瘦身（仅在 Phase 2 仍过拟合时）

**目标**：将模型容量匹配数据规模，预期 Dice 0.80~0.85。

| 改动 | 效果 | 风险 |
|------|------|------|
| base dim: 96 → 48 | 图像编码器 ~44M → ~11M | 可能影响表达能力 |
| encoder depths: [2,2,2,2] → [1,1,2,1] | 参数约减半 | 影响较小 |
| 去掉 Pixel-Text Cross-Attention | 省 ~3-4M | 需消融验证 |
| 扫描方向: 6 → 3 | 参数和计算各减半 | 与 SegMamba 对齐 |

可训练参数：~56M → ~20M

**实施顺序**：Phase 1 → Phase 2 → 评估 → 如需再做 Phase 3。不要在评估体系有缺陷时调模型。

---

## 四、GPU 显存需求与选型

### 4.1 显存构成

以 AMP (fp16) + 梯度检查点为前提：

| 显存组成 | Phase 2 (dim=96) | Phase 2+3 (dim=48) |
|----------|-----------------|---------------------|
| 模型权重 (fp32 + fp16) | ~1.1 GB | ~0.8 GB |
| 优化器状态 (AdamW) | ~0.45 GB | ~0.16 GB |
| 梯度 | ~0.22 GB | ~0.08 GB |
| **固定开销** | **~1.8 GB** | **~1.0 GB** |
| 激活值 | 取决于 patch 和 batch | 取决于 patch 和 batch |

### 4.2 不同配置显存估算

#### Phase 2（冻结 BERT，dim=96，~56M 可训练）

| Patch | Batch | 预估显存 | 最低 GPU |
|-------|-------|---------|---------|
| 64³ | 1 | ~7 GB | 8 GB (4060) |
| 96³ | 1 | ~14 GB | 16 GB (T4) |
| 96³ | 2 | ~24 GB | 24 GB (3090) |
| 128³ | 2 | ~38 GB | 40 GB (A100) |

#### Phase 2+3（dim=48，~20M 可训练）

| Patch | Batch | 预估显存 | 最低 GPU |
|-------|-------|---------|---------|
| 64³ | 2 | ~6 GB | 8 GB (4060) |
| 96³ | 2 | ~12 GB | 16 GB (T4) |
| 96³ | 4 | ~21 GB | 24 GB (3090) |
| 128³ | 4 | ~35 GB | 40 GB (A100) |

### 4.3 重要：显存安全余量

GPU 显存满载会导致 OOM 崩溃、cuDNN 调优失效、碎片化等问题。经验法则：

```
安全使用量 = 标称显存 × 80~85%

16GB GPU → 实际可规划 ~13GB
24GB GPU → 实际可规划 ~20GB
40GB GPU → 实际可规划 ~34GB
```

**估算 20GB 就选 24GB 的卡，不要卡着极限跑。**

### 4.4 推荐方案

| 方案 | GPU | 配置 | 成本 | 推荐度 |
|------|-----|------|------|--------|
| **性价比最优** | T4 16GB | Phase 2+3, patch 96³, batch 2 | Colab Free | ★★★★★ |
| 平衡方案 | RTX 3090 24GB | Phase 2, patch 96³, batch 2 | AutoDL ~3元/时 | ★★★★ |
| 最佳体验 | A100 40GB | Phase 2, patch 128³, batch 2 | Colab Pro ~$12/月 | ★★★ |

### 4.5 训练环境分工

| 环境 | 用途 |
|------|------|
| 本地 4060 8GB | 代码开发、调试、冒烟测试 (`--max-samples 50`) |
| 远程 GPU (16GB+) | 正式训练、消融实验、5-fold CV |

---

## 五、文本引导价值评估

### 当前证据

当前 +2.7% 增益**不可靠**，原因：
1. 模型未收敛（Dice 0.62 远低于 baseline 0.84）
2. 无统计检验（单次划分、单次运行）
3. 无法排除参数容量的混淆（文本分支带来 +21M 可训练参数）

### 验证文本引导价值所需的实验

1. **公平消融**：去掉文本分支，将省下的参数补到图像编码器，确保总可训练参数接近
2. **模型先收敛**：Dice > 0.80 后再测量文本增益
3. **多次运行**：至少 3 次不同 seed，报告均值±标准差
4. **统计检验**：paired t-test 或 Wilcoxon signed-rank test

### 预期结论

TextBraTS 原论文在同数据集上验证了文本引导有效（+1.2%，p=0.0077），但增益上限不高。TextMamba3D 的复杂融合架构（FiLM + Cross-Attn + MambaFusion）是否优于原论文的简单 sequential cross-attention，需要收敛后的公平对比才能回答。

---

## 六、关键文件路径

| 文件 | 需修改内容 |
|------|-----------|
| `data/brats_textbrats_dataset.py` | `_split_cases()` — 分离 val/test，或实现 5-fold CV |
| `models/mamba_block.py` | 添加 dropout 参数到 Mamba blocks |
| `models/encoder_3d.py` | 支持可配置 dim 和 depths |
| `models/text_encoder.py` | 支持完全冻结 |
| `configs/textbrats.yaml` | 更新 dropout、weight_decay、unfreeze_text_layers 等 |
| `train.py` | 添加 train loss 曲线记录、支持 k-fold CV |

---

## 七、参考资料

- TextBraTS 原论文 (MICCAI 2025): https://arxiv.org/html/2506.16784v2
- TextBraTS GitHub: https://github.com/Jupitern52/TextBraTS
- SegMamba 论文: https://arxiv.org/abs/2401.13560
- SegMamba GitHub: https://github.com/ge-xing/SegMamba

---

## 八、行动清单（快速参考）

```
1. [本地] 修改 _split_cases() → 三划分或 5-fold CV
2. [本地] 添加 dropout=0.1 到 Mamba blocks
3. [本地] 修改 yaml: weight_decay=1e-2, unfreeze_text_layers=0, no_text_ratio=0.25
4. [本地] 冒烟测试 --max-samples 50 确认代码正确
5. [远程 GPU] 正式训练 Phase 2，patch 96³, batch 2+
6. [远程 GPU] 评估是否需要 Phase 3 (dim 48)
7. [远程 GPU] 收敛后做文本引导消融实验
```
