# Findings

## V8.0 核心假设 (2026-04-01)

**假设：文本引导的有效性依赖于视觉特征质量。**

证据：
- V5.0 (Mamba2 从头训练 295 样本): text delta = ET +0.01% (几乎为零)
- TextBraTS (SwinUNETR ImageNet 预训练): text delta = ET +2.3%
- V8.0 Stage 1 (BraTS2021 1251 cases 预训练): epoch 77 val ET=0.8828 (远超所有版本)

预期：V8.0 Stage 2 加文本微调后，text delta 应从 +0.01% 提升到 +1-2%。
风险：Mamba 序列扫描特征可能不适合 cross-attention 融合，即使 backbone 更强。

## 实验总结

### 数据量 >> 文本引导 >> loss/架构调整
- BraTS2021 (1251 cases) 预训练 77 epoch 即超过 BraTS2020 (295 cases) 所有版本
- 文本引导贡献: Mean +0.67% (V5.0), 仅对 TC 有效 (+1.5%), 对 ET 无效
- Loss 改动 (FTL, Boundary, Hierarchy): Mean 变化 < 0.1%
- 架构改动 (Bottleneck SeqCA, EdgeEnhance): Mean 变化 < 0.1%

### Ensemble 是稳定的提升手段
- 2-model (V5.0+V6.0): +0.17% over baseline
- 3-model (V5.0+V6.0+V7.0): +0.32% over baseline
- 互补性来源: V5.0 ET 强, V6.0 TC 强, V7.0 WT 强

### BraTS SOTA 模型训练方式
- SwinUNETR/Transformer: 需要 ImageNet 预训练
- nnU-Net/MedNeXt (纯卷积): 可从头训练 (归纳偏置)
- Mamba: 类似 Transformer，需要预训练

### V5.0 仍是 BraTS2020 上的最佳单模型
- Mean Dice 0.8479 (text+TTA)
- 超过 nnU-Net baseline (0.841)
- 距 TextBraTS SOTA (0.853) 差 0.5%
