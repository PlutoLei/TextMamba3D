# SwinBTS: A Method for 3D Multimodal Brain Tumor Segmentation Using Swin Transformer

> Brain Sciences | Jiang et al.
> KG quality score: 1.00 (citation_proxy)

## 一句话总结

Swin Transformer + CNN 混合架构用于 3D 多模态脑肿瘤分割，结合全局 self-attention 和局部卷积特征。

## 核心贡献

1. **Hybrid Swin Transformer-CNN**：Swin Transformer 编码器 + CNN 解码器
2. **Improved Transformer Module**：针对 3D 医学影像优化的 Swin blocks
3. **BraTS 基准评估**：多个 BraTS 数据集上的竞争性结果

## 核心机制

```
4-modality MRI (T1, T1ce, T2, FLAIR)
    ↓ 3D Patch Embedding
    ↓ Swin Transformer Encoder (shifted window attention)
    ↓ CNN Decoder (skip connections)
    ↓ 3-class segmentation (ET, TC, WT)
```

## 与 TextMamba3D 的关系

- **同数据集竞品**：BraTS 脑肿瘤分割
- **Backbone 差异**：Swin Transformer vs Mamba SSM
- **TextBraTS 的前身/同族**：TextBraTS (MICCAI 2025) 使用 Swin UNETR，与 SwinBTS 同族
- **对比价值**：展示纯视觉 Transformer 在 BraTS 上的性能，作为 text-guided 方法的 baseline 参考

## 待深读要点

- [ ] BraTS 2020/2021 上的具体 Dice 数值
- [ ] 与 UNETR、Swin UNETR 的对比
- [ ] 模型参数量和训练细节
