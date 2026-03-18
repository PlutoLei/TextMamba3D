# CKD-TransBTS: Clinical Knowledge-Driven Hybrid Transformer with Modality-Correlated Cross-Attention for Brain Tumor Segmentation

> IEEE Transactions on Medical Imaging (TMI) | Lin et al.
> KG quality score: 0.92

## 一句话总结

利用临床先验知识将多模态 MRI 分组，结合 Transformer-CNN 混合架构和模态相关 cross-attention 实现脑肿瘤分割。

## 核心贡献

1. **Clinical Knowledge-Driven Modality Grouping**：基于影像学原理将 4 种 MRI 模态分为互补组（T1+T1ce 结构组 / T2+FLAIR 水肿组）
2. **Modality-Correlated Cross-Attention**：组间 cross-attention 捕获互补信息
3. **Hybrid CNN-Transformer**：CNN 提取局部特征，Transformer 建模全局依赖

## 核心机制

```
MRI 输入: T1, T1ce, T2, FLAIR (4 模态)
    ↓ 临床知识分组
组 A: T1 + T1ce → 增强肿瘤/坏死边界
组 B: T2 + FLAIR → 水肿/浸润范围
    ↓ 各组独立 CNN 编码
    ↓ 组间 Cross-Attention 融合
    ↓ Transformer 全局建模
    ↓ 解码 + 分割
```

## 关键实验结果 (BraTS)

- 在 BraTS 2019/2020 上达到竞争性结果
- 模态分组 vs 直接拼接：分组策略更有效

## 与 TextMamba3D 的关系

- **同数据集竞品**：都在 BraTS 上做脑肿瘤分割
- **区别**：CKD-TransBTS 用模态间 cross-attention，TextMamba3D 用文本-图像 cross-attention
- **启发**：临床知识驱动的模态分组思路值得借鉴——TextMamba3D 的文本本质上也是一种"临床知识注入"
- **论文引用**：Related Work 中 BraTS 方法综述必引

## 待深读要点

- [ ] 具体的 Dice 数值（ET/TC/WT 各多少）
- [ ] Cross-attention 的实现细节
- [ ] 与 TextBraTS 的对比
