# VSSD-UNet: Vision State Space Duality for Medical Image Segmentation

> ICLR 2025 |
> 副标题：Enhancing Precision through Non-Causal Modeling

## 一句话总结

将 Vision State Space Duality (VSSD) 集成到 UNet 架构中，通过非因果 SSM 建模和混合 VSSD-Attention 解码器实现线性复杂度的高精度医学分割。

## 核心贡献

1. **Non-Causal SSM Adaptation**：解决标准 Mamba/SSM 因果扫描在图像中方向偏差的问题
2. **Hybrid VSSD-Attention Decoder**：SSM 负责全局建模，Attention 负责精细恢复
3. **线性空间复杂度**：相比标准 Transformer 的 O(n²)，SSM 实现 O(n)

## 核心机制

- VSSD blocks 替代 Transformer encoder 中的 self-attention
- 非因果建模：双向或多方向扫描，消除因果序列的方向偏差
- 混合解码器：VSSD blocks + 局部 attention blocks

## 与 TextMamba3D 的关系

- **最接近的架构竞品**：同为 SSM/Mamba 用于医学分割
- **关键区别**：VSSD-UNet **没有文本引导**，是纯视觉 SSM 分割模型
- **TextMamba3D 的定位**：在 Mamba 医学分割基础上加入文本引导 = Mamba + Language
- **论文引用**：Related Work 中 SSM-based medical segmentation 的代表性工作
- **潜在威胁**：如果 VSSD-UNet 在 BraTS 上表现极强，TextMamba3D 的 baseline 优势可能被削弱

## 待深读要点

- [ ] VSSD-UNet 在 BraTS 上的具体表现（是否有 BraTS 实验）
- [ ] 非因果 SSM 的具体实现（bidirectional scan vs multi-directional）
- [ ] 与 SegMamba、U-Mamba、VM-UNet 等其他 Mamba 医学分割方法的对比
- [ ] ICLR 2025 reviewer 评分和评价
