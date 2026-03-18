# SwinBTS | SwinBTS 论文分析

**SwinBTS: A Method for 3D Multimodal Brain Tumor Segmentation Using Swin Transformer**

| Item | Detail |
|------|--------|
| Authors | Yun Jiang, Yuan Zhang, Xin Lin, Jinkun Dong, Tongtong Cheng, Jing Liang |
| Venue | Brain Sciences (MDPI), 2022 |
| DOI | https://doi.org/10.3390/brainsci12060797 |
| Paper Type | Methodological |
| Analysis Depth | Standard |
| Analysis Date | 2026-03-17 |
| Analyzer | Claude Code (academic-paper-reading skill, pdftoppm) |

## Files | 文件

| File | Description |
|------|-------------|
| [SwinBTS_BrainSci2022.pdf](./SwinBTS_BrainSci2022.pdf) | Original paper / 原始论文 |
| [SwinBTS_analysis_en.md](./SwinBTS_analysis_en.md) | English analysis (Standard depth) / 英文分析 |
| [SwinBTS_analysis_cn.md](./SwinBTS_analysis_cn.md) | Chinese analysis (Standard depth) / 中文分析 |
| [notes.md](./notes.md) | Quick notes / 速记笔记 |

## Key Contributions | 核心贡献

1. **3D Swin Transformer encoder-decoder:** Uses 3D Swin Transformer as both encoder and decoder backbone (not just as an attention layer), extending SwinUNet from 2D to 3D volumetric segmentation.
2. **NFCE module:** A depth-wise separable convolution bridge (Conv3d 1x1x1 + DwConv3d 3x3x3 + Conv3d 1x1x1) inserted between Swin Transformer stages and down/upsampling layers, reducing information loss at resolution transitions (+0.87% average Dice).
3. **ETrans module:** An Enhanced Transformer at the bottleneck that elevates convolution from second-order to third-order feature mapping via Hadamard products, improving detail extraction for small tumor sub-regions (+1.32% average Dice).

1. **3D Swin Transformer 编码器-解码器：** 将 3D Swin Transformer 同时用作编码器和解码器骨干（而非仅作为注意力层），将 SwinUNet 从 2D 扩展至 3D 体积分割。
2. **NFCE 模块：** 深度可分离卷积桥接模块（Conv3d 1x1x1 + DwConv3d 3x3x3 + Conv3d 1x1x1），插入 Swin Transformer 阶段与上下采样层之间，减少分辨率转换时的信息损失（+0.87% 平均 Dice）。
3. **ETrans 模块：** 位于瓶颈的增强 Transformer，通过 Hadamard 积将卷积从二阶提升至三阶特征映射，改善小肿瘤子区域的细节提取（+1.32% 平均 Dice）。

## Key Results | 核心结果

| Dataset | ET Dice | TC Dice | WT Dice | Avg Dice | Avg HD95 (mm) |
|---------|---------|---------|---------|----------|---------------|
| BraTS 2019 (56-case test) | 74.43% | 79.28% | 89.75% | 81.15% | -- |
| BraTS 2020 (125-case val) | 77.36% | 80.30% | 89.06% | 82.24% | 17.06 |
| BraTS 2021 (219-case val) | 83.21% | 84.75% | 91.83% | 86.60% | 11.39 |

## Key Takeaway | 核心要点

SwinBTS adapts the 2D Swin Transformer to 3D multimodal brain tumor segmentation by extending shifted window attention to volumetric data. The encoder-decoder architecture uses 3D Swin Transformer blocks at all stages, with NFCE modules bridging resolution transitions and an ETrans bottleneck that achieves third-order feature mapping via Hadamard products. On BraTS 2019, it reaches 81.15% average Dice (outperforming TransBTS by +1.32% and VTU-Net by +0.76%). On BraTS 2021, it achieves 86.60% average Dice and 11.39 mm average HD95. The main weakness is boundary precision: HD95 on BraTS 2020 (17.06 mm) lags behind TransBTS (15.06 mm).

SwinBTS 将 2D Swin Transformer 适配至 3D 多模态脑肿瘤分割，将移位窗口注意力扩展至体积数据。编码器-解码器架构在所有阶段使用 3D Swin Transformer 块，NFCE 模块桥接分辨率转换，ETrans 瓶颈通过 Hadamard 积实现三阶特征映射。在 BraTS 2019 上平均 Dice 达 81.15%（超过 TransBTS +1.32%、VTU-Net +0.76%）。在 BraTS 2021 上平均 Dice 达 86.60%，平均 HD95 为 11.39 mm。主要短板为边界精度：BraTS 2020 上 HD95（17.06 mm）弱于 TransBTS（15.06 mm）。

## Relevance to TextMamba3D | 与 TextMamba3D 的关系

SwinBTS belongs to the same architectural family as Swin UNETR, which serves as the base encoder in TextBraTS (MICCAI 2025). It provides a simpler BraTS baseline for comparing Swin Transformer versus Mamba-based approaches for 3D brain tumor segmentation. The shifted window attention mechanism is the direct comparison point for Mamba's linear-complexity long-range modeling. The ETrans module's Hadamard-product attention at the bottleneck may offer design inspiration for enhancing local detail extraction in TextMamba3D's architecture, particularly for the small ET sub-region.

SwinBTS 与 Swin UNETR 属同一架构家族，后者是 TextBraTS（MICCAI 2025）的基础编码器。它为比较 Swin Transformer 与 Mamba 方法在 3D 脑肿瘤分割中的表现提供了更简单的 BraTS 基线。移位窗口注意力机制是 Mamba 线性复杂度长程建模的直接对比参照。ETrans 模块在瓶颈处的 Hadamard 积注意力可为 TextMamba3D 架构中增强局部细节提取（尤其是小型 ET 子区域）提供设计启发。
