# E2ENet: Dynamic Sparse Feature Fusion for Accurate and Efficient 3D Medical Image Segmentation

**E2ENet：面向精确高效三维医学图像分割的动态稀疏特征融合**

| Item | Detail |
|------|--------|
| Authors | Boqian Wu, Qiao Xiao, Shiwei Liu, Lu Yin, Mykola Pechenizkiy, Decebal Constantin Mocanu, Maurice van Keulen, Elena Mocanu |
| Venue | NeurIPS 2024 |
| Paper Type | Methodological |
| Code | [github.com/boqian333/E2ENet-Medical](https://github.com/boqian333/E2ENet-Medical) |

## Key Contributions / 核心贡献

1. **Dynamic Sparse Feature Fusion (DSFF):** Learnable binary masks with L1-based prune-and-grow topology updates to sparsify multi-scale feature connections during training, reducing parameters and FLOPs without sacrificing segmentation accuracy.
   - **动态稀疏特征融合 (DSFF)：** 使用可学习二值掩码和基于 L1 范数的剪枝-再生策略在训练中稀疏化多尺度特征连接，降低参数量和计算量而不损失分割精度。

2. **Restricted Depth-Shift in 3D Convolution:** Channel shifting by {-1, 0, +1} along the depth axis before 1x3x3 2D convolutions, capturing inter-slice 3D context at 2D computational cost.
   - **受限深度位移卷积：** 在 1x3x3 二维卷积前沿深度轴进行 {-1, 0, +1} 通道位移，以二维计算代价捕获切片间三维上下文。

## Key Results / 关键结果

| Benchmark | E2ENet mDice | Params | FLOPs | vs. nnUNet |
|-----------|-------------|--------|-------|------------|
| AMOS-CT (S=0.8) | 90.3% | 9.44M | 778.74G | -0.2% mDice, -69% params, -27% FLOPs |
| BraTS/MSD (S=0.7) | 74.5% | 11.24M | 1067.06G | +0.4% mDice, -64% params |
| BTCV (S=0.7) | 88.2% | 11.25M | 449.00G | +0.2% mDice, -64% params |

## Files / 文件

| File | Description |
|------|-------------|
| `E2ENet_NeurIPS2024.pdf` | Original paper / 原始论文 |
| `E2ENet_analysis_en.md` | Standard-depth analysis (English) / 标准深度分析（英文） |
| `E2ENet_analysis_cn.md` | Standard-depth analysis (Chinese) / 标准深度分析（中文） |
| `notes.md` | Additional reading notes / 补充阅读笔记 |

## Relevance to TextMamba3D / 与 TextMamba3D 的关联

E2ENet's DSFF mechanism demonstrates that sparse multi-scale feature fusion can match dense fusion performance at a fraction of the computational cost. The restricted depth-shift strategy offers a lightweight alternative to full 3D convolutions for capturing inter-slice context, directly relevant to efficient 3D backbone design in TextMamba3D.

E2ENet 的 DSFF 机制证明稀疏多尺度特征融合可以以极低的计算代价匹配密集融合的性能。受限深度位移策略为捕获切片间上下文提供了一种轻量替代全三维卷积的方案，直接关联到 TextMamba3D 中高效三维骨干网络的设计。
