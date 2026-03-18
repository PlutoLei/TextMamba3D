# VSSD-UNet | VSSD-UNet 论文分析

**Vision State Space Duality for Medical Image Segmentation: Enhancing Precision through Non-Causal Modeling**

| Item | Detail |
|------|--------|
| Authors | Anonymous (double-blind review) |
| Venue | ICLR 2025 (under review) |
| Paper Type | Methodological |
| Analysis Depth | Standard |
| Analysis Date | 2026-03-17 |
| Analyzer | Claude Code (academic-paper-reading skill, pdftoppm) |

## Files | 文件

| File | Description |
|------|-------------|
| [VSSD-UNet_ICLR2025.pdf](./VSSD-UNet_ICLR2025.pdf) | Original paper / 原始论文 |
| [VSSD-UNet_analysis_en.md](./VSSD-UNet_analysis_en.md) | English analysis (Standard depth) / 英文分析 |
| [VSSD-UNet_analysis_cn.md](./VSSD-UNet_analysis_cn.md) | Chinese analysis (Standard depth) / 中文分析 |
| [notes.md](./notes.md) | Quick notes / 速记笔记 |

## Key Takeaway | 核心要点

VSSD-UNet adapts Non-Causal State Space Duality (NC-SSD) into a UNet architecture for 2D skin lesion segmentation, achieving 78.30% mIoU on ISIC2017 and 80.65% on ISIC2018 -- outperforming 14 baselines including VMUNet, H-vmunet, and ULVM-UNet. The core innovation is transforming the SSD state transition matrix from a matrix to a scalar, enabling bidirectional non-causal processing without multi-directional scan routes.

VSSD-UNet 将非因果状态空间对偶 (NC-SSD) 整合进 UNet 架构用于 2D 皮肤病变分割，在 ISIC2017 达到 78.30% mIoU、ISIC2018 达到 80.65% mIoU，超越 VMUNet、H-vmunet、ULVM-UNet 等 14 种基线方法。核心创新是将 SSD 状态转移矩阵从矩阵简化为标量，实现无需多方向扫描路线的双向非因果处理。

## Relevance to TextMamba3D | 与 TextMamba3D 的关系

The NC-SSD formulation for removing causal constraints in state space models is conceptually relevant to any Mamba-based 3D medical segmentation architecture. The paper is limited to 2D evaluation and does not explore text-guided or 3D settings.

NC-SSD 去除状态空间模型因果约束的公式化方法，对任何基于 Mamba 的 3D 医学分割架构在概念层面具有参考价值。该论文仅限于 2D 评估，未探索文本引导或 3D 设定。
