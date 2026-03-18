# CKD-TransBTS | CKD-TransBTS 论文分析

**CKD-TransBTS: Clinical Knowledge-Driven Hybrid Transformer with Modality-Correlated Cross-Attention for Brain Tumor Segmentation**

| Item | Detail |
|------|--------|
| Authors | Jianwei Lin, Jiatai Lin, Cheng Lu, Hao Chen, et al. |
| Venue | IEEE Transactions on Medical Imaging (TMI), 2022 |
| Paper Type | Methodological |
| Analysis Depth | Standard |
| Analysis Date | 2026-03-17 |
| Analyzer | Claude Code (academic-paper-reading skill, pdftoppm) |

## Files | 文件

| File | Description |
|------|-------------|
| [CKD-TransBTS_TMI2022.pdf](./CKD-TransBTS_TMI2022.pdf) | Original paper / 原始论文 |
| [CKD-TransBTS_analysis_en.md](./CKD-TransBTS_analysis_en.md) | English analysis (Standard depth) / 英文分析 |
| [CKD-TransBTS_analysis_cn.md](./CKD-TransBTS_analysis_cn.md) | Chinese analysis (Standard depth) / 中文分析 |
| [notes.md](./notes.md) | Quick notes / 速记笔记 |

## Key Takeaway | 核心要点

CKD-TransBTS re-groups four MRI modalities into two clinically correlated pairs -- {T1, T1Gd} for tumor core and {T2, T2FLAIR} for edema -- mirroring radiological diagnostic practice. A dual-branch hybrid encoder with Modality-Correlated Cross-Attention (MCCA) processes paired modalities, and a Trans&CNN Feature Calibration (TCFC) decoder bridges the transformer-CNN semantic gap via 3D direction-wise attention. On BraTS 2021, it achieves 0.9066 mean Dice and 6.22 mm mean HD95, with ET HD95 of 5.93 mm (3 mm better than the second-best).

CKD-TransBTS 将四种 MRI 模态按临床相关性重组为两对：{T1, T1Gd} 用于肿瘤核心评估，{T2, T2FLAIR} 用于水肿评估，模拟放射科诊断流程。双分支混合编码器通过 MCCA 处理配对模态，TCFC 解码器通过 3D 方向级注意力弥合 Transformer-CNN 语义差距。在 BraTS 2021 上达到 0.9066 平均 Dice 和 6.22 mm 平均 HD95，其中 ET HD95 为 5.93 mm（比次优低 3 mm）。

## Relevance to TextMamba3D | 与 TextMamba3D 的关系

Directly relevant as a competitive baseline on BraTS benchmarks. The clinical knowledge-driven modality grouping principle and the MCCA cross-attention design for multi-modal fusion provide architectural inspiration for integrating text guidance with MRI modality processing in TextMamba3D.

作为 BraTS 基准上的竞争性基线直接相关。临床知识驱动的模态分组原则和 MCCA 跨模态注意力设计，为 TextMamba3D 中文本引导与 MRI 模态处理的整合提供架构层面的启发。
