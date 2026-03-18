# TextBraTS: Text-Guided Volumetric Brain Tumor Segmentation with Innovative Dataset Development and Fusion Module Exploration

**TextBraTS：创新数据集开发与融合模块探索驱动的文本引导体积脑肿瘤分割**

| Item | Detail |
|------|--------|
| Authors | Xiaoyu Shi, Rahul Kumar Jain, Yinhao Li, Ruibo Hou, Jingliang Cheng, Jie Bai, Guohua Zhao, Lanfen Lin, Rui Xu, Yen-wei Chen |
| Venue | MICCAI 2025 |
| Paper Type | Empirical |
| Code | [github.com/Jupitern52/TextBraTS](https://github.com/Jupitern52/TextBraTS) |

## Key Contributions / 核心贡献

1. **TextBraTS Dataset:** The first publicly available volume-level text-image brain MRI tumor segmentation dataset (369 cases from BraTS2020 with GPT-4o-generated and dual-radiologist verified textual annotations).
   - **TextBraTS 数据集：** 首个公开的体积级文本-图像脑部 MRI 肿瘤分割数据集（369 例，GPT-4o 生成 + 双放射科医师验证）。

2. **Sequential Cross-Attention (SeqCA) Fusion:** A two-step cross-attention module (T2I then I2T) that integrates BioBERT text features with SwinUNETR image features at the bottleneck layer.
   - **顺序交叉注意力 (SeqCA) 融合：** 在瓶颈层通过两步交叉注意力（先 T2I 再 I2T）整合 BioBERT 文本特征与 SwinUNETR 图像特征。

3. **Text Template Ablation:** Comprehensive experiments on 4 text formats (raw, location-only, features-only, fully templated) demonstrating that structured templates yield the best segmentation performance.
   - **文本模板消融：** 对 4 种文本格式的全面实验，证明结构化模板取得最优分割性能。

## Key Results / 关键结果

| Metric | ET | WT | TC | Avg. |
|--------|-----|-----|-----|------|
| Dice (%) | 83.3 | 89.9 | 82.8 | 85.3 |
| HD95 (mm) | 4.58 | 5.48 | 5.34 | 5.13 |
| vs. SwinUNETR (image-only) | +2.3 | +0.4 | +2.0 | +1.5 Dice |
| vs. NestedFormer (prior best) | +0.7 | +0.4 | +2.6 | +1.2 Dice |

## Files / 文件

| File | Description |
|------|-------------|
| `TextBraTS_MICCAI2025.pdf` | Original paper / 原始论文 |
| `TextBraTS_analysis_en.md` | Standard-depth analysis (English) / 标准深度分析（英文） |
| `TextBraTS_analysis_cn.md` | Standard-depth analysis (Chinese) / 标准深度分析（中文） |
| `deep_reading.md` | Additional deep reading notes / 补充深度阅读笔记 |

## Relevance to TextMamba3D / 与 TextMamba3D 的关联

TextBraTS is the most directly relevant paper to TextMamba3D, as both address text-guided 3D medical image segmentation. Key takeaways for TextMamba3D:
- The TextBraTS dataset (369 cases with volume-level text-image pairs) is a candidate benchmark for TextMamba3D evaluation.
- SeqCA provides a baseline fusion module that TextMamba3D's Mamba-based architecture could improve upon.
- The finding that fully templated text outperforms raw text (85.3% vs. 84.6% Dice) informs text preprocessing strategy.
- The GPT-4o + expert refinement annotation pipeline is directly reusable for constructing text annotations for other datasets.

TextBraTS 是与 TextMamba3D 最直接相关的论文，两者均针对文本引导的三维医学图像分割。对 TextMamba3D 的关键启示：
- TextBraTS 数据集（369 例体积级文本-图像对）是 TextMamba3D 评估的候选基准。
- SeqCA 提供了 TextMamba3D 基于 Mamba 架构可以改进的基线融合模块。
- 全模板化文本优于原始文本的发现（85.3% vs. 84.6% Dice）为文本预处理策略提供了指导。
- GPT-4o + 专家精炼的标注流水线可直接复用于为其他数据集构建文本标注。
