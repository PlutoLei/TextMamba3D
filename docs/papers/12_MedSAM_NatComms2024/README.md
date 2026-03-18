# 11. MedSAM: Segment Anything in Medical Images

**Paper:** Segment Anything in Medical Images
**Authors:** Jun Ma, Yuting He, Feifei Li, Lin Han, Chenyu You, Bo Wang
**Venue:** Nature Communications, 2024 (arXiv:2304.12306)
**Pages:** 20

---

## Summary / 概要

MedSAM is a foundation model for universal medical image segmentation, fine-tuned from SAM on 1,570,263 image-mask pairs spanning 10 imaging modalities and 30+ cancer types. Using bounding box prompts, MedSAM outperforms vanilla SAM and rivals 20 modality-wise specialist models across 146 validation tasks. It reduces expert annotation time by ~83%.

MedSAM 是一个通用医学图像分割基础模型，在 SAM 基础上使用 1,570,263 对图像-掩码数据微调，覆盖 10 种成像模态和 30 余种癌症类型。通过边界框提示，MedSAM 超越原始 SAM 并与 20 个按模态训练的专家模型性能相当，在 146 项验证任务上均有验证。辅助标注可将专家标注时间减少约 83%。

## Key Results / 核心结果

| Metric | Value | Context |
|--------|-------|---------|
| DSC on nasopharynx cancer | 87.8% (IQR: 85.0-91.4%) | 52.3% improvement over SAM |
| Annotation time reduction | 82.37-82.95% | 2 expert radiologists, 10 adrenal tumor cases |
| Training data | 1,570,263 image-mask pairs | 10 modalities, 30+ cancer types |
| Trainable parameters | 93,729,252 | Image encoder + mask decoder |

## Files / 文件

| File | Description |
|------|-------------|
| `MedSAM_NatComms2024.pdf` | Original paper |
| `MedSAM_analysis_en.md` | Standard depth analysis (English) |
| `MedSAM_analysis_cn.md` | Standard depth analysis (Chinese) |

## Relevance to TextMamba3D / 与 TextMamba3D 的关联

MedSAM demonstrates that a single promptable foundation model can achieve universal medical image segmentation. Its ViT-Base encoder features and bounding box prompt mechanism are directly relevant to TextMamba3D's goal of text-guided 3D medical segmentation. MedSAM's 2D-only limitation (no volumetric context) represents exactly the gap that TextMamba3D aims to address.

---

*Analysis generated on 2026-03-17 by Claude Code (academic-paper-reading skill, pdftoppm)*
