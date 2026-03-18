# 12. RegionCLIP: Region-based Language-Image Pretraining

**Paper:** RegionCLIP: Region-based Language-Image Pretraining
**Authors:** Yiwu Zhong, Jianwei Yang, Pengchuan Zhang, Chunyuan Li, Noel Codella, Liunian Harold Li, Luowei Zhou, Xiyang Dai, Lu Yuan, Yin Li, Jianfeng Gao
**Venue:** CVPR 2022 (arXiv:2112.09106)
**Pages:** 12

---

## Summary / 概要

RegionCLIP extends CLIP from image-level to region-level visual-semantic alignment by generating pseudo region-text pairs from concept pools and a CLIP teacher model. Combined with contrastive learning and knowledge distillation, the pretrained visual encoder achieves state-of-the-art open-vocabulary object detection on COCO (35.2 AP50 novel, +37.7% over OVR) and LVIS (32.3 mAP, +3.6 over ViLD), while supporting zero-shot inference.

RegionCLIP 将 CLIP 从图像级扩展至区域级视觉-语义对齐，通过概念池和 CLIP 教师模型生成伪区域-文本对。结合对比学习与知识蒸馏，预训练的视觉编码器在 COCO（新类别 35.2 AP50，较 OVR 提升 37.7%）和 LVIS（32.3 mAP，较 ViLD 提升 +3.6）上达到开放词汇检测 SOTA，同时支持零样本推理。

## Key Results / 核心结果

| Benchmark | Metric | Value | Comparison |
|-----------|--------|-------|------------|
| COCO (Novel) | AP50 | 35.2 (RN50) / 43.3 (RN50x4) | OVR: 27.5; ViLD*: 27.6 |
| COCO (All) | AP50 | 50.4 (RN50) / 55.7 (RN50x4) | OVR: 39.9; ViLD*: 51.3 |
| LVIS | mAP | 32.3 (RN50x4) | ViLD: 28.7 |
| Zero-shot (COCO, GT) | AP50 All | 65.6 (RN50x4) | CLIP: 58.3; OVR: 44.5 |

## Files / 文件

| File | Description |
|------|-------------|
| `RegionCLIP_CVPR2022.pdf` | Original paper |
| `RegionCLIP_analysis_en.md` | Standard depth analysis (English) |
| `RegionCLIP_analysis_cn.md` | Standard depth analysis (Chinese) |

## Relevance to TextMamba3D / 与 TextMamba3D 的关联

RegionCLIP's core paradigm -- bootstrapping region-level text alignment from image-level vision-language models using pseudo labels and concept pools -- is directly relevant to TextMamba3D's challenge of aligning text descriptions with 3D volumetric regions. The concept pool construction and pseudo labeling strategy could be adapted for generating text supervision for 3D medical structures without exhaustive human annotation.

---

*Analysis generated on 2026-03-17 by Claude Code (academic-paper-reading skill, pdftoppm)*
