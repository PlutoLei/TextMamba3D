# TextMamba3D Paper Reading Index / 论文阅读索引

> Step 4 (Text Guidance Effectiveness) related papers / 相关论文
> KG retrieval update / KG 检索更新: 2026-03-14 (56K paper knowledge graph / 56K 论文知识图谱)
> Numbering = learning priority for TextMamba3D (01 = highest) / 编号 = 学习优先级（01 = 最高）
> Last updated / 最后更新: 2026-03-17

---

## Overview / 优先级排序总览

| # | Paper / 论文 | Venue / 会议 | Core Value / 核心价值 | Overall / 评级 | Deep Read? / 值得精读？ | Status / 状态 |
|---|-------------|-------------|----------------------|---------------|----------------------|--------------|
| 01 | [CRIS](./01_CRIS_CVPR2022/) | CVPR 2022 | Text-to-pixel contrastive = per-pixel BCE — V4.3 TextToVoxelLoss direct source / V4.3 TextToVoxelLoss 直接来源 | Strong | Yes / 是 | ✅ Deep Read / 深读 |
| 02 | [LAVT](./02_LAVT_CVPR2022/) | CVPR 2022 | Multiplicative early fusion (PWAM) — V4.3 PWAM3D direct source / 乘性 early fusion (PWAM) — V4.3 PWAM3D 直接来源 | Strong | Yes / 是 | ✅ Deep Read / 深读 |
| 03 | [RLEG](./03_RLEG_ICML2023/) | ICML 2023 | Embedding-level diffusion augmentation — V4.3 EmbeddingPerturbation source / V4.3 EmbeddingPerturbation 来源 | Strong | Yes / 是 | ✅ Analyzed / 已分析 |
| 04 | [LViT](./04_LViT_TMI2023/) | IEEE TMI 2023 | First text-guided medical segmentation Transformer; closest methodological competitor / 首个文本引导医学分割 Transformer，最直接的方法类竞品 | Moderate-Strong (0.77) | Yes / 是 | ✅ Analyzed / 已分析 |
| 05 | [GLoRIA](./05_GLoRIA_ICCV2021/) | ICCV 2021 | Medical VL pretraining, global-local contrastive, word-level alignment / 医学 VL 预训练，全局-局部对比，词级对齐 | Strong | Yes / 是 | ✅ Analyzed / 已分析 |
| 06 | [DenseCLIP](./06_DenseCLIP_CVPR2022/) | CVPR 2022 | Pixel-text matching + context-aware prompting; validates per-pixel text alignment / 像素-文本匹配 + 上下文感知提示，验证逐像素文本对齐 | Strong (0.87) | Yes / 是 | ✅ Analyzed / 已分析 |
| 07 | [TextBraTS](./07_TextBraTS_MICCAI2025/) | MICCAI 2025 | Swin UNETR + BioBERT SeqCA, closest work to TextMamba3D; BraTS Mean Dice 85.3% / Swin UNETR + BioBERT 双向 SeqCA，与 TextMamba3D 最直接相关；BraTS Mean Dice 85.3% | Strong | Yes (Priority) / 是（优先） | ✅ Deep Read / 深读 |
| 08 | [VSSD-UNet](./08_VSSD-UNet_ICLR2025/) | ICLR 2025 | SSM + medical segmentation, non-causal modeling (Mamba-family strongest competitor) / SSM + 医学分割，非因果建模（Mamba 同族最强竞品） | Strong | Yes / 是 | ✅ Analyzed / 已分析 |
| 09 | [CKD-TransBTS](./09_CKD-TransBTS_TMI2022/) | IEEE TMI 2022 | Clinical knowledge-driven Transformer, same BraTS dataset / 临床知识驱动 Transformer，同 BraTS 数据集 | Moderate-Strong | Yes / 是 | ✅ Analyzed / 已分析 |
| 10 | [SwinBTS](./10_SwinBTS_BrainSci2022/) | Brain Sciences 2022 | Swin Transformer BraTS segmentation (TextBraTS family baseline) / Swin Transformer BraTS 分割（TextBraTS 同族 baseline） | Moderate | No / 否 | ✅ Analyzed / 已分析 |
| 11 | [Back-Modality](./11_Back-Modality_NeurIPS2023/) | NeurIPS 2023 | Cross-modal backward augmentation framework / 跨模态反向增强框架 | Moderate | No / 否 | ✅ Analyzed / 已分析 |
| 12 | [MedSAM](./12_MedSAM_NatComms2024/) | Nature Comms 2024 | SAM medical adaptation, prompt-based universal segmentation / SAM 医学适配，prompt-based 通用分割 | Moderate | Yes / 是 | ✅ Analyzed / 已分析 |
| 13 | [RegionCLIP](./13_RegionCLIP_CVPR2022/) | CVPR 2022 | Region-level template captioning / 区域级 template captioning | Moderate | Yes / 是 | ✅ Analyzed / 已分析 |
| 14 | [3D-CT-GPT++](./14_3D-CT-GPT++_ICLR2025/) | ICLR 2025 | 3D radiology report generation with DPO / 3D 放射学报告生成 (DPO) | Moderate | Yes / 是 | ✅ Analyzed / 已分析 |
| 15 | [MIMIC-R3G](./15_MIMIC-R3G_ICLR2025/) | ICLR 2025 | LLM-based radiology report generation pipeline / 基于 LLM 的放射学报告生成管线 | Moderate | Yes / 是 | ✅ Analyzed / 已分析 |
| 16 | [Barlow Twins Analysis](./16_BarlowTwins-Analysis_ICLR2025/) | arXiv 2021 | Cross-correlation normalization analysis / 交叉相关归一化分析 | Moderate | Applied: No; Theory: Yes / 应用否；理论是 | ✅ Analyzed / 已分析 |
| 17 | [E2ENet](./17_E2ENet_NeurIPS2024/) | NeurIPS 2024 | Dynamic sparse fusion for efficient 3D segmentation / 动态稀疏融合高效 3D 分割 | Moderate | Yes / 是 | ✅ Analyzed / 已分析 |
| 18 | [SynthSeg](./18_SynthSeg_MedImageAnal/) | Med Image Anal 2023 | Extreme domain randomization, label-to-image synthesis / 极端域随机化 label-to-image synthesis | Moderate | Yes / 是 | ✅ Analyzed / 已分析 |

---

## Tier Ranking Logic / 排序逻辑

**Tier 1 (#01-03): V4.3 Direct Design Sources / V4.3 直接设计来源** — Deep-read papers with module-level borrowing / 已深读，模块级借鉴

**Tier 2 (#04-06): Core Methodological References / 方法类核心参考** — Key methodology papers for text-guided segmentation / 文本引导分割的关键方法论文

**Tier 3 (#07-10): Closest Competitor & BraTS Baselines / 最直接竞品 & BraTS baseline** — TextBraTS (closest related work), VSSD-UNet (Mamba competitor), BraTS baselines / TextBraTS（最直接竞品），VSSD-UNet（Mamba 竞品），BraTS baseline

**Tier 4 (#11-18): Extended References / 扩展参考** — Cross-modal augmentation, foundation models, report generation, theory / 跨模态增强、基础模型、报告生成、理论分析

---

## Thematic Groups / 按主题分类

### Text Injection Mechanisms / 文本注入机制
- **#01 CRIS** — Text-conditioned dynamic conv + per-pixel BCE / 文本条件化动态卷积 + 逐像素 BCE
- **#02 LAVT** — Multiplicative PWAM + gated residual (early fusion) / 乘性 PWAM + 门控残差（早期融合）
- **#06 DenseCLIP** — Pixel-text matching + context-aware prompting / 像素-文本匹配 + 上下文感知提示
- **#05 GLoRIA** — Global-local contrastive with word-level alignment / 全局-局部对比 + 词级对齐

### Text-Guided Medical Segmentation / 文本引导医学分割
- **#04 LViT** — Language-Vision Transformer, text-assisted pseudo labels / 视觉-语言 Transformer，文本辅助伪标签
- **#07 TextBraTS** — Swin UNETR + BioBERT bidirectional SeqCA, BraTS Mean Dice 85.3% / Swin UNETR + BioBERT 双向 SeqCA，BraTS Mean Dice 85.3%

### Text/Embedding Augmentation / 文本/Embedding 增强
- **#03 RLEG** — Diffusion-based embedding generation / 基于扩散的嵌入生成
- **#11 Back-Modality** — Cross-modal augmentation bridging / 跨模态增强桥接
- **#13 RegionCLIP** — Template captioning for regions / 区域级模板描述生成

### SSM/Mamba Medical Segmentation / SSM/Mamba 医学分割
- **#08 VSSD-UNet** — Vision State Space Duality + UNet, non-causal SSM, ICLR 2025 / 视觉状态空间对偶 + UNet，非因果 SSM

### BraTS Brain Tumor Segmentation / BraTS 脑肿瘤分割
- **#09 CKD-TransBTS** — Clinical knowledge-driven, modality-grouped cross-attention / 临床知识驱动，模态分组交叉注意力
- **#10 SwinBTS** — Swin Transformer + CNN hybrid / Swin Transformer + CNN 混合架构

### Medical Foundation Models / 医学基础模型
- **#12 MedSAM** — SAM medical adaptation, prompt-based universal segmentation / SAM 医学适配，prompt-based 通用分割

### Efficient 3D Segmentation / 高效 3D 分割
- **#17 E2ENet** — Dynamic sparse fusion, efficient 3D segmentation / 动态稀疏融合，高效 3D 分割

### Medical Report Generation / 医学报告生成
- **#14 3D-CT-GPT++** — 3D volume to radiology report (DPO) / 3D 体积到放射学报告（DPO）
- **#15 MIMIC-R3G** — LLM pipeline for instructional reports / 基于 LLM 的指令式报告管线

### Theory & Analysis / 理论/分析
- **#16 Barlow Twins Analysis** — Essence of negative-free contrastive learning / 无负样本对比学习的本质
- **#18 SynthSeg** — Extreme domain randomization philosophy / 极端域随机化哲学

---

## How to Add a New Paper / 如何添加新论文

1. Create folder / 创建文件夹: `Papers/<NN_ShortName_VenueYear>/`
2. Place the PDF in the folder / 将 PDF 放入文件夹
3. Run `/academic-paper-reading` on the PDF (Standard depth) / 运行论文阅读 skill（Standard 深度）
4. The skill auto-generates `README.md`, `*_analysis_en.md`, `*_analysis_cn.md` / skill 自动生成分析文件
5. Update this overview table / 更新本总览表
