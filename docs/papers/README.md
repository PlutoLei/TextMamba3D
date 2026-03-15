# TextMamba3D 论文阅读索引

> Step 4 (Text Guidance Effectiveness) 相关论文
> KG 检索更新: 2026-03-14（56K 论文知识图谱）
> 编号 = 与 TextMamba3D 的匹配度排序（1 = 最高）

## 优先级排序总览

| # | 论文 | 会议 | 核心价值 | 状态 |
|---|------|------|---------|------|
| 1 | [CRIS](./1.CRIS_CVPR2022/) | CVPR 2022 | Text-to-pixel contrastive = per-pixel BCE — V4.3 TextToVoxelLoss 直接来源 | ✅ 深读 |
| 2 | [LAVT](./2.LAVT_CVPR2022/) | CVPR 2022 | 乘性 early fusion (PWAM) — V4.3 PWAM3D 直接来源 | ✅ 深读 |
| 3 | [RLEG](./3.RLEG_ICML2023/) | ICML 2023 | Embedding-level diffusion augmentation — V4.3 EmbeddingPerturbation 来源 | ✅ 深读 |
| 4 | [LViT](./4.LViT_TMI2023/) | IEEE TMI | 首个文本引导医学分割 Transformer，最直接的方法类竞品 | 待细读 |
| 5 | [GLoRIA](./5.GLoRIA_ICCV2021/) | ICCV 2021 | 医学 VL 预训练，global-local contrastive, word-level 对齐 | 待细读 |
| 6 | [DenseCLIP](./6.DenseCLIP_CVPR2022/) | CVPR 2022 | Pixel-text matching + context-aware prompting，验证逐像素文本对齐 | 待细读 |
| 7 | [VSSD-UNet](./7.VSSD-UNet_ICLR2025/) | ICLR 2025 | SSM + 医学分割，非因果建模（Mamba 同族最强竞品） | 待细读 |
| 8 | [CKD-TransBTS](./8.CKD-TransBTS_TMI2022/) | IEEE TMI | 临床知识驱动 Transformer，同 BraTS 数据集 | 待细读 |
| 9 | [SwinBTS](./9.SwinBTS_BrainSci2022/) | Brain Sciences | Swin Transformer BraTS 分割（TextBraTS 同族 baseline） | 待细读 |
| 10 | [Back-Modality](./10.Back-Modality_NeurIPS2023/) | NeurIPS 2023 | 跨模态反向增强框架 | 待细读 |
| 11 | [MedSAM](./11.MedSAM_NatComms2024/) | Nature Comms | SAM 医学适配，prompt-based 通用分割（对比定位） | 待细读 |
| 12 | [RegionCLIP](./12.RegionCLIP_CVPR2022/) | CVPR 2022 | 区域级 template captioning | 待细读 |
| 13 | [3D-CT-GPT++](./13.3D-CT-GPT++_ICLR2025/) | ICLR 2025 | 3D radiology report generation with DPO | 待细读 |
| 14 | [MIMIC-R3G](./14.MIMIC-R3G_ICLR2025/) | ICLR 2025 | LLM-based radiology report generation pipeline | 待细读 |
| 15 | [Barlow Twins Analysis](./15.BarlowTwins-Analysis_ICLR2025/) | ICLR 2025 | Cross-correlation 归一化分析 | 待细读 |
| 16 | [E2ENet](./16.E2ENet_NeurIPS2024/) | NeurIPS 2024 | 动态稀疏融合高效 3D 分割 | 待细读 |
| 17 | [SynthSeg](./17.SynthSeg_MedImageAnal/) | Med Image Anal | 极端域随机化 label-to-image synthesis | 待细读 |

## 排序逻辑

**Tier 1 (1-3)：V4.3 直接设计来源** — 已深读，模块级借鉴

**Tier 2 (4-6)：方法类核心参考** — 文本引导分割的关键方法论文

**Tier 3 (7-9)：直接竞品 & BraTS baseline** — 同架构族或同数据集

**Tier 4 (10-17)：扩展参考** — 跨模态增强、基础模型、报告生成、理论分析

## 按主题分类

### 文本注入机制
- **#1 CRIS** — text-conditioned dynamic conv + per-pixel BCE
- **#2 LAVT** — 乘性 PWAM + 门控残差 (early fusion)
- **#6 DenseCLIP** — pixel-text matching + context-aware prompting
- **#5 GLoRIA** — global-local contrastive with word-level alignment

### 文本引导医学分割
- **#4 LViT** — Language-Vision Transformer，文本辅助伪标签
- *(外部)* **TextBraTS** (MICCAI 2025) — Swin UNETR + BioBERT 双向 SeqCA，BraTS Mean Dice 85.3%

### 文本/Embedding 增强
- **#3 RLEG** — diffusion-based embedding generation
- **#10 Back-Modality** — cross-modal augmentation bridging
- **#12 RegionCLIP** — template captioning for regions

### SSM/Mamba 医学分割
- **#7 VSSD-UNet** — Vision State Space Duality + UNet，非因果 SSM，ICLR 2025

### BraTS 脑肿瘤分割
- **#8 CKD-TransBTS** — 临床知识驱动，模态分组 cross-attention
- **#9 SwinBTS** — Swin Transformer + CNN 混合

### 医学基础模型
- **#11 MedSAM** — SAM 医学适配，prompt-based 通用分割

### 高效 3D 分割
- **#16 E2ENet** — 动态稀疏融合，高效 3D 分割

### 医学报告生成
- **#13 3D-CT-GPT++** — 3D volume → radiology report (DPO)
- **#14 MIMIC-R3G** — LLM pipeline for instructional reports

### 理论/分析
- **#15 Barlow Twins Analysis** — negative-free contrastive 的本质
- **#17 SynthSeg** — 极端域随机化哲学
