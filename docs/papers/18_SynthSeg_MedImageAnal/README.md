# SynthSeg: Segmentation of Brain MRI Scans of Any Contrast and Resolution Without Retraining

**SynthSeg：无需重训练即可分割任意对比度和分辨率的脑部 MRI**

| Item | Detail |
|------|--------|
| Authors | Benjamin Billot, Douglas N. Greve, Oula Puonti, Axel Thielscher, Koen Van Leemput, Bruce Fischl, Adrian V. Dalca, Juan Eugenio Iglesias |
| Venue | Medical Image Analysis, 2023 |
| Paper Type | Methodological |
| Code | [github.com/BBillot/SynthSeg](https://github.com/BBillot/SynthSeg) |

## Key Contributions / 核心贡献

1. **Full Domain Randomisation for Brain MRI:** Trains a CNN exclusively on synthetic data from a generative model with fully randomised contrast, resolution, morphology, bias field, and noise -- no real images needed.
   - **全面域随机化：** 仅使用成像参数全面随机化的生成模型产出的合成数据训练 CNN，无需真实图像。

2. **Joint Contrast and Resolution Invariance:** The first single model to segment brain MRI across 6 modalities and 10 resolutions without retraining or fine-tuning.
   - **对比度与分辨率联合不变性：** 首个无需重训练即可跨 6 种模态和 10 种分辨率分割脑部 MRI 的单一模型。

3. **FreeSurfer Integration:** The trained model is distributed with FreeSurfer, providing ~10-second inference per scan on a standard GPU.
   - **FreeSurfer 集成：** 训练模型已集成至 FreeSurfer，标准 GPU 上约 10 秒完成单例分割。

## Key Results / 关键结果

| Metric | T1-39 (training domain) | 9 Target Domains | Resolution Robustness |
|--------|------------------------|-----------------|----------------------|
| Dice | 0.88 (vs. 0.91 supervised) | Best on 6/9 with statistical significance | -3.8 Dice pts from 1mm to 7mm |
| SD95 | 1.5 mm | Best on 9/9 domains | -- |
| vs. SAMSEG | +3 Dice points average | Outperforms on all non-T1 domains | SAMSEG loses 7.6 pts (1mm to 7mm) |

## Files / 文件

| File | Description |
|------|-------------|
| `SynthSeg_MedImageAnal.pdf` | Original paper / 原始论文 |
| `SynthSeg_analysis_en.md` | Standard-depth analysis (English) / 标准深度分析（英文） |
| `SynthSeg_analysis_cn.md` | Standard-depth analysis (Chinese) / 标准深度分析（中文） |

## Relevance to TextMamba3D / 与 TextMamba3D 的关联

SynthSeg's domain randomisation paradigm -- training on synthetic data with extreme parameter variation -- is directly applicable to building domain-agnostic 3D segmentation models. The key insight that a CNN can learn robust anatomical features from shape alone (without realistic intensities) informs training strategies for TextMamba3D when dealing with heterogeneous multi-site MRI data.

SynthSeg 的域随机化范式——在参数极端变化的合成数据上训练——可直接应用于构建域无关的三维分割模型。CNN 仅从形状（不依赖真实强度）即可学习鲁棒解剖特征这一核心发现，为 TextMamba3D 处理异质多中心 MRI 数据时的训练策略提供了启示。
