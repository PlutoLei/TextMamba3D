# 13. 3D-CT-GPT++ | ICLR 2025 (under review)

## Paper Info / 论文信息

| Field | Content |
|-------|---------|
| **Title** | 3D-CT-GPT++: Enhancing 3D Radiology Report Generation with Direct Preference Optimization and Large Vision-Language Models |
| **Authors** | Anonymous (double-blind review) |
| **Venue** | ICLR 2025 (under review) |
| **Paper Type** | Methodological |
| **Pages** | 23 (10 main + 13 appendix) |

## Quick Summary / 快速摘要

**EN:** 3D-CT-GPT++ proposes a vision-language model for automatic radiology report generation from 3D chest CT scans. It introduces three integrated innovations: (1) CTViT-V, an enhanced 3D encoder with a Slice Transformer using full bidirectional attention and relative position encoding to capture global inter-slice dependencies; (2) integration with LLaVA-1.5 / Vicuna-1.5 (7B) for report generation; (3) Direct Preference Optimization (DPO) with GPT-4-scored preference data to reduce hallucinations. Evaluated on CT-RATE (public, 21,304 cases) and Dataset-XY (private, 1,886 cases), the model achieves BLEU-4 13.32, ROUGE-L 0.3692, METEOR 0.3542, and GREEN 0.3527, outperforming baseline 3D-CT-GPT, RadFM, and M3D across all metrics.

**CN:** 3D-CT-GPT++ 提出一种面向 3D 胸部 CT 扫描的自动放射学报告生成视觉-语言模型。该模型包含三项集成创新：(1) CTViT-V，增强型 3D 编码器，引入具有全双向注意力和相对位置编码的切片 Transformer 以捕获全局切片间依赖；(2) 与 LLaVA-1.5 / Vicuna-1.5 (7B) 集成用于报告生成；(3) 利用 GPT-4 评分的偏好数据进行直接偏好优化（DPO）以减少幻觉。在 CT-RATE（公开，21,304 例）和 Dataset-XY（私有，1,886 例）上的评估表明，模型达到 BLEU-4 13.32、ROUGE-L 0.3692、METEOR 0.3542 和 GREEN 0.3527，在所有指标上超越基线 3D-CT-GPT、RadFM 和 M3D。

## Files / 文件列表

| File | Description |
|------|-------------|
| `3D-CT-GPT++_ICLR2025.pdf` | Original paper PDF / 原始论文 PDF |
| `3D-CT-GPT++_analysis_en.md` | English analysis (Standard depth) / 英文分析（标准深度） |
| `3D-CT-GPT++_analysis_cn.md` | Chinese analysis (Standard depth) / 中文分析（标准深度） |

## Key Contributions / 核心贡献

1. **CTViT-V encoder** -- Enhanced 3D CT encoder replacing causal temporal attention with full bidirectional Slice Transformer + relative position encoding; BLEU-1 improves by 7.3% (52.17 to 55.98) with reduced memory (30.6 GB vs. 36 GB for 3DViT at batch size 8).
2. **3D-CT-GPT++ architecture** -- Integration of CTViT-V with LLaVA-1.5 / Vicuna-1.5 (7B) via 2-layer MLP projection for 3D CT report generation.
3. **DPO with GPT-4 scoring** -- First application of Direct Preference Optimization to 3D CT report generation; GPT-4 scores 6 candidate reports per image on a 1-5 scale to construct preference pairs; GREEN improves from 0.2596 (SFT) to 0.3527 (SFT+DPO), a 35.9% relative gain.

## Key Results / 核心结果

| Metric | 3D-CT-GPT (Baseline) | 3D-CT-GPT++ (SFT+DPO) | Improvement |
|--------|----------------------|------------------------|-------------|
| BLEU-1 | 52.17 | 56.76 | +4.59 |
| BLEU-4 | 11.49 | 13.32 | +1.83 |
| ROUGE-L | 0.3353 | 0.3692 | +0.0339 |
| METEOR | 0.3308 | 0.3542 | +0.0234 |
| GREEN | -- | 0.3527 | -- |

## Relevance to TextMamba3D / 与 TextMamba3D 的关联

- **3D CT encoding:** CTViT-V's Slice Transformer with full bidirectional attention and RPE provides an alternative to Mamba-based 3D encoding for capturing global spatial dependencies across CT slices.
- **Report generation pipeline:** The four-stage training process (CT-CLIP pre-training, MLP pre-training, SFT, DPO) offers a replicable framework for any 3D medical image-to-report system.
- **DPO for hallucination reduction:** The GPT-4-scored preference data strategy and the finding that moderate-contrast pairs outperform extreme-contrast pairs (BLEU-4 13.32 vs. 12.41) are directly transferable to TextMamba3D's report generation objectives.
- **GREEN metric:** Demonstrates the value of hallucination-aware evaluation beyond standard NLG metrics for 3D medical report generation.

## Verdict / 结论

**Worth Deep Reading? Yes** -- Provides a complete pipeline for 3D CT report generation with explicit hallucination mitigation via DPO; the CTViT-V encoder and DPO methodology are directly relevant to TextMamba3D.

**是否值得深读？是** -- 提供了完整的 3D CT 报告生成管线，通过 DPO 显式抑制幻觉；CTViT-V 编码器和 DPO 方法论与 TextMamba3D 直接相关。
