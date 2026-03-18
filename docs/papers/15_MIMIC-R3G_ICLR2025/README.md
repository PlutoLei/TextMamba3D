# 14. MIMIC-R3G | ICLR 2025 (under review)

## Paper Info / 论文信息

| Field | Content |
|-------|---------|
| **Title** | Benchmark Dataset for Radiology Report Generation with Instructions and Contexts |
| **Authors** | Anonymous (double-blind review) |
| **Venue** | ICLR 2025 (under review) |
| **Paper Type** | Methodological (Dataset + Model) |
| **Pages** | 30 |

## Quick Summary / 快速摘要

**EN:** MIMIC-R3G is the first benchmark dataset for real-world radiology report generation that incorporates clinical instructions and contextual information. Built via a GPT-4-based automatic pipeline and validated by 5 radiologists (95.5% acceptance, 9.58/10 plausibility), it covers 5 sub-tasks: no-context generation, report revision, template-based generation, previous-visit-as-context, and medical-records-as-context. The accompanying model DeMMo (Flamingo + BioViL + pathological guidance) achieves the best average F1 of 0.608 across all tasks.

**CN:** MIMIC-R3G 是首个引入临床指令和上下文信息的真实放射学报告生成基准数据集。通过 GPT-4 自动管线构建并经 5 位放射科医生验证（95.5% 接受率，9.58/10 可信度），涵盖 5 个子任务：无上下文生成、报告修订、模板化生成、基于既往就诊的生成、基于病历的生成。配套模型 DeMMo（Flamingo + BioViL + 病理引导）在所有任务上取得最佳平均 F1 0.608。

## Files / 文件列表

| File | Description |
|------|-------------|
| `MIMIC-R3G_ICLR2025.pdf` | Original paper PDF / 原始论文 PDF |
| `MIMIC-R3G_analysis_en.md` | English analysis (Standard depth) / 英文分析（标准深度） |
| `MIMIC-R3G_analysis_cn.md` | Chinese analysis (Standard depth) / 中文分析（标准深度） |

## Key Contributions / 核心贡献

1. **MIMIC-R3G dataset** -- First R3G benchmark with 5 clinically realistic sub-tasks under unified (V, I, C, R') instruction-following format; 95.5% radiologist acceptance.
2. **GPT-4 data generation pipeline** -- Automatic construction of instructions, contexts, and modified reports with multi-stage quality control including CheXpert-based leakage detection.
3. **DeMMo model** -- Flamingo-based architecture with dual vision encoders (general + BioViL medical), zero-initialized gated fusion, and pathological guidance; top average F1 = 0.608.

## Verdict / 结论

**Worth Deep Reading? Yes** -- Establishes a new benchmark for context-aware radiology report generation; both dataset design and the DeMMo architecture are directly applicable to medical AI research.

**是否值得深读？是** -- 为上下文感知的放射学报告生成建立新基准；数据集设计和 DeMMo 架构均可直接应用于医学 AI 研究。
