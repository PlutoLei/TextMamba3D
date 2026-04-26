# TextMamba3D Abstract Draft (2026-04-12 更新)

## 推荐版本：Phase Transition Story

**Title:** When Does Text Help? Discovering Modality Redundancy in Pretrained Mamba Models for Brain Tumor Segmentation

**Abstract:**

文本引导的医学图像分割利用放射学报告为深度学习模型提供辅助语义信息。现有方法如 TextBraTS（SwinUNETR + ImageNet 预训练）在 BraTS2020 上报告了 +1.5% 的文本增益。然而，我们发现了一个反直觉的"相变"现象：**同域监督预训练会完全消除辅助文本模态的贡献，且这一现象与融合机制无关。** 我们开发了 TextMamba3D，一个基于 Mamba2 的 3D 分割架构，在 BraTS2020 上系统地评估了 10+ 种文本融合策略。当从零训练（295 例）时，模型展现出预期的正向文本增益（+0.55% Mean Dice）。然而，在 BraTS2021（1251 例，同域）监督预训练后，绝对性能达到 0.8753 Mean Dice（超越 TextBraTS SOTA 的 0.853），但文本增益降至精确的 0.00%。我们穷尽性地测试了 9 种补救策略——包括 SeqCA、MambaConcatFusion、冻结-解冻训练、视觉模态丢弃、InfoNCE 对齐损失、三阶段冻结训练、特征噪声注入与解码器端 FiLM 条件化、以及 R-Super 启发的文本监督损失——所有方案均无法恢复文本增益。基于信息论分析（条件互信息 ≈ 0）和模态崩溃框架（ICML 2025），我们将此归因于：当预训练数据覆盖目标任务分布时，骨干网络已吸收了辅助模态所能提供的全部信息。进一步地，我们证明自监督预训练（掩码图像建模，无分割标签）可保留"知识缺口"，使文本增益恢复至 +0.77% Mean Dice（ET 区域 +1.25%）。我们的发现挑战了"更强骨干必然受益于多模态"的常见假设，并为该领域提供了一个"何时应投资多模态"的决策框架。

**Keywords:** 医学图像分割, Mamba, 文本引导, 多模态学习, 模态冗余, 自监督预训练, 脑瘤分割

---

## English Version

**Title:** When Does Text Help? Discovering Modality Redundancy in Pretrained Mamba Models for Brain Tumor Segmentation

**Abstract:**

Text-guided medical image segmentation leverages radiology reports as auxiliary semantic signals. While existing methods like TextBraTS (SwinUNETR + ImageNet pretraining) report +1.5% text delta on BraTS2020, we discover a counter-intuitive phase transition: **same-domain supervised pretraining completely eliminates auxiliary text utility, regardless of the fusion mechanism employed.** We develop TextMamba3D, a Mamba2-based 3D segmentation architecture, and systematically evaluate 10+ text fusion strategies on BraTS2020 with paired radiology reports. When trained from scratch on 295 cases, our model exhibits the expected positive text delta (+0.55% Mean Dice). However, after supervised pretraining on BraTS2021 (1,251 same-domain cases), absolute performance reaches 0.8753 Mean Dice (surpassing TextBraTS SOTA at 0.853), yet text delta drops to exactly 0.00%. We exhaustively test 9 remediation strategies — including Sequential Cross-Attention, MambaConcatFusion, frozen-backbone warmup, vision modality dropout, InfoNCE alignment, 3-stage freeze training, feature noise injection with decoder-side FiLM, and R-Super-inspired text supervision — all of which preserve the 0% text delta. Grounded in information-theoretic analysis (conditional mutual information ≈ 0) and the modality collapse framework (ICML 2025 Spotlight), we attribute this to the backbone absorbing all information the auxiliary modality could provide. We further demonstrate that self-supervised pretraining (masked image modeling, no segmentation labels) preserves a "knowledge gap" that restores text utility to +0.77% Mean Dice (+1.25% for the challenging enhancing tumor region). Our findings challenge the assumption that stronger backbones always benefit from additional modalities, and provide a principled framework for deciding when to invest in multimodal approaches.

**Keywords:** Medical image segmentation, Mamba, text guidance, multimodal learning, modality redundancy, self-supervised pretraining, brain tumor segmentation

---

## 数据汇总表 (论文 Table 1)

| Version | Fusion | Backbone | Mean Dice | Text Delta |
|---------|--------|----------|-----------|------------|
| V5.0 | SeqCA | From scratch (295) | 0.8479 | +0.55% |
| **V8.0** | SeqCA | **Sup. pretrain (1251)** | **0.8753** | 0.00% |
| V9.0 | SeqCA + freeze/dropout/align | Sup. pretrain | 0.8723 | -0.02% |
| V9.1 | MambaConcatFusion | Sup. pretrain | 0.8719 | -0.01% |
| V9.2a | MambaConcatFusion | From scratch | 0.8482 | +0.06% |
| V9.2b | 3-stage freeze | Sup. pretrain | 0.8714 | -0.01% |
| V10.0 | Feature noise + Decoder FiLM | Sup. pretrain | 0.8710 | -0.02% |
| V10.1 | R-Super text supervision | Sup. pretrain | 0.8690 | ~0.00% |
| **V10.2** | SeqCA | **SSL pretrain (200ep)** | **0.8024** | **+0.77%** |
| TextBraTS | Bi-directional CA | SwinUNETR + ImageNet | 0.853 | +1.5% |

**V10.2 per-region:**
| Region | text+TTA | notext+TTA | Delta |
|--------|----------|------------|-------|
| ET | 0.7243 | 0.7118 | **+1.25%** |
| TC | 0.8147 | 0.8131 | +0.16% |
| WT | 0.8683 | 0.8593 | +0.90% |
| Mean | 0.8024 | 0.7947 | +0.77% |
