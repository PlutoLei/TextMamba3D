# Back-Modality: Leveraging Modal Transformation for Data Augmentation

**Back-Modality：利用模态变换进行数据增强**

| Item | Detail |
|------|--------|
| Authors | Zhi Li, Yifan Liu, Yin Zhang (Zhejiang University) |
| Venue | NeurIPS 2023 (37th Conference on Neural Information Processing Systems) |
| Paper Type | Methodological |
| Code | [github.com/zhilizju/Back-Modality](https://github.com/zhilizju/Back-Modality) |

## Quick Summary / 快速摘要

Back-Modality is a modality-agnostic data augmentation framework based on round-trip cross-modal transformation. Data in an initial modality A is converted to an intermediate modality B via a pretrained cross-modal model F, augmented in B using augmentation operator H, and converted back to A via model G, yielding augmented data X_aug = G(H(F(X))). Three instantiations are proposed: back-captioning (image->text->image), back-imagination (text->image->text), and back-speech (text->speech->text).

Back-Modality 是一种与模态无关的数据增强框架，基于跨模态往返变换。初始模态 A 的数据通过预训练跨模态模型 F 转换到中间模态 B，在 B 中施加增强算子 H，再通过模型 G 转换回 A，得到增强数据 X_aug = G(H(F(X)))。论文提出三种实例化方案：back-captioning（图像->文本->图像）、back-imagination（文本->图像->文本）和 back-speech（文本->语音->文本）。

## Key Contributions / 核心贡献

1. **Back-Modality Framework:** A universal, modality-agnostic data augmentation framework formulated as X_aug = G(H(F(X))), enabling cross-modal transfer of augmentation techniques.
   - **Back-Modality 框架：** 通用的模态无关数据增强框架，公式化为 X_aug = G(H(F(X)))，实现增强技术的跨模态迁移。

2. **Cross-Modality of Augmentation Methods:** Demonstrates that augmentation techniques designed for one modality (e.g., text augmentation via GPT) can be leveraged to augment data in another modality (e.g., images) through the round-trip pipeline.
   - **增强方法的跨模态迁移：** 证明了为一种模态设计的增强技术（如 GPT 文本增强）可以通过往返管线用于增强另一种模态的数据（如图像）。

3. **Three Instantiations:** Back-captioning (OFA + GPT-3.5 + Stable Diffusion v2), back-imagination (Stable Diffusion v2 + OFA), and back-speech (FastSpeech2 + pitch shifting/time stretching + wav2vec2), each validated on a different task.
   - **三种实例化方案：** Back-captioning、back-imagination 和 back-speech，分别在不同任务上验证。

## Key Results / 关键结果

| Task | Dataset | Method | Best Accuracy | vs. Next-Best |
|------|---------|--------|--------------|---------------|
| Image Classification | Tiny ImageNet (10-shot) | Back-captioning | 20.07% | +4.41 vs. Puzzle Mix (15.66%) |
| Textual Entailment | TNCC (10-shot) | Back-imagination | 89.14% | +1.73 vs. Treemix (87.41%) |
| Sentiment Classification | SST-2 (10-shot) | Back-speech | 63.21% | +0.84 vs. Treemix (62.37%) |

## Files / 文件

| File | Description |
|------|-------------|
| `Back-Modality_NeurIPS2023.pdf` | Original paper / 原始论文 |
| `Back-Modality_analysis_en.md` | Standard-depth analysis (English) / 标准深度分析（英文） |
| `Back-Modality_analysis_cn.md` | Standard-depth analysis (Chinese) / 标准深度分析（中文） |

## Relevance to TextMamba3D / 与 TextMamba3D 的关联

Back-Modality is relevant to TextMamba3D as a potential data augmentation strategy for addressing the chronic data scarcity in 3D medical image segmentation. Key takeaways:

- The round-trip framework (image -> text -> image) could be adapted to 3D medical volumes: volume -> radiology report -> synthetic volume, using report generation and text-conditioned 3D generation models.
- The forward pass (F: image -> text) naturally produces text descriptions that could serve as weak supervision signals for text-guided segmentation, bridging data augmentation and multimodal learning.
- The finding that cross-modal round-trip transformation generates data with higher diversity (1.723 vs. 1.621 for Random Erasing on Tiny ImageNet) while preserving label semantics (99.2% label invariance) suggests this paradigm could produce more effective training data than traditional medical image augmentation.
- The CMMaaS design (no fine-tuning needed) is practical for medical imaging where annotated data is scarce and model training is expensive.

Back-Modality 作为一种潜在的数据增强策略，与 TextMamba3D 的关联在于缓解三维医学图像分割中长期存在的数据稀缺问题：

- 往返框架（图像->文本->图像）可适配到三维医学体积数据：体积->放射学报告->合成体积，利用报告生成和文本条件三维生成模型。
- 前向变换（F: 图像->文本）天然产生文本描述，可作为文本引导分割的弱监督信号，在数据增强与多模态学习之间架起桥梁。
- 跨模态往返变换生成更高多样性数据（Tiny ImageNet 上 1.723 vs. Random Erasing 1.621）且保持标签语义（99.2% 标签不变性），表明该范式可能比传统医学图像增强产生更有效的训练数据。
- CMMaaS 设计（无需微调）在标注数据稀缺、模型训练昂贵的医学影像场景中具有实际价值。
