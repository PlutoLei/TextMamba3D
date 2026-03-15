# MedSAM: Segment Anything in Medical Images

> Nature Communications | Ma et al.
> arXiv: https://arxiv.org/abs/2304.12306
> GitHub: https://github.com/bowang-lab/MedSAM

## 一句话总结

基于 SAM 的通用医学图像分割基础模型，在大规模多模态医学数据上训练，实现跨模态跨任务的 prompt-based 分割。

## 核心贡献

1. **大规模多模态预训练**：涵盖 CT、MRI、X-ray、超声、内镜等 10+ 模态
2. **Prompt-based 分割**：bbox/point prompt，无需文本
3. **广泛的外部验证**：多个独立测试集验证泛化能力

## 核心机制

- SAM (Segment Anything Model) 的医学适配版
- 使用 bounding box 或点 prompt 而非文本描述
- ViT-B/H encoder + mask decoder

## 与 TextMamba3D 的关系

- **不是直接竞品**：MedSAM 用空间 prompt（bbox/point），不用文本引导
- **对比定位**：TextMamba3D 的创新点在于用**临床文本报告**替代人工空间标注
- **论文写作参考**：在 Related Work 中讨论 prompt-based vs text-guided 的区别
- MedSAM 是通用模型，TextMamba3D 是 task-specific（脑肿瘤分割）

## 待深读要点

- [ ] MedSAM 在 BraTS 上的表现（如有）
- [ ] Prompt 类型对分割精度的影响
- [ ] 与 text-guided 方法的互补可能性
