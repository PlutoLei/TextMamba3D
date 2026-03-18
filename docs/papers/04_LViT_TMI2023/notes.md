# LViT: Language Meets Vision Transformer in Medical Image Segmentation

> IEEE Transactions on Medical Imaging (TMI) | Li et al.
> arXiv: https://arxiv.org/abs/2206.14718

## 一句话总结

首个将医学文本标注融入 Vision Transformer 进行医学图像分割的工作，通过 Language-Vision Loss 和指数伪标签迭代实现文本引导的半监督分割。

## 核心贡献

1. **Language-Vision Loss**：利用文本描述引导分割模型关注正确区域
2. **Exponential Pseudo Label Iteration**：基于文本的伪标签质量迭代提升
3. 证明文本标注在**小数据集**和**标注稀缺**场景下特别有效

## 核心机制

- 将医学文本编码为语义特征，与 ViT 的视觉特征做对齐
- 文本辅助生成更准确的 pseudo labels，用于半监督学习
- 指数迭代策略逐步提升伪标签质量

## 与 TextMamba3D 的关系

- **最直接的方法类竞品**：同为"文本 + 医学分割"
- **区别**：LViT 用 ViT backbone，TextMamba3D 用 Mamba SSM
- **区别**：LViT 侧重半监督/伪标签场景，TextMamba3D 是全监督 + 文本引导
- **引用价值**：证明文本信息对医学分割的有效性，尤其在标注有限时

## 待深读要点

- [ ] Language-Vision Loss 的具体公式和实现
- [ ] 与 TextMamba3D contrastive loss 的对比
- [ ] 指数伪标签迭代是否可以借鉴
- [ ] 实验数据集和具体提升幅度
