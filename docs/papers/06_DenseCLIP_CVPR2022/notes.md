# DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting

> CVPR 2022 | Rao et al.
> arXiv: https://arxiv.org/abs/2112.01518

## 一句话总结

将 CLIP 的 image-text matching 转化为 **pixel-text matching**，通过 context-aware prompting 实现语言引导的密集预测（分割/检测/深度估计）。

## 核心贡献

1. 将 CLIP 的全局对比学习转化为像素级文本匹配，用于 dense prediction
2. 提出 context-aware prompting：用视觉上下文动态调整文本 prompt embedding
3. 无需额外标注，利用 CLIP 预训练知识迁移到 dense prediction

## 核心机制

```
Image → CLIP visual encoder → pixel features [H×W, C]
Text  → CLIP text encoder → text features [K, C]

# Pixel-Text Score Map
score_map[i,k] = pixel_features[i] · text_features[k]  # [H×W, K]

# Context-Aware Prompting
visual_context = global_pool(pixel_features)
prompted_text = text_features + MLP(visual_context)  # 视觉反馈修正文本
```

## 关键实验结果

| 任务 | 数据集 | DenseCLIP vs baseline |
|------|--------|----------------------|
| 语义分割 | ADE20K | +1.7 mIoU |
| 目标检测 | COCO | +0.7 AP |
| 深度估计 | NYUv2 | 改善 |

## 与 TextMamba3D 的关系

- **验证 pixel-text 对齐方向正确**：DenseCLIP 证明了逐像素的文本引导在 dense prediction 中有效
- V4.3 的 TextToVoxelLoss 是类似思路的 3D 版本：文本特征 → 动态卷积核 → 逐体素预测
- **区别**：DenseCLIP 用于自然图像分类语义；TextMamba3D 用于医学影像中特定病灶区域

## 待深读要点

- [ ] Context-aware prompting 的具体实现细节
- [ ] 与 CRIS 的 text-to-pixel contrastive 的本质区别
- [ ] 是否可以将 prompting 思路用于 TextMamba3D 的文本编码器
