# E2ENet: Dynamic Sparse Feature Fusion for Accurate and Efficient 3D Medical Image Segmentation

> NeurIPS 2024 |
> OpenReview review score: 0.56

## 一句话总结

通过动态稀疏融合机制减少特征冗余，结合受限深度偏移卷积，在 3D 医学分割中平衡精度与效率。

## 核心贡献

1. **Dynamic Sparse Feature Fusion**：自适应选择重要特征通道，减少冗余融合
2. **Restricted Depth-Shift Convolution**：轻量级 3D 卷积替代方案，减少参数量
3. **Multi-challenge Cross-validation**：跨多个医学分割挑战赛验证泛化性

## 核心机制

- 动态稀疏融合：学习每个特征通道的重要性权重，剪枝不重要的通道
- 深度偏移卷积：沿 depth 轴偏移特征后做 2D 卷积，近似 3D 卷积但参数更少
- 端到端训练，无需额外的修剪/蒸馏步骤

## 与 TextMamba3D 的关系

- **效率参考**：TextMamba3D (20M params) 已经比较轻量，但 E2ENet 的稀疏融合思路可以进一步参考
- **不是直接竞品**：E2ENet 不涉及文本引导
- **论文引用**：efficient 3D segmentation 的 NeurIPS 2024 代表作

## 待深读要点

- [ ] 具体在哪些医学分割数据集上测试
- [ ] 参数量和计算量对比
- [ ] 动态稀疏融合的实现细节
