# V7.0: Boundary Loss + Hierarchy Loss 设计

## 目标

在 V5.0 (Mean Dice 0.8479) 基础上，通过 Boundary Loss（边界距离变换）和 Hierarchy Loss（ET⊆TC⊆WT 软约束）提升分割质量，特别是 ET 边界。

## Loss 架构

```
Total = Dice + CE + Edge + 0.05*Contrastive + α(epoch)*Boundary + 0.1*Hierarchy
```

- **Boundary Loss**: softmax × signed distance transform (Kervadec MIDL 2019)
- **Hierarchy Loss**: max(0, p_ET - p_TC) + max(0, p_TC - p_WT) (STPF 提取)
- **α 线性 annealing**: α = (epoch - 110) / 80, 从 0 增到 1.0
- Hierarchy Loss 固定权重 0.1，不需要 annealing

## 数据流

1. 训练前预计算：GT mask → scipy EDT → distance_map.npy (每 case 一个)
2. __getitem__: 加载 distance_map.npy，和 image/mask 同步 crop 128³
3. train_epoch: batch 包含 distance_map，传给 CombinedLoss
4. CombinedLoss.forward 接收 distance_map 和 epoch 参数

## 训练策略

从 V5.0 resume，reset optimizer + lr：
- lr: 1e-4 (2x V5.0)
- epochs: 190 (从 110 跑到 190，实际 80 epoch)
- patience: 40
- reset_optimizer: true
- reset_lr: true

## 评估

只跑 2 个配置：text+TTA, notext+TTA。不跑 PP。

## 文件改动

| 文件 | 改动 |
|------|------|
| losses/boundary_loss.py | 已创建 (69 行) |
| losses/hierarchy_loss.py | 已创建 (40 行) |
| losses/__init__.py | CombinedLoss 加 boundary + hierarchy 支持 |
| data/brats_textbrats_dataset.py | __getitem__ 加载 distance_map |
| train.py | 传递 distance_map 和 epoch 给 criterion |
| scripts/precompute_distance_maps.py | 新建，预计算脚本 |
| configs/autoresearch/V7.0_boundary_hierarchy.yaml | 新建 |
| TextMamba3D_V7.0.ipynb | 新建 notebook |
