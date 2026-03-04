# TextMamba3D v2 — Architecture Proposal

> Phase D output. 基于 research_summary.md 的发现，转化为具体的架构和训练配置变更。

## 变更总览

```
优先级排序：
P0 [紧急] LR scheduler 修复 + gradient clipping   ← NaN 根因
P1 [高]   embed_dim 96→48 参数缩减                ← 过拟合根因
P2 [中]   训练超参数调优                           ← 收敛质量
P3 [低]   可选的额外稳定性措施                      ← 防御性
```

## P0: LR Scheduler 修复

### 问题

```python
# 当前代码 (train.py:337-347) — SequentialLR resume 有 bug
warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=5)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=145)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
# load_state_dict 后子 scheduler 内部计数器不同步 → LR 跳到 0.002-0.004
```

### 修复方案

**方案 A（推荐）：手动 warmup + 纯 CosineAnnealingLR**

```python
# 废弃 SequentialLR，在 train_epoch 中手动实现 warmup
def get_lr(epoch, warmup_epochs, base_lr, total_epochs):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    # cosine decay
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

# 每 epoch 设置 LR
for param_group in optimizer.param_groups:
    param_group['lr'] = get_lr(epoch, warmup_epochs, base_lr, total_epochs)
```

优点：resume 完全由 epoch 计数决定，不依赖 scheduler state dict。

### Gradient Clipping

```python
# 在 optimizer.step() 前添加
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## P1: 模型缩减

### 配置变更

```yaml
# configs/textbrats.yaml
model:
  embed_dim: 48          # 96 → 48
  depths: [2, 2, 2, 2]   # 保持不变
  # 其余不变
```

### 预估参数量

| 组件 | embed_dim=96 | embed_dim=48 |
|------|-------------|-------------|
| Stem (Conv3d) | ~0.1M | ~0.03M |
| Encoder (4 stages, CrossScanBiMamba3D) | ~45M | ~12M |
| → 每 stage: 6 SSM + merge + dwconv + uncertainty | — | — |
| Decoder (Conv3d + aux_heads) | ~10M | ~3M |
| FiLM Fusion | ~2M | ~0.5M |
| PubMedBERT (frozen) | 14M | 14M |
| PubMedBERT (unfrozen 2 layers) | +14M | +14M |
| **Total (estimated)** | **~70M** | **~30M** |
| **Trainable (BERT frozen)** | **~56M** | **~16M** |

注意：embed_dim=48 + frozen BERT 估计 ~16M trainable，294 样本给出 ~54K:1 比例，接近 SegMamba 水平。

### 需要验证

实际参数量需要在代码修改后用以下方式验证：
```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable/1e6:.1f}M, Total: {total/1e6:.1f}M")
```

## P2: 训练超参数

### 新 config

```yaml
training:
  epochs: 300             # 150 → 300 (小模型需要更多 epoch 收敛)
  lr: 0.0001              # 保持 1e-4（修复 scheduler 后应该稳定）
  weight_decay: 0.01      # 1e-5 → 1e-2 (增强正则化，对抗过拟合)
  warmup_epochs: 10       # 5 → 10 (更平滑的 warmup)
  patience: 50            # 30 → 50 (配合更长训练)
  gradient_accumulation: 4
  gradient_checkpointing: true
  deep_supervision: true
  no_text_ratio: 0.15
  gradient_clip_norm: 1.0  # 新增

  # 精度选择：优先 bfloat16，NaN 仍发生则回退 FP32
  amp_dtype: "bfloat16"
```

### weight_decay 调整理由

当前 weight_decay=1e-5 几乎无正则化效果。对于过参数化的 Mamba 模型：
- AdamW 的标准 weight_decay 在 0.01-0.1 范围
- SegMamba 用 SGD + decay=1e-5，但 SGD 的 weight_decay 机制不同（L2 vs decoupled）
- 增到 0.01 是 AdamW 的标准起点

## P3: 可选额外措施

仅在 P0-P2 修复后仍有问题时启用：

| 措施 | 实现难度 | 预期收益 |
|------|---------|---------|
| FP32 训练（禁用 AMP） | 低 | 彻底消除精度相关 NaN |
| Stochastic Layer-Wise Shuffle | 中 | 正则化防止 Mamba 过拟合 |
| SGD + polynomial decay | 中 | 参考 SegMamba 的验证配置 |
| Freeze 更多 BERT layers | 低 | 减少 trainable params |

## 不兼容性说明

**embed_dim 96→48 使所有现有 checkpoint 不兼容。** 必须从头训练。

这不是额外的代价——当前所有 checkpoint 都已因 NaN 崩溃而失效，需要重新训练。

## 文件变更清单

| 文件 | 变更 | 优先级 |
|------|------|--------|
| `train.py` | 废弃 SequentialLR → 手动 warmup；添加 gradient clipping | P0 |
| `configs/textbrats.yaml` | embed_dim, weight_decay, epochs, warmup, gradient_clip | P0+P1+P2 |
| `models/textmamba3d.py` | 验证 embed_dim=48 兼容性（应自动适配） | P1 |
| `models/decoder_3d.py` | 验证 aux_heads 与新 embed_dim 的兼容性 | P1 |
