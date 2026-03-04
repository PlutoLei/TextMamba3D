# TextMamba3D Optimization — Research Summary

> Phase R output for RDIV workflow. Sources: Idea2Paper KG (56K papers), Mamba GitHub issues, SegMamba/VM-UNet papers.

## 1. Current Problem Diagnosis

TextMamba3D 在 BraTS 294 样本上训练，反复在 ~Epoch 20 出现 NaN 崩溃。

### 1.1 Root Cause #1: LR Scheduler Bug (CRITICAL)

TensorBoard 记录显示 LR 在 resume 后从 0.002 跳到 0.004，而 config 的 base lr 仅为 1e-4（偏差 20-40x）。

**Bug 机制：** `SequentialLR(warmup + CosineAnnealing)` 的 `load_state_dict()` 在 resume 时不能正确恢复子 scheduler 的内部状态。CosineAnnealingLR 的 `last_epoch` 计数器与 SequentialLR 的 `_last_epoch` 不同步，导致 LR 计算错误。

**证据：** 第一轮训练（从头开始）LR 稳定在 1e-4；resume 后 LR 飙升。

### 1.2 Root Cause #2: Model Over-parameterization

| 指标 | TextMamba3D | SegMamba (MICCAI'24) | 推荐范围 |
|------|------------|---------------------|---------|
| Parameters | ~70M | ~23M | 5-25M |
| Training samples | 294 | 876 | — |
| Param/Sample ratio | 237K:1 | ~26:1 | 25-60K:1 |
| embed_dim | 96 | 48 | 32-48 |

6 方向 cross-scan BiMamba + multi-scale + uncertainty gating 使参数量急剧膨胀。

### 1.3 Root Cause #3: Mamba SSM Numerical Sensitivity

**来源：** [mamba issue #72](https://github.com/state-spaces/mamba/issues/72), [issue #529](https://github.com/state-spaces/mamba/issues/529)

- `selective_scan_fn` 输出值可达百万级别，bfloat16 虽然指数范围与 fp32 相同，但精度仅 8 bit（vs fp32 的 23 bit），累积误差仍可能导致 NaN
- Mamba2 的 `ngroups > 1` 会导致 layer_norm backward 计算错误 → gradient explosion
- **官方建议：** 首选 FP32 训练，或至少使用 gradient clipping

## 2. Literature Findings

### 2.1 Mamba Medical Segmentation Architectures (KG: 27 papers)

| 论文 | 关键技术 | 与 TextMamba3D 的关系 |
|------|---------|---------------------|
| **SegMamba** (MICCAI'24) | U-shape + depthwise 3D conv + tri-scan | TextMamba3D 已有 cross-scan 和 dwconv，但参数量是 SegMamba 的 3x |
| **VSSD-UNet** | Non-causal SSM + hybrid VSSD-Attention decoder | 提示可用 non-causal 替代 bidirectional |
| **Swin-UMamba** | ImageNet pretrained Mamba backbone | TextMamba3D 已用 PubMedBERT，无需额外 pretrain |
| **UD-Mamba** | Uncertainty-driven scanning | TextMamba3D 已有 uncertainty gating |
| **GroupMamba** | Channel grouping + distillation loss for stability | **可借鉴：** distillation loss 稳定训练 |
| **VMamba** | 2D SS2D 4-direction scan | TextMamba3D 的 6-direction 3D scan 是合理扩展 |

### 2.2 Training Stability Strategies (KG: 88 papers)

| 策略 | 来源 | 适用性 |
|------|------|--------|
| **SSM-specific scaling parameterization** | "On Feature Learning in SSMs" (ICLR) | 理论指导，但实现复杂 |
| **Stochastic Layer-Wise Shuffle** | ICLR paper | **推荐：** 专为 Vision Mamba 设计的正则化 |
| **Distillation-based training objective** | GroupMamba | 可选，需要 teacher 模型 |
| **Adaptive Scaling for robustness** | Adversarial Robustness of SSMs | 可选 |

### 2.3 Training Hyperparameters (Web Search)

| 模型 | Optimizer | LR | Scheduler | Epochs | Batch |
|------|-----------|-----|-----------|--------|-------|
| **SegMamba** | SGD | 1e-2 | Polynomial | 1000 | 2/GPU |
| **VM-UNet** | AdamW | 1e-3 | CosineAnnealing | 300 | 32 |
| **Mamba-UNet** | SGD | 1e-2 | — | 10K iter | 24 |
| **VMKLA-UNet** | AdamW | 1e-4 | CosineAnnealing | 300 | 32 |
| **TextMamba3D** | AdamW | 1e-4 | SequentialLR (broken) | 150 | 1 |

**关键发现：** 所有成功的 Mamba 分割模型都使用**单调递减**的 LR schedule（polynomial 或 cosine），没有使用 warmup + cosine 的组合。

## 3. Proposed Optimizations — Evidence-Based Assessment

### 3.1 Shrink Model (embed_dim 96→48)

- **强烈支持。** SegMamba 在 3x 多的数据上用 embed_dim=48 达到 SOTA
- **建议：** embed_dim=48, depths 保持 [2,2,2,2]，估计参数量 ~20M
- **风险：** 低。缩小后仍有 6-direction cross-scan 的架构优势

### 3.2 Fix LR Scheduler (最高优先级)

- **必须修复。** 这是 NaN 的直接诱因
- **方案：** 废弃 SequentialLR，改用 CosineAnnealingWarmRestarts 或手动 warmup
- **额外建议：** 添加 gradient clipping (max_norm=1.0)

### 3.3 Lower Learning Rate (1e-4 → ?)

- **需要分场景讨论：**
  - 修复 scheduler 后，1e-4 + CosineAnnealing 可能就够了（VMKLA-UNet 用同样配置成功）
  - 如果仍不稳定，降到 5e-5
  - SGD + lr=1e-2 + polynomial 也是文献支持的选项

### 3.4 Dataset Split (294→220 train)

- **条件性支持。** 仅在能获取原论文 exact sample IDs 时才有意义
- 如果只是比例相同，不如保持 294 train

## 4. References

- SegMamba: https://arxiv.org/abs/2401.13560
- Mamba NaN issue: https://github.com/state-spaces/mamba/issues/72
- Mamba gradient explosion: https://github.com/state-spaces/mamba/issues/529
- VM-UNet: https://arxiv.org/abs/2402.02491
- GroupMamba: KG paper (openreview_peer_review, q=0.50)
- On Feature Learning in SSMs: KG paper (openreview_peer_review, q=0.50)
- Stochastic Layer-Wise Shuffle: KG paper (openreview_peer_review, q=0.50)
