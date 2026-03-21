# Mamba-3 Integration Lessons Learned

TextMamba3D V5.1 将真正的 Mamba-3（ICLR 2026）集成到 3D 医学图像分割 pipeline 中的工程经验总结。

---

## 1. 根因定位过程

### 1.1 问题表象

V5.1 training smoke test 在 Colab A100 上反复 crash，历经多轮修复均未解决。错误类型包括：

- `RuntimeError: mat1 and mat2 must have the same dtype`
- `RuntimeError: GET was unable to find an engine to execute this computation`
- `RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered`

### 1.2 误导性诊断

| 阶段 | 尝试的修复 | 结果 | 误判原因 |
|------|----------|------|---------|
| 1 | text_encoder proj dtype cast | 修复了一个 bug 但不是根因 | 只看到第一个报错就以为找到根因 |
| 2 | cross_entropy class_weights fp32 | 修复了一个 bug 但不是根因 | 同上 |
| 3 | pure bf16 → autocast | 引入新 crash | 不理解 Mamba3 缺少 @custom_fwd |
| 4 | autocast(enabled=False) + SSM bf16 cast | 引入新 crash | SSM 权重还是 fp32 |
| 5 | prepare_for_amp() 只 cast SSM | 引入新 crash | autocast 仍然干扰 Triton kernel |
| 6 | 全部撤销，回到标准 autocast | crash | Mamba3 没有 @custom_fwd，autocast 不兼容 |
| 7 | 回到 pure bf16 + loss 全 fp32 | crash | 根因不在 loss |
| 8 | 禁用 deep supervision | crash | 根因不在 deep supervision |
| 9 | batch_size=1 | 错误变了但仍 crash | batch 不是根因 |

### 1.3 真正的根因定位

**方法：隔离测试 + 逐维度二分搜索**

```python
# Step 1: d_state 隔离 → 全部通过，排除 d_state
for d_state in [16, 32, 64, 128]:
    Mamba3(d_model=96, d_state=d_state).backward()  # ALL OK

# Step 2: seq_len + dim 隔离 → dim=48 始终失败
tests = [
    (384, 64),   # Stage 3 → OK
    (192, 512),  # Stage 2 → OK
    (96, 4096),  # Stage 1 → OK
    (48, 4096),  # Stage 0 → FAILED
    (48, 32768), # Stage 0 → FAILED
]
```

**根因：Mamba-3 Triton backward kernel `mamba3_siso_bwd_kernel_dzdo` 在 d_model<96 时存在 bug。**

具体来说，`d_model=48, expand=2 → d_inner=96, headdim=48, nheads=2`，nheads=2 触发了 Triton kernel 的非法内存访问。

---

## 2. 关键技术发现

### 2.1 Mamba-3 vs Mamba-2 的 AMP 兼容性差异

| 特性 | Mamba-2 | Mamba-3 |
|------|---------|---------|
| `@custom_fwd` 装饰器 | ✅ 有 | ❌ 没有 |
| `torch.is_autocast_enabled()` 检查 | ✅ 有 | ❌ 没有 |
| PyTorch AMP autocast 兼容 | ✅ 完全兼容 | ❌ backward crash |
| 推荐训练精度 | fp32 参数 + autocast | pure bf16（论文 Appendix D） |

**关键引用：** Mamba-3 论文 Appendix D: "All models at each scale follow the same procedure and were trained with bfloat16."

### 2.2 Pure bf16 训练的 Loss 兼容性

模型在 bf16 下前向和反向传播正常，但 loss 计算中某些操作缺少 bf16 backward kernel：

| 操作 | bf16 backward | 解决方案 |
|------|--------------|---------|
| F.cross_entropy | ✅ | - |
| F.softmax | ✅ | - |
| F.interpolate (trilinear) | ⚠️ 需验证 | 在 interpolate 前 cast 到 fp32 |
| torch.sqrt (EdgeLoss Sobel) | ⚠️ 数值不稳定 | loss 入口统一 fp32 |
| F.normalize | ⚠️ 低精度下不稳定 | loss 入口统一 fp32 |

**最终方案：** 在 `CombinedLoss.forward()` 入口统一把所有 float 输入 cast 到 fp32：

```python
pred = pred.float()
if img_feat is not None: img_feat = img_feat.float()
if text_feat is not None: text_feat = text_feat.float()
if pixel_feat is not None: pixel_feat = pixel_feat.float()
if aux_preds is not None: aux_preds = [a.float() for a in aux_preds]
```

PyTorch autograd 自动处理 fp32→bf16 的梯度回传。

### 2.3 Test Notebook 的局限性

Test Notebook Cell 8 通过了 full model forward+backward，但它的 loss 是 `out[0].sum()`——完全绕过了 `CombinedLoss`。这意味着以下路径从未被测试：

- DiceLoss（softmax + stack + normalize）
- EdgeLoss（Sobel conv + sqrt）
- ContrastiveLoss（adaptive_pool + normalize + matmul）
- Deep supervision（trilinear interpolation）
- return_features=True 的额外 backward 路径

**教训：验证测试必须覆盖实际训练路径，不能用简化的 loss 代替。**

---

## 3. 最终架构方案

### 3.1 混合 SSM 策略

```
Stage 0 (dim=48):  Mamba-2  ← 自动 fallback（Mamba-3 Triton kernel bug）
Stage 1 (dim=96):  Mamba-3  ← 真正的 complex-valued SSM
Stage 2 (dim=192): Mamba-3
Stage 3 (dim=384): Mamba-3
```

实现在 `_create_ssm()` 中，dim < 96 时自动 fallback 到 Mamba-2 并打印 warning。

### 3.2 精度配置

```yaml
training:
  use_amp: false        # Mamba-3 无 @custom_fwd，不兼容 autocast
  bf16_mode: "pure"     # 整个模型 bf16（论文方案），BERT 保持 fp32
```

### 3.3 已知限制

- Mamba-3 Triton backward 不支持 d_model<96（nheads<3）
- 需要从 GitHub source 安装（PyPI 只有 Mamba-2）
- `mamba3_step_fn` 需要 stub（依赖 CUTLASS DSL，训练不需要）
- gradient_checkpointing 不兼容（需禁用）

---

## 4. 工程方法论

### 4.1 RCA（Root Cause Analysis）最佳实践

1. **不要假设第一个报错就是根因** — 我们连续修了 9 个"根因"，实际根因在第 10 轮才找到
2. **用隔离测试缩小范围** — 逐个变量（d_state → seq_len → dim）排除
3. **对比已知可工作的配置** — Test Notebook 的精确参数是黄金标准
4. **CUDA_LAUNCH_BLOCKING=1** — 获取精确的 CUDA 错误位置，而非异步报告的模糊 traceback

### 4.2 混合精度工程原则

1. **先查论文怎么训的** — Mamba-3 论文明确写了 "trained with bfloat16"，不是 AMP
2. **检查 @custom_fwd 装饰器** — 没有这个装饰器的自定义 autograd function 不兼容 autocast
3. **Loss 始终在 fp32** — bf16 的 loss backward 有太多坑，统一 fp32 最安全
4. **验证测试必须覆盖实际训练路径** — 不能用 `.sum()` 代替 `CombinedLoss`

### 4.3 避免的反模式

- ❌ 遇到 dtype 报错就加 `.float()` / `.to(bf16)` — 治标不治本
- ❌ 在不理解 Triton kernel 兼容性的情况下切换 autocast / pure bf16
- ❌ 用 `torch.autocast(enabled=False)` 包裹 SSM 调用 — 如果 SSM 权重还是 fp32 就会 crash
- ❌ 修了多轮后不回头审视是否在正确方向上 — 应该更早做隔离测试

---

## 5. 文件清单

| 文件 | 关键改动 |
|------|---------|
| `models/mamba_block.py` | `_create_ssm()` dim<96 自动 fallback 到 Mamba-2 |
| `models/text_encoder.py` | BERT fp32 → proj bf16 边界 cast |
| `models/textmamba3d.py` | `to_bf16_with_fp32_text()` 保留 BERT fp32 |
| `losses/__init__.py` | 入口统一 fp32 cast + EdgeLoss Sobel buffer fp32 |
| `losses/edge_loss.py` | Sobel kernel 显式 `.float()` |
| `configs/textbrats_a100_v5.1.yaml` | use_amp=false, bf16_mode=pure, d_state=64 |
| `utils/precision.py` | get_amp_context nullcontext for pure bf16 |

---

*记录于 2026-03-21。基于 TextMamba3D V5.1 Mamba-3 集成全过程。*
