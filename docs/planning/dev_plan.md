# TextMamba3D v2 — Development Plan

> Phase D output. 基于 architecture_proposal.md 拆解为可执行的实现任务。

## Implementation Tasks

### Task 1: Fix LR Scheduler (P0)
**File:** `train.py`
**Changes:**
1. 删除 SequentialLR + LinearLR + CosineAnnealingLR 相关代码 (lines 334-347)
2. 新增 `get_lr(epoch, warmup_epochs, base_lr, total_epochs)` 函数（手动 warmup + cosine decay）
3. 在训练循环中用 `optimizer.param_groups[i]['lr'] = get_lr(...)` 替代 `scheduler.step()`
4. 移除 checkpoint 中 scheduler state_dict 的保存/加载（不再需要）
5. 更新 TensorBoard LR 日志为 `get_lr(epoch, ...)` 的返回值
6. 在 `optimizer.step()` 前添加 `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.gradient_clip_norm)`

**测试:** 打印前 20 epoch 的 LR 值，验证 warmup 从 0→base_lr 线性递增，之后 cosine 递减。

### Task 2: Shrink Model (P1)
**File:** `configs/textbrats.yaml`
**Changes:**
1. `embed_dim: 96` → `embed_dim: 48`
2. `weight_decay: 0.00001` → `weight_decay: 0.01`
3. `epochs: 150` → `epochs: 300`
4. `warmup_epochs: 5` → `warmup_epochs: 10`
5. `patience: 30` → `patience: 50`
6. 新增 `gradient_clip_norm: 1.0`

**验证:** 模型构建后打印 trainable/total 参数量，确认 trainable < 20M。

### Task 3: Verify Architecture Compatibility (P1)
**Files:** `models/textmamba3d.py`, `models/decoder_3d.py`, `models/fusion.py`
**Changes:** 无代码修改，仅验证 embed_dim=48 下各模块的维度传递正确。
**验证:** `python -c "from models import TextMamba3D; m = TextMamba3D(embed_dim=48, ...); print('OK')"`

### Task 4: Update Training Stability (P2)
**File:** `train.py`
**Changes:**
1. 从 config 读取 `gradient_clip_norm`
2. 添加 NaN loss 检测：单 batch loss 为 NaN 时 skip（不更新参数），而非终止训练
3. 添加 gradient norm 日志到 TensorBoard（用于监控）

### Task 5: Clean Up Old Checkpoints
**操作:** 删除旧的不兼容 checkpoint（best.pth, best_no_text.pth, last.pth），清空 logs/
**注意:** 需要用户确认

## Dependencies

```
Task 1 (scheduler fix) ──┐
Task 2 (config update) ──┼──→ Task 3 (verify compat) ──→ Task 5 (cleanup) ──→ 训练
Task 4 (stability)    ────┘
```

Task 1, 2, 4 可并行执行。Task 3 依赖 Task 2。Task 5 最后执行。

## Crossfire 执行策略

使用 `crossfire` Actor-Critic pipeline:
- **Actor (Codex):** 执行 Task 1-4 的代码修改
- **Critic (Claude):** 审查每个 Task 的修改是否符合 architecture_proposal.md

## 训练命令（实现完成后）

```bash
cd /mnt/e/VSCode_Project/BS6207/TextMamba3D
source .venv/bin/activate
# 从头训练（不 resume，因为 checkpoint 不兼容）
python3 train.py --config configs/textbrats.yaml 2>&1 | tee training_log_v2.txt
```
