# TextMamba3D A100 40GB 全面优化方案 -- 头脑风暴

> 生成时间: 2026-03-07
> 基于代码审计: 实际 24M trainable params, best Dice 68.67% (no-text) / 68.48% (with-text), epoch ~86/300
> 目标: 在单张 A100 40GB 上最大化 Dice 性能, 缩小与 SOTA 85.3% 的差距

---

## 关键发现（代码审计后的修正）

在深入代码后，发现几个与之前认知不同的事实，对方案设计有重大影响：

| 项目 | 之前认知 | 代码实际 |
|------|---------|---------|
| Trainable params (embed_dim=48) | ~16M | **24M**（PubMedBERT 110M 全冻结，24M 为 encoder+decoder+fusion） |
| 扫描方向 | 6 方向 BiMamba | **3 个单向 SSM**（DHW/HWD/WDH 各一个，无反向扫描） |
| 文本引导效果 | 文本应优于无文本 | **无文本 (68.67%) > 有文本 (68.48%)**，文本引导轻微负面 |
| Loss clamp | 无 | **loss > 5.0 直接 skip 该 batch**，可能丢失难样本学习信号 |
| Uncertainty gating | 独立模块 | **嵌入每个 CrossScanBiMamba3DBlock 内部**，无法单独消融 |
| 多尺度扫描 | 已启用 | `use_multiscale = False`，**代码写死为关闭** |

这些发现直接影响下文的方案优先级。

---

## 方向 1: A100 硬件红利最大化

### 1A: Patch Size 提升至 128^3（推荐首选）

**方案:** 将 `patch_size` 从 64^3 提升至 128^3，使每个 patch 覆盖从 ~3% 提升至 ~25% 的全脑体积。

**VRAM 估算:**

| 配置 | Patch | Batch | AMP | GradCkpt | 预估 VRAM | 预估可行性 |
|------|-------|-------|-----|----------|----------|-----------|
| 当前 (4060) | 64^3 | 1 | off | on | ~6GB | 已验证 |
| A100 方案 A | 128^3 | 2 | bf16 | on | ~18-22GB | 大概率可行 |
| A100 方案 B | 128^3 | 4 | bf16 | on | ~32-36GB | 可能可行 |
| A100 方案 C | 96^3 | 4 | bf16 | on | ~14-18GB | 几乎确定可行 |

关键变量：128^3 的 token 序列长度 = (128/4)^3 = 32768 tokens（vs 64^3 的 4096 tokens），Mamba SSM 的内存随序列长度线性增长（不像 Transformer 的二次增长），所以 8x 的 token 数对 Mamba 是可控的。但 3D DWConv 和 Patch Merging 的中间激活值会显著增加。

**预期收益:**
- BraTS MRI 典型尺寸 240x240x155。64^3 patch 仅覆盖 3%，模型几乎看不到全局解剖结构。128^3 覆盖 ~25%，能看到肿瘤与周围组织的关系。
- 文献参考：SegMamba 用 128^3 达到 91.3%，SwinUNETR 用 128x128x128，nnU-Net 用 128x128x128。**128^3 是 BraTS 的事实标准。**
- 预期 Dice 提升: **+8-15%**（这可能是最大的单一改进因素）

**风险:**
- 若 128^3 bs=2 + bf16 + GradCkpt 仍 OOM，回退到 96^3 bs=4（仍然远优于 64^3 bs=1）
- 需要调整 data pipeline 的 crop/padding 策略

**推荐度: 5/5（必做）**

### 1B: Batch Size 提升 + bf16 AMP

**方案:** 在 A100 上启用 bf16 AMP 并增大 batch size。

A100 的 bf16 Tensor Core 吞吐量是 fp32 的 ~5x（312 TFLOPS vs 19.5 TFLOPS）。更关键的是，bf16 的动态范围与 fp32 相同（8-bit exponent），**不需要 GradScaler**（代码中已正确处理了这一点）。

| 配置 | 优势 | 注意事项 |
|------|------|---------|
| bf16 + bs=2 | 对比学习可用（需 bs>=2）；训练速度 ~3-4x | 最保守选择 |
| bf16 + bs=4 | 更强的对比学习；梯度估计更稳定 | 可能需要 96^3 而非 128^3 |
| bf16 + bs=2 + grad_accum=2 | 有效 bs=4 但 VRAM 仅 bs=2 | 对比学习仍仅在 bs=2 内计算 |

**关键决策:** batch size 的选择直接决定对比学习是否有效。当前 contrastive_weight=0.0 是因为 bs=1。在 A100 上 bs>=2 后，应重新启用。

**预期收益:**
- 训练速度提升 3-5x（从 ~2s/iter 降至 ~0.5s/iter）
- bs>1 使对比学习可用
- 更稳定的梯度（减少 NaN skip）

**风险:**
- 若 bf16 引入数值不稳定（Mamba SSM 的 S6 选择性扫描有复杂的数值路径），需在前 5 epoch 监控是否有精度退化
- LR 可能需要随 batch size 调整（线性缩放规则）

**推荐度: 5/5（必做）**

### 1C: 数据加载优化

**方案:** `num_workers=4`（A100 Colab 实例通常有 12 个 CPU 核心），`persistent_workers=True`，`prefetch_factor=2`。

**预期收益:** 消除 CPU 数据加载瓶颈。当前 num_workers=0 是 Windows 限制，Colab Linux 无此问题。可能使端到端训练速度再提升 20-30%。

**风险:** 几乎无风险。Colab 的 `/dev/shm` 可能较小，必要时设置 `multiprocessing_context='fork'`。

**推荐度: 4/5（应做）**

---

## 方向 2: 模型架构优化

### 2A: embed_dim=48 vs 64 的抉择

**方案:** 将 embed_dim 从 48 恢复到 64。

**参数量对比:**

| embed_dim | Encoder+Decoder trainable | 总 trainable | 参数/样本比 (N=220) |
|-----------|--------------------------|-------------|-------------------|
| 48 | ~24M | ~24M | 109K:1 |
| 64 | ~42M | ~42M | 191K:1 |
| 96 (原始) | ~70M (推测) | ~70M | 318K:1 |

**分析:**

48 维的 bottleneck dim = 48 * 2^3 = 384，而文本 embed_dim = 256。384:256 的比例合理。但问题在于每个 Mamba SSM 的 d_state=16，当 dim=48 时，SSM 的状态空间可能过于受限，无法充分表达 3D 空间依赖。

embed_dim=64 的 bottleneck = 512，参数增加 ~75% 但仍远小于原始 96 的配置。考虑到 A100 有充足 VRAM，**在 128^3 bs=2 能装下的前提下，64 优于 48**。

但更重要的是：参数/样本比 191K:1 仍然极高。在 220 个训练样本上，即使 42M 参数也面临严重过拟合风险。关键不在参数量本身，而在**正则化是否足够**。

**推荐决策:** 先用 embed_dim=48 + 128^3 bs=2 跑一轮，VRAM 有余量再考虑 64。**patch size 的收益远大于 embed_dim 的差异。**

**预期 Dice 变化:** +1-3%（与 patch size 的 +8-15% 相比，优先级较低）

**推荐度: 3/5（可选，视 VRAM 余量而定）**

### 2B: 架构创新消融优先级

代码审计后，对 6 项创新的价值判断如下：

| 创新 | 代码位置 | 消融难度 | 预估价值 | 理由 |
|------|---------|---------|---------|------|
| 1. 3D DWConv | `CrossScanBiMamba3DBlock.__init__` | 中 | **高** | 为 SSM 提供局部空间先验，UlikeMamba 已验证有效 |
| 2. 物理模态分组 | `ModalityGroupPatchEmbed3D` | 低 | **中** | CKD-TransBTS 验证有效，但增加了参数（2 个 Conv3d + merge Linear） |
| 3. 多尺度扫描 | `use_multiscale=False` | N/A | **零** | **代码写死关闭，从未实际使用** |
| 4. Pixel-Text CrossAttn | `fusion.py` 定义但**未在 forward 中调用** | N/A | **零** | `MultiScalePixelTextAttention` 已定义但 `textmamba3d.py` 的 forward 从未调用它 |
| 5. Uncertainty Gating | `CrossScanBiMamba3DBlock` 内部 | 中 | **低-中** | UD-Mamba 启发，但 zero-init 意味着早期训练等于 identity |
| 6. FiLM Bounded Gating + MultiScale FiLM | `fusion.py` + `textmamba3d.py` | 低 | **高** | 文本引导的核心机制，但当前"无文本>有文本"说明可能有实现问题 |

**重大发现: 创新 #3 和 #4 实际上从未被使用。** 多尺度扫描 `use_multiscale` 硬编码为 `False`；`PixelTextCrossAttention` 虽然定义了完整的模块，但 `TextMamba3D.forward()` 中根本没有调用它。这意味着论文/提案中声称的 6 项创新实际只有 4 项在工作。

**推荐消融实验设计（最小集，3 次训练）:**

| 实验 | 配置 | 验证什么 |
|------|------|---------|
| Baseline | embed_dim=48, 128^3, bs=2, bf16, 全部创新开启 | 新硬件下的基准 |
| -DWConv | 移除 CrossScanBiMamba3DBlock 中的 dwconv | DWConv 的贡献 |
| -FiLM (no text) | 移除 MultiScaleFiLM + MambaFusion, 纯视觉模型 | 文本引导的净贡献 |

**如果 -FiLM 实验结果等于或优于 Baseline**，则说明文本引导机制存在根本问题，需要 debug。

**推荐度: 5/5（消融实验必做，否则无法判断各组件价值）**

### 2C: 扫描方向与 depths 调整

**扫描方向:** 当前每个 CrossScanBiMamba3DBlock 有 3 个 SSM（DHW/HWD/WDH），每个是单向的。代码注释说是 "6-direction" 但实际是 3 个单向扫描（没有反向）。

方案选项:
1. **保持 3 方向（现状）:** 最轻量，可能已经够用
2. **真正实现 6 方向 BiMamba:** 每个轴加一个反向 SSM。参数翻倍，merge 从 3x→6x。VRAM 增加 ~50%。
3. **减少到 1 方向 + 反向（经典 BiMamba）:** 大幅减参数，但失去跨轴覆盖

**推荐:** 在 128^3 时 token 数已经是 32768，3 方向的计算量足够大。**先保持 3 方向**，如果 VRAM 有余量，可以尝试方案 2。

**depths [2,2,2,2]:** 4 个 stage 各 2 层。SegMamba 也是 [2,2,2,2]。这已经是标准配置，不建议修改。如果要减少计算，可以考虑 [2,2,2,1]（bottleneck 减一层），但收益不明显。

**推荐度: 2/5（低优先级，保持现状）**

---

## 方向 3: 训练策略

### 3A: 对比学习启用方案

**方案:** 在 bs>=2 时启用 contrastive loss。

**权重设置分析:**

当前 loss 组成（contrastive_weight=0 时）:
- `dice_weight=1.0` + `ce_weight=1.0` + `edge_weight=1.0` → 典型 total ~1.5-3.0
- 对比 loss 的量级：InfoNCE 在 bs=2 时 ~0.7（1 正例 1 负例），bs=4 时 ~1.4

**推荐权重:**

| Batch size | 推荐 contrastive_weight | 理由 |
|-----------|------------------------|------|
| 2 | 0.1 | bs=2 的对比学习信号极弱（仅 1 正 1 负），权重应低 |
| 4 | 0.3 | 更多负例，信号更有意义 |
| 8+ | 0.5 | 原论文推荐值，但 220 样本 + bs=8 每 epoch 仅 27 步 |

**但需要先解决"无文本>有文本"的问题。** 在文本引导本身有负面效果的情况下，加对比学习可能进一步恶化。建议先做 2B 中的 -FiLM 消融，确认文本引导方向正确后再启用对比学习。

**预期收益:** +1-3% Dice（前提是文本引导机制正常工作）

**风险:** 在 bs=2 时对比学习可能太弱，增加计算但无实质贡献

**推荐度: 3/5（条件性推荐，需先修复文本引导）**

### 3B: 学习率与训练超参

**当前:** lr=5e-5, warmup=10 epochs, cosine decay, weight_decay=0.05, 300 epochs

**A100 调整方案:**

| 参数 | 当前值 | 推荐值 | 理由 |
|------|--------|--------|------|
| lr | 5e-5 | **1e-4** (bs=2) 或 **2e-4** (bs=4) | 线性缩放：有效 bs 从 4 (bs=1 * accum=4) 变为 8 (bs=4 * accum=2) |
| warmup | 10 epochs | **15-20 epochs** | 更大 lr 需要更长 warmup |
| grad_accum | 4 | **1** (bs=4) 或 **2** (bs=2) | A100 上 batch 已够大，减少累积步数以加快更新频率 |
| weight_decay | 0.05 | **0.05** (保持) | 已经较高，适合小数据集 |
| epochs | 300 | **500** | A100 更快，可以跑更久；SegMamba 用 1000 epochs |
| patience | 50 | **80** | 配合更长训练 |
| loss clamp | 5.0 (skip) | **10.0 或移除** | 当前 skip > 5.0 可能丢失难样本的学习信号 |

**关于 loss clamp 的重要发现:**

代码中 `train_epoch` 里有两层保护：
1. `loss.item() * grad_accum > 5.0` → skip batch
2. `losses['total'].clamp(0.0, 5.0)` in CombinedLoss

这意味着**任何 total loss > 5.0 的 batch 都被直接跳过**。在训练初期或难样本上，这可能导致模型永远学不到困难案例。建议将阈值提高到 10.0 或 15.0，用 gradient clipping（已有 max_norm=1.0）来保护训练稳定性。

**预期收益:** +2-5% Dice（主要来自更长训练 + 不跳过难样本）

**推荐度: 4/5（应做）**

### 3C: Progressive Training（渐进式分辨率）

**方案:** 借鉴 EfficientSegMamba 的 64->96->128 渐进训练策略。

| Phase | Patch | Epochs | LR | 目的 |
|-------|-------|--------|-----|------|
| 1 | 64^3 | 100 | 1e-4 | 快速学习基本分割能力 |
| 2 | 96^3 | 150 | 5e-5 | 过渡到更大感受野 |
| 3 | 128^3 | 250 | 2e-5 | 精细化全局上下文 |

**优点:**
- Phase 1 训练速度极快（64^3 bs=8 完全可行），可以快速迭代超参
- 逐步增大分辨率，模型不会在大 patch 上 "迷失"
- 类似课程学习（curriculum learning）

**缺点:**
- 实现复杂度增加（需要每个 phase 结束后调整 DataLoader 和模型的 img_size）
- `PatchMerging3D` 和 `CrossScanBiMamba3DBlock` 的 `spatial_dims` 是在 `__init__` 时固定的，**改变 patch size 需要重建模型**。这意味着不能简单地 resume checkpoint——需要实现权重迁移。
- 模型中 `PatchMerging3D` 和 `PatchExpanding3D` 的 spatial_dims 是 hardcoded 的

**可行性评估:** 由于 spatial_dims 在模型初始化时固定（每个 stage 的 D/H/W 写死在 `__init__` 中），渐进训练需要：
1. 每个 phase 创建新模型实例
2. 手动迁移兼容的权重（SSM 参数、Linear 层与 spatial_dims 无关，可以迁移；PatchMerging/PatchExpanding 的 Linear 尺寸不变，也可以迁移）
3. `ModalityGroupPatchEmbed3D` 的 Conv3d kernel 是 patch_size，若 patch_size 从 (4,4,4) 不变则权重可迁移

**结论:** 技术上可行但需要额外 ~50 行的权重迁移代码。考虑到 A100 的算力，**直接用 128^3 从头训练可能更高效**（省去 Phase 1-2 的时间 + 权重迁移的 debug）。

**预期收益:** +2-4% Dice（相比直接 128^3 从头训练的边际收益）

**推荐度: 2/5（不推荐，直接 128^3 更简单高效）**

### 3D: 文本编码器解冻

**方案:** 解冻 PubMedBERT 最后 2-4 层，使其适应 BraTS 领域。

**当前状态:** `unfreeze_text_layers=0`，PubMedBERT 完全冻结。这意味着文本特征是通用生物医学语义，未针对脑肿瘤分割任务微调。

**参数量影响:**

| 解冻层数 | 新增 trainable params | 总 trainable |
|---------|---------------------|-------------|
| 0 (当前) | 0 | 24M |
| 2 | ~14M | ~38M |
| 4 | ~28M | ~52M |
| 全部 (12) | ~86M | ~110M |

**分析:**

在 220 个训练样本上解冻 BERT 层有严重过拟合风险。但考虑到当前"无文本 > 有文本"，问题可能在于：
1. 冻结 BERT 的特征与 Mamba fusion 模块的期望不匹配
2. `TextMambaEncoder` 的 projection + Mamba adaptation 层（depth=2 的单向 MambaLayer）可能不足以弥合这一差距

**建议:** 不解冻 BERT，而是增加 projection 层的容量（当前是 `Linear(768, 256) + LN + GELU + Dropout`，可以加到 2-3 层 MLP）。这比解冻 BERT 更安全。

**预期收益:** +1-2%（如果文本引导本身有效的话）

**风险:** 高过拟合风险

**推荐度: 2/5（不推荐解冻 BERT，可考虑增强 projection）**

### 3E: no_text_ratio 与 5-fold CV

**no_text_ratio (当前 0.15):**

15% 的训练步骤不使用文本。考虑到"无文本 > 有文本"的现状，这个比例可能太低——模型在 85% 的时间里被有害的文本信号影响。

建议调整为 **0.3**（30%），让模型有更多机会学习纯视觉特征。但这是治标不治本，根本问题是文本引导机制。

**5-fold CV:**

220 个训练样本 + 5-fold CV = 每 fold 176 训练 / 44 验证。

- 优点：更可靠的性能估计，减少数据划分的随机性
- 缺点：5x 训练时间。在 A100 上 128^3 bs=2 的单次训练约 8-12 小时，5-fold 需要 40-60 小时

**推荐:** 不做 5-fold。用固定的 220/55/94 划分（与原论文一致），将算力用于消融实验和超参搜索。最终报告时做 3 次不同 seed 的训练取均值 +/- std 即可。

**推荐度: no_text_ratio 调整 3/5, 5-fold CV 2/5**

---

## 方向 4: Colab 适配

### 4A: 数据存储与加载

**方案对比:**

| 方案 | 读取速度 | 稳定性 | 容量 | 推荐 |
|------|---------|--------|------|------|
| Google Drive | 慢（NFS over FUSE, ~50MB/s） | 高（持久化） | 15GB 免费 | 存 checkpoint |
| Colab 本地 SSD `/content/` | 快（NVMe, ~2GB/s） | 低（断连即失） | ~200GB | **存训练数据** |
| GCS bucket | 中（需 gsutil） | 高 | 按需付费 | 大数据集 |

**推荐策略:**
1. 首次运行：从 Google Drive 解压 BraTS2020 到 `/content/data/`（~15GB 解压后）
2. PubMedBERT 模型：从 Google Drive 加载到 `/content/pretrained/`
3. Checkpoint 保存：每 epoch 保存到 `/content/checkpoints/`，每 10 epochs 同步到 Google Drive
4. 训练数据**不放 Google Drive**——FUSE 挂载对 NIfTI 随机读取极慢

**数据大小估算:**

TextBraTS 369 例 BraTS2020，每例 4 modality + seg，每个 ~40MB → 总计 ~15GB。A100 Colab 通常有 200GB+ 本地存储，完全装得下。

**推荐度: 4/5（应做，数据加载策略直接影响训练速度）**

### 4B: 断连恢复策略

**Colab Pro+ 的 A100 session 最长 24 小时，普通 Pro 约 6-8 小时。**

**推荐策略:**

```
checkpoints/
  last.pth          ← 每 epoch 保存
  best.pth          ← val_dice 最优
  best_no_text.pth  ← no-text val_dice 最优
  epoch_050.pth     ← 里程碑保存
```

1. **频繁保存:** 每 epoch 保存 `last.pth`（已实现），每 10 epochs 同步到 Google Drive
2. **Resume 命令:** `python train.py --config configs/textbrats_a100.yaml --resume checkpoints/last.pth`
3. **训练脚本加自动 Drive 同步:**
   ```python
   if epoch % 10 == 0:
       shutil.copy('checkpoints/last.pth', '/content/drive/MyDrive/TextMamba3D/last.pth')
   ```
4. **Notebook 结构:** 第一个 cell 检查是否有 `last.pth` 在 Drive 中，自动设置 resume

**推荐度: 5/5（必做，Colab 断连是确定性事件）**

### 4C: Notebook 组织

**推荐结构（单个 Notebook）:**

```
Cell 1: [Setup] Mount Drive, install deps, copy data to /content/
Cell 2: [Config] A100-specific config (patch=128, bs=2, bf16=True, etc.)
Cell 3: [Resume Check] 检查 Drive 中是否有 checkpoint，自动设置 --resume
Cell 4: [Train] 主训练循环（调用 train.py 或内联代码）
Cell 5: [Evaluate] 加载 best.pth，在 test set 上评估
Cell 6: [Sync] 同步结果到 Drive
```

**关键原则:**
- 所有路径用绝对路径 `/content/...`
- 不要在 Notebook 中定义模型/训练逻辑——保留在 `.py` 文件中，Notebook 只做调用和可视化
- 用 `%%time` magic 监控每个 cell 的运行时间

**推荐度: 4/5（应做）**

---

## 方向 5: 实验设计

### 5A: 最小消融实验集

**时间预算:** 假设 Colab A100 可用 ~50 小时总计（5-6 个 session）。每次训练 128^3 bs=2 约需 8-12 小时。

**推荐实验计划（5 次训练，按优先级排序）:**

| 优先级 | 实验 | 配置变更 | 验证什么 | 预估时间 |
|--------|------|---------|---------|---------|
| P0 | **A100 Baseline** | patch=128^3, bs=2, bf16, 其余保持 | A100 红利 + 大 patch 的收益 | 10h |
| P1 | **No Text (纯视觉)** | 移除 MultiScaleFiLM + MambaFusion + text_encoder | 文本引导是否有正贡献 | 10h |
| P2 | **Text Fix** | 基于 P0/P1 结论修复文本引导 | 修复后文本是否优于纯视觉 | 10h |
| P3 | **Contrastive On** | 在 P2 基础上 contrastive_weight=0.1 | 对比学习的边际贡献 | 10h |
| P4 | **Ablation: -DWConv** | 移除 CrossScanBiMamba3DBlock 中的 dwconv | DWConv 的贡献 | 10h |

**P0 和 P1 可以并行**（如果有两个 Colab session 或连续两天跑）。

**核心逻辑:**
- P0 建立 A100 基准，量化硬件升级的收益
- P1 是最关键的消融——如果纯视觉模型 Dice > 带文本模型，则文本引导需要根本性修复
- P2 依赖 P0/P1 的结论来决定修复方向
- P3/P4 是锦上添花

### 5B: 文本引导价值的公平验证

**当前问题:** "无文本 68.67% > 有文本 68.48%"，差距虽小但方向反直觉。

**可能原因（按可能性排序）:**

1. **FiLM 破坏了有用特征** — `MultiScaleFiLM` 在所有 4 个 encoder stage 都对特征做了 `gamma * x + beta` 调制。即使 gamma 初始化为 ~1.0，经过训练后可能偏离有用方向。更大 patch + 更多 epoch 可能自行修复。

2. **MambaFusion 引入噪声** — 将文本 token 拼接到图像 token 前面再做单向 Mamba，文本的 "先入为主" 可能干扰了图像特征的 SSM 状态累积。

3. **default_text_embed 的 LayerNorm** — 无文本时，`default_text_embed`（随机初始化 * 0.02）经过 LayerNorm 后成为单位方差的随机向量，这可能比 PubMedBERT 真实输出更 "安全"（因为真实输出的分布更复杂）。

4. **文本描述质量** — TextBraTS 的专家标注可能存在噪声或不一致性。

**公平验证方案:**

| 实验 | 配置 | 对比 |
|------|------|------|
| Pure Vision | 完全移除 text encoder, FiLM, MambaFusion | 架构开销 = 0 |
| Text via FiLM Only | 保留 MultiScaleFiLM, 移除 MambaFusion | FiLM 单独贡献 |
| Text via Fusion Only | 移除 MultiScaleFiLM, 保留 MambaFusion | Fusion 单独贡献 |
| Full Text (当前) | FiLM + MambaFusion | 完整文本引导 |

这组实验可以精确定位问题是在 FiLM 还是 MambaFusion。但需要 4 次训练，时间紧张时只做 Pure Vision vs Full Text。

**推荐度: 5/5（这是整个项目最关键的问题）**

### 5C: 基线对比

**必须报告的基线（无需自己训练，引用文献即可）:**

| 方法 | BraTS2020 Dice | 来源 |
|------|---------------|------|
| TextBraTS (原论文) | 85.3% | Li et al. 2024 |
| nnU-Net | ~84-86% | Isensee et al. 2021 |
| SwinUNETR | ~83-85% | Tang et al. 2022 |
| TransBTS | ~80-82% | Wang et al. 2021 |
| UNETR | ~78-80% | Hatamizadeh et al. 2022 |
| SegMamba (BraTS2023) | 91.3% | Xing et al. 2024 |

**注意:** TextMamba3D 用的是 BraTS2020 (369 例) + TextBraTS 划分，与 BraTS2023 (1251 例) 不可直接比较。应引用同样使用 BraTS2020 的论文结果。

**应该自己跑的基线:**

| 基线 | 理由 |
|------|------|
| TextMamba3D (Pure Vision) | 证明文本引导的增量价值 |
| TextMamba3D (Full) | 完整模型性能 |

**不需要自己跑 nnU-Net/SwinUNETR 等**——它们的 BraTS2020 结果已在文献中充分报告，引用即可。

---

## 综合推荐：优先级排序

将所有方案按投入产出比排序：

| 排名 | 方案 | 预期 Dice 提升 | 工作量 | 推荐度 |
|------|------|---------------|--------|--------|
| 1 | **1A: Patch 128^3** | +8-15% | 改 config | 5/5 |
| 2 | **1B: bf16 + bs=2-4** | +2-4% (间接) | 改 config | 5/5 |
| 3 | **5B: 文本引导诊断** | 定性（决定整个文本策略） | 1 次消融训练 | 5/5 |
| 4 | **3B: LR/超参调优 + 取消 loss clamp** | +2-5% | 改 config + 几行代码 | 4/5 |
| 5 | **4B: Colab 断连恢复** | 0%（但保护训练不白费） | ~20 行代码 | 5/5 |
| 6 | **1C: num_workers=4** | 训练速度 +20-30% | 改 config | 4/5 |
| 7 | **3A: 对比学习** | +1-3% | 改 config（需先修复文本） | 3/5 |
| 8 | **2A: embed_dim=64** | +1-3% | 改 config（需重新训练） | 3/5 |
| 9 | **3C: Progressive 64->96->128** | +2-4% | ~50 行代码 | 2/5 |
| 10 | **3D: 解冻 BERT** | +1-2% | 改 config | 2/5 |

## 推荐的 A100 配置文件

```yaml
# configs/textbrats_a100.yaml
data:
  patch_size: [128, 128, 128]
  batch_size: 2
  num_workers: 4

model:
  img_size: [128, 128, 128]
  embed_dim: 48       # 先保持 48，VRAM 有余量再试 64
  dropout: 0.1
  unfreeze_text_layers: 0

loss:
  contrastive_weight: 0.0  # 先关闭，修复文本引导后再启用

training:
  epochs: 500
  lr: 0.0001           # 有效 bs=2, 比 bs=1 翻倍
  weight_decay: 0.05
  warmup_epochs: 20
  patience: 80
  gradient_accumulation: 1  # A100 上 bs=2 已够大
  gradient_checkpointing: true
  use_amp: true        # bf16 on A100
  no_text_ratio: 0.15
  gradient_clip_norm: 1.0
```

## 预期最终结果

| 场景 | 预期 Dice | 条件 |
|------|----------|------|
| 保守（仅硬件升级） | 75-78% | patch=128, bs=2, 超参微调 |
| 中等（+ 修复文本引导） | 78-82% | 上述 + 文本机制 debug |
| 乐观（+ 消融优化 + 长训练） | 80-85% | 上述 + 500 epochs + 对比学习 + 最优架构 |
| 理论上限 | ~85% | 接近 TextBraTS 原论文 SOTA |

**核心判断:** 当前 68.67% → SOTA 85.3% 的 16.6% 差距中，**patch size 从 64^3→128^3 贡献约 8-12%**，其余来自训练策略和文本引导修复。这是最值得投入的方向。
