# TextMamba3D 实验使用手册

> Text-guided 3D Brain Tumor Segmentation with Mamba SSM
>
> 最后更新：2026-03-01

---

## 1. 环境概览

| 组件 | 说明 |
|------|------|
| 硬件 | RTX 4060 Laptop (8GB VRAM) |
| 运行环境 | WSL2 Ubuntu（必须，Mamba CUDA kernel 不支持 Windows 原生） |
| Python | 3.12 |
| PyTorch | 2.5.1+cu121 |
| mamba-ssm | 1.2.2（标准版，非 SegMamba 修改版） |
| causal-conv1d | 1.6.0 |
| 文本编码器 | PubMedBERT（frozen） |

**模型规模对比：**

| 模式 | 参数量 | 说明 |
|------|--------|------|
| 真实 Mamba CUDA (WSL2) | ~180.2M | 使用 selective_scan_cuda |
| MLP Fallback (Windows) | ~126.6M | 仅用于 smoke test，不用于正式实验 |

---

## 2. 环境激活

所有训练、评估、测试命令都在 **WSL2** 中执行：

```bash
# 进入项目目录并激活虚拟环境
cd /mnt/e/VSCode_Project/BS6207/TextMamba3D
source .venv/bin/activate

# 验证环境
python -c "import mamba_ssm; print('mamba-ssm OK')"
python -c "import causal_conv1d; print('causal-conv1d OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

> **注意：** 项目中有两个 venv 目录，`venv/`（旧，缺少依赖）和 `.venv/`（新，完整配置）。始终使用 `.venv/`。

---

## 3. 数据准备

### 3.1 数据集结构

项目支持两种数据集：

| 数据集 | 配置文件 | 文本来源 | 推荐场景 |
|--------|---------|---------|---------|
| TextBraTS (BraTS2020) | `configs/textbrats.yaml` | 专家标注（无信息泄露） | **正式实验（推荐）** |
| BraTS2021 | `configs/default.yaml` | 自动生成文本 | 大规模预训练 |

TextBraTS 数据已就位：

```
data/
├── BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/
│   └── BraTS20_Training_001/ ... (369 cases)
├── TextBraTS/TextBraTSData/
│   └── (专家标注的文本描述)
└── BraTS2021/ (空，需另行下载)
```

### 3.2 PubMedBERT 预训练权重

文本编码器使用本地 PubMedBERT 权重，避免训练时下载：

```bash
# 如果 pretrained/pubmedbert/ 已存在则跳过
ls pretrained/pubmedbert/

# 如果不存在，运行下载脚本
python scripts/download_pubmedbert.py
```

---

## 4. 配置说明

### 4.1 textbrats.yaml（推荐配置）

```yaml
data:
  data_dir: "./data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
  dataset_type: "textbrats"
  patch_size: [64, 64, 64]    # 8GB VRAM 安全尺寸
  batch_size: 1
  num_workers: 0               # WSL/Windows 必须为 0
  train_ratio: 0.8             # 80% 训练 / 20% 验证

model:
  img_size: [64, 64, 64]       # 与 patch_size 保持一致
  in_channels: 4               # T1, T1ce, T2, FLAIR
  out_channels: 4              # background, necrotic, edema, enhancing
  embed_dim: 96
  depths: [2, 2, 2, 2]         # 4 级 encoder
  text_embed_dim: 256
  text_max_len: 256            # TextBraTS 文本较长
  text_model_path: "./pretrained/pubmedbert"

loss:
  dice_weight: 1.0
  ce_weight: 1.0
  edge_weight: 1.0
  contrastive_weight: 0.0      # batch_size=1 时无效
  class_weights: [0.25, 3.0, 1.0, 4.0]  # background 低权重，enhancing 最高

training:
  epochs: 100
  lr: 0.0001                   # 1e-4
  weight_decay: 0.00001        # 1e-5
  warmup_epochs: 5
  patience: 20                 # Early stopping
  gradient_accumulation: 4     # 有效 batch_size = 4
  gradient_checkpointing: true # 节省 ~30-50% VRAM
  no_text_ratio: 0.0           # TextBraTS 文本质量高，无需无文本训练
```

### 4.2 default.yaml（BraTS2021 配置）

主要区别：`patch_size` 为 `[96, 96, 96]`（更大体积，VRAM 可能不够），`text_max_len` 为 128，`no_text_ratio` 为 0.1（10% 样本不使用文本）。

### 4.3 关键参数调优

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `patch_size` | [64,64,64] | 8GB VRAM 安全值；有更多显存可尝试 [96,96,96] |
| `gradient_accumulation` | 4 | 有效 batch = batch_size × accum = 1×4 = 4 |
| `gradient_checkpointing` | true | 必须开启，否则 OOM |
| `no_text_ratio` | 0.0 (TextBraTS) / 0.1 (BraTS2021) | TextBraTS 有高质量文本，不需要无文本训练 |
| `class_weights` | [0.25, 3.0, 1.0, 4.0] | Enhancing tumor 临床最重要且最稀少 |
| `warmup_epochs` | 5 | 学习率从 1% 线性升至 100% |
| `patience` | 20 | 基于 no-text Dice 的 early stopping |

---

## 5. 训练

### 5.1 完整训练

```bash
cd /mnt/e/VSCode_Project/BS6207/TextMamba3D
source .venv/bin/activate

# TextBraTS 完整训练（推荐）
python train.py --config configs/textbrats.yaml

# BraTS2021 完整训练
python train.py --config configs/default.yaml
```

### 5.2 快速测试训练（少量样本）

```bash
# 仅用 50 个样本快速验证训练流程
python train.py --config configs/textbrats.yaml --max-samples 50

# 更少样本，更快反馈
python train.py --config configs/textbrats.yaml --max-samples 10
```

### 5.3 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `configs/default.yaml` | 配置文件路径 |
| `--resume` | None | 从 checkpoint 恢复训练 |
| `--no-amp` | False | 禁用混合精度（不推荐，会 OOM） |
| `--no-text-ratio` | 0.1 | 无文本训练比例（覆盖配置文件） |
| `--grad-accum` | 4 | 梯度累积步数（覆盖配置文件） |
| `--max-samples` | None | 限制训练样本数（快速测试用） |

### 5.4 恢复训练

```bash
# 从上次中断处恢复
python train.py --config configs/textbrats.yaml --resume checkpoints/last.pth
```

### 5.5 训练输出

训练过程中会产生以下文件：

```
checkpoints/
├── best.pth           # 最佳 with-text Dice 模型
├── best_no_text.pth   # 最佳 no-text Dice 模型（公平评估）
└── last.pth           # 最新 checkpoint（含 scheduler 状态）

logs/                  # TensorBoard 日志
```

### 5.6 监控训练

```bash
# 在另一个终端中启动 TensorBoard
tensorboard --logdir logs --port 6006
# 然后在浏览器中访问 http://localhost:6006
```

### 5.7 训练策略说明

训练采用**双轨验证**：每个 epoch 分别进行 with-text 和 without-text 两次验证。

- **with-text Dice**：模型利用文本引导的完整能力
- **without-text Dice**：公平评估模式，模型仅依赖影像（用于与纯视觉模型对比）
- **Early stopping 基于 no-text Dice**：确保模型不过度依赖文本，保持影像分割底线能力

学习率调度：5 epoch 线性 warmup → cosine annealing 衰减至 0。

---

## 6. 评估

### 6.1 标准评估

```bash
# 使用文本引导评估（默认）
python evaluate.py --config configs/textbrats.yaml --checkpoint checkpoints/best.pth

# 无文本评估（公平模式）
python evaluate.py --config configs/textbrats.yaml --checkpoint checkpoints/best_no_text.pth --no-text

# 启用 TTA（三轴翻转增强）
python evaluate.py --config configs/textbrats.yaml --checkpoint checkpoints/best.pth --tta

# 保存预测结果
python evaluate.py --config configs/textbrats.yaml --checkpoint checkpoints/best.pth --save_pred
```

### 6.2 评估参数

| 参数 | 说明 |
|------|------|
| `--checkpoint` | checkpoint 路径（必需） |
| `--config` | 配置文件（需与训练一致） |
| `--no-text` | 无文本模式评估 |
| `--tta` | 启用 test-time augmentation |
| `--split` | 评估集：train / val / test |
| `--save_pred` | 保存预测 .npy 到 `predictions/` |
| `--no-amp` | 禁用 AMP 推理 |

### 6.3 评估指标

| 指标 | 区域 | 说明 |
|------|------|------|
| Dice Score | ET, TC, WT | BraTS 标准三区域 Dice 系数 |
| HD95 | ET, TC, WT | 95% Hausdorff Distance (mm) |

区域定义：
- **ET** (Enhancing Tumor)：增强肿瘤（class 3）
- **TC** (Tumor Core)：肿瘤核心（class 1 + 3）
- **WT** (Whole Tumor)：整体肿瘤（class 1 + 2 + 3）

---

## 7. Smoke Test（训练流水线验证）

在正式训练前，运行 smoke test 验证整个训练流水线：

```bash
python smoke_test.py
```

smoke test 使用合成数据（无需真实数据），验证以下内容：

| 测试项 | 验证内容 |
|--------|---------|
| Test 1 | Forward + backward + optimizer step（带文本） |
| Test 2 | Forward without text（无文本推理） |
| Test 3 | 第二步训练（检查状态无残留corruption） |

预期输出：

```
Device: cuda
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
Total params: 180,229,504
Trainable params: 180,229,504

--- Test 1: Forward + backward (with text) ---
  Output shape: torch.Size([1, 4, 64, 64, 64])
  Loss: ~3.09
  Peak VRAM: ~3576 MB (3.5 GB)
  Gradients: 776/781 params have grad, finite=True

--- Test 2: Forward (without text) ---
  Output shape: torch.Size([1, 4, 64, 64, 64])
  Output finite: True

--- Test 3: Second forward+backward step ---
  Loss step 2: ~3.06 (should decrease)

=== SMOKE TEST PASSED ===
```

---

## 8. VRAM 预算参考

| 配置 | Peak VRAM | 说明 |
|------|-----------|------|
| smoke_test (64³, batch=1) | ~3.5 GB | 合成数据 |
| textbrats (64³, batch=1, grad_ckpt) | ~5-6 GB | 预估值，含 PubMedBERT |
| default (96³, batch=1, grad_ckpt) | ~7-8 GB | 可能接近 OOM |

**OOM 应急措施（按优先级）：**

1. 确认 `gradient_checkpointing: true`
2. 将 `patch_size` 从 [96,96,96] 降至 [64,64,64]
3. 降低 `embed_dim` 从 96 到 48
4. 减少 `depths` 从 [2,2,2,2] 到 [1,1,1,1]

---

## 9. 模型架构简述

TextMamba3D 是基于 Mamba SSM 的 3D U-Net 变体，引入了文本引导机制。

### 核心模块（KG 优化后）

| 模块 | 来源论文 | 作用 |
|------|---------|------|
| 3D DWConv | UlkeMamba | 增强局部特征提取（在 SSM 扫描前） |
| Modality Grouping | CKD-TransBTS | 4 个 MRI 模态分组处理后再融合 |
| 6-Direction Scanning | Multi-Scale VMamba | 6 个方向扫描捕获 3D 空间结构 |
| Pixel-Text Cross-Attention | DenseCLIP | 文本特征像素级引导影像特征 |
| Uncertainty Gating | UD-Mamba | 不确定性自适应门控，抑制噪声区域 |
| Multi-Scale Branch | Multi-Scale VMamba | 多尺度特征提取（2x 下采样 → Mamba → 上采样） |

### 架构流程

```
Input (B, 4, D, H, W)
    ↓
Patch Embedding + Modality Grouping
    ↓
┌─────────────────────────────────────┐
│ CrossScanBiMamba3DBlock (×depths)   │
│  ├── 3D DWConv (局部特征)            │
│  ├── 6-Direction Mamba SSM Scan     │
│  ├── Cross-Attention (text→image)   │
│  ├── Uncertainty Gating             │
│  └── Multi-Scale Branch             │
└─────────────────────────────────────┘
    ↓ (4 级 encoder-decoder)
Segmentation Head → (B, 4, D, H, W)
```

文本分支：PubMedBERT (frozen) → Linear projection → 256-dim text embeddings → Cross-Attention 注入到每个 Mamba block。

---

## 10. 实验 Checklist

### 首次运行

- [ ] WSL2 Ubuntu 已安装且正常运行
- [ ] `.venv/` 环境已激活，`mamba_ssm` 和 `causal_conv1d` 可导入
- [ ] `pretrained/pubmedbert/` 目录存在
- [ ] TextBraTS 数据在 `data/BraTS2020/` 和 `data/TextBraTS/` 中
- [ ] `python smoke_test.py` 通过
- [ ] `python train.py --config configs/textbrats.yaml --max-samples 10` 跑通

### 正式实验

- [ ] 清空旧的 `checkpoints/` 和 `logs/`
- [ ] 运行完整训练：`python train.py --config configs/textbrats.yaml`
- [ ] 监控 TensorBoard 确认 loss 下降和 Dice 上升
- [ ] 训练完成后运行评估（with-text 和 no-text 两种模式）
- [ ] 如需 TTA 评估：`--tta`

### 消融实验

如需对比不同文本策略，可以调整以下参数：

```bash
# 完全无文本训练（纯视觉 baseline）
python train.py --config configs/textbrats.yaml --no-text-ratio 1.0

# 50% 无文本训练
python train.py --config configs/textbrats.yaml --no-text-ratio 0.5

# 默认 TextBraTS 配置（100% 有文本）
python train.py --config configs/textbrats.yaml
```

---

## 11. 快速启动（一键命令）

### 从 Windows CMD 启动

```bat
wsl -d Ubuntu -e bash -c "cd /mnt/e/VSCode_Project/BS6207/TextMamba3D && source .venv/bin/activate && python train.py --config configs/textbrats.yaml --max-samples 50"
```

### WSL2 终端完整流程

```bash
cd /mnt/e/VSCode_Project/BS6207/TextMamba3D
source .venv/bin/activate

# 1. Smoke test
python smoke_test.py

# 2. 快速验证（50 样本）
python train.py --config configs/textbrats.yaml --max-samples 50

# 3. 正式训练
python train.py --config configs/textbrats.yaml

# 4. 评估
python evaluate.py --config configs/textbrats.yaml --checkpoint checkpoints/best.pth
python evaluate.py --config configs/textbrats.yaml --checkpoint checkpoints/best_no_text.pth --no-text
```

---

## 12. 常见问题

**Q: Windows 上训练提示 "Mamba SSM fallback to MLP"**
A: 正常。Mamba CUDA kernel 仅在 Linux 下可用。必须在 WSL2 中训练。Windows 上的 MLP fallback 仅供 smoke test 验证逻辑。

**Q: OOM (Out of Memory)**
A: 确认 `gradient_checkpointing: true`，将 `patch_size` 改为 `[64,64,64]`。如果仍然 OOM，减小 `embed_dim` 或 `depths`。

**Q: WSL2 崩溃 (Wsl/Service/E_UNEXPECTED)**
A: 在 Windows 终端中执行 `wsl --shutdown`，等待几秒后重新进入 WSL2。

**Q: `bimamba_type == "v3"` assertion error**
A: 使用了 SegMamba 修改版的 mamba-ssm。确保使用 `.venv/` 环境（标准 mamba-ssm 1.2.2），而非系统或 conda 环境中的版本。

**Q: 如何恢复中断的训练？**
A: `python train.py --config configs/textbrats.yaml --resume checkpoints/last.pth`
