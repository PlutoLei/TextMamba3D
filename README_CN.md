# TextMamba3D

[![中文版](https://img.shields.io/badge/lang-中文版-red)](./README_CN.md) | [![English](https://img.shields.io/badge/lang-English-blue)](./README.md)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-ee4c2c?logo=pytorch&logoColor=white)
![Mamba](https://img.shields.io/badge/Mamba_SSM-%3E%3D1.2.0-4B0082)
![License](https://img.shields.io/badge/License-Apache_2.0-green)
![Tests](https://img.shields.io/badge/Tests-214_passed-brightgreen)

**基于统一 Mamba 架构的文本引导 3D 脑肿瘤分割框架。** O(n) 序列复杂度，可训练参数约 ~60M（embed_dim=48，PubMedBERT 最后 2 层解冻）。利用临床诊断文本引导 MRI 体积分割，在 BraTS2020 TextBraTS 数据集上进行验证。

![Architecture](docs/architecture.png)

## 架构总览

架构图中保持主干数据流简洁，以下模块已在代码中实现并以注释方式说明：

- `Modality Grouping`: `T1 + T1ce` / `T2 + FLAIR`
- 每阶段主块：`3D DWConv → CrossScan BiMamba3D`（multi-scale）
- 双路融合：`FiLM + Pixel-Text Cross-Attention`
- `Uncertainty Gating` 在 CrossScan 块内抑制高不确定性特征
- `Deep Supervision` 用于解码阶段多尺度监督

<details>
<summary><b>ASCII 架构图（fallback）</b></summary>

```
    3D MRI 体积                                        临床诊断文本
    (4ch: T1, T1ce, T2, FLAIR)                         "MRI示左侧额叶占位性病变..."
             |                                                  |
    [Modality Grouping]                               [PubMedBERT (部分冻结)]
  (T1+T1ce / T2+FLAIR)                                   ~109.5M params
             |                                                  |
      [Patch Embed 3D]                                [Projection + Mamba]
             |                                                  |
    +----------------------+                    text_global + text_seq
    | Encoder Stage 1..4   |<----- FiLM --------------+     |
    | 3D DWConv →          |                           |     |
    | CrossScan BiMamba3D  |<-- Pixel-Text Cross-Attn-+     |
    | (multi-scale)        |                                 |
    +----------+-----------+                                 |
               |                                             |
       [Causal Mamba Fusion + Uncertainty Gating] <----------+
               |
     Decoder (对称结构 + Skip + Deep Supervision)
               |
      [Final Expand + Conv3D]
               v
       分割输出 [B, 4, D, H, W]
       (背景 / 坏死 / 水肿 / 强化肿瘤)
```

</details>

## 核心亮点

- **统一 Mamba 架构** — 3D 编码器、文本编码器、融合模块、解码器以 State Space Model 为主干，辅以轻量级 Cross-Attention 实现跨模态对齐，整体保持 O(n) 序列复杂度。

- **CrossScan BiMamba3D** — 沿 3 个空间轴双向扫描（共 6 个方向），提供完整的体积上下文信息，解决单向 SSM 的信息传播盲区。

- **PubMedBERT 预训练文本编码器** — 部分冻结的生物医学语言模型（~109.5M 参数），最后 2 层解冻用于任务适配，配合轻量级 Mamba 适配层（仅 ~18K 额外参数）。

- **多尺度 FiLM 融合** — 文本通过 Feature-wise Linear Modulation 在编码器的全部 4 个阶段调制图像特征，而非仅在瓶颈层融合。

- **因果 Mamba 瓶颈融合** — 文本 token 置于图像 token 之前，利用 Mamba 因果扫描特性，让文本信息自然流入图像表征。

- **鲁棒训练策略 (v2)** — 手动 warmup + cosine LR schedule、梯度裁剪（max_norm=1.0）、NaN batch skip、bfloat16 AMP、类别加权 Dice + 3D Sobel 边缘损失 + 对比损失。

- **Pixel-Text Cross-Attention** — 在像素 token 与文本 token 间进行跨模态对齐，用于细粒度语义引导。

- **Uncertainty Gating** — 在 CrossScan BiMamba3D 块内通过不确定性门控抑制高噪声区域特征。

- **Deep Supervision** — 在解码阶段加入多尺度辅助监督，约束中间层预测一致性。

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/PlutoLei/TextMamba3D.git
cd TextMamba3D

# 安装依赖
pip install -r requirements.txt

# 下载 PubMedBERT 到本地（需要联网，约 440MB）
python scripts/download_pubmedbert.py

# 快速测试训练（50 个样本）
python train.py --config configs/textbrats.yaml --max-samples 50

# 评估
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pth
```

---

## 目录

- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [模型权重准备](#模型权重准备)
- [训练](#训练)
- [评估](#评估)
- [推理](#推理)
- [模型架构详解](#模型架构详解)
- [初步实验结果](#初步实验结果)
- [项目结构](#项目结构)
- [已知限制与展望](#已知限制与展望)
- [常见问题](#常见问题)
- [引用](#引用)
- [致谢](#致谢)
- [许可证](#许可证)

---

## 环境配置

### 系统要求

| 项目 | 要求 |
|------|------|
| Python | >= 3.8 |
| CUDA | >= 11.8（mamba-ssm 需要 CUDA 编译） |
| GPU 显存 | >= 8GB（推荐 12GB+） |
| 操作系统 | Linux / WSL2（Windows 原生不支持 mamba-ssm） |

### 快速安装

```bash
pip install -r requirements.txt
```

<details>
<summary><b>完整环境搭建（从零开始）</b></summary>

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux / WSL2

# 安装 PyTorch（根据 CUDA 版本选择）
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 Mamba SSM（需要 CUDA 工具链）
pip install mamba-ssm

# 安装其余依赖
pip install monai nibabel numpy scipy transformers pyyaml tensorboard tqdm einops pytest

# 验证安装
python scripts/verify_installation.py
```

</details>

---

## 数据准备

### TextBraTS 数据集（推荐）

TextBraTS 为每个病例提供放射科专家撰写的临床描述，避免从分割标注自动生成文本带来的信息泄露问题。

- 数据来源：[Kaggle BraTS2020](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) + [HuggingFace TextBraTS](https://huggingface.co/datasets/Jupitern52/TextBraTS)
- 共 369 例，按原论文划分为 **220 训练 / 55 验证 / 94 测试**

```
data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_t1.nii(.gz)
│   ├── BraTS20_Training_001_t1ce.nii(.gz)
│   ├── BraTS20_Training_001_t2.nii(.gz)
│   ├── BraTS20_Training_001_flair.nii(.gz)
│   ├── BraTS20_Training_001_seg.nii(.gz)
│   └── BraTS20_Training_001_flair_text.txt   # 专家标注文本
└── ...
```

数据加载时自动完成预处理：4 模态堆叠、Z-score 归一化（仅非零体素）、随机/中心裁剪。

<details>
<summary><b>备选：BraTS 2021</b></summary>

从 [Synapse](https://www.synapse.org/#!Synapse:syn25829067) 或 [Kaggle](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1) 下载。

```bash
python train.py --config configs/default.yaml
```

</details>

---

## 模型权重准备

文本编码器使用 [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)（~440MB）。训练前需先下载到本地：

```bash
python scripts/download_pubmedbert.py
```

模型将保存到 `./pretrained/pubmedbert/`，配置文件中已默认指向该路径。如果训练环境可直接访问 HuggingFace，也可以将配置中 `text_model_path` 设为 `null`，程序会自动从线上下载。

---

## 训练

### v2 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| embed_dim | 48 | 从 96 缩减，改善参数/样本比 |
| 学习率 | 5e-5 | 手动 warmup (10 epochs) + cosine 衰减 |
| Weight decay | 0.01 | AdamW 标准正则化 |
| 训练轮数 | 300 | 配合 patience=50 早停 |
| 梯度裁剪 | max_norm=1.0 | 防止梯度爆炸 |
| 混合精度 | bfloat16 | 无需 GradScaler |
| 梯度累积 | 4 | 等效 batch size = 4 |
| 数据划分 | 220/55/94 | 训练/验证/测试（原论文划分） |

### 基础训练

```bash
python train.py --config configs/textbrats.yaml
```

### 快速测试（限制样本数）

```bash
python train.py --config configs/textbrats.yaml --max-samples 50
```

### 恢复训练

```bash
python train.py --config configs/textbrats.yaml --resume checkpoints/last.pth
```

### 查看训练曲线

```bash
tensorboard --logdir logs
```

<details>
<summary><b>命令行参数</b></summary>

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `configs/default.yaml` | 配置文件路径 |
| `--resume` | None | 从 checkpoint 恢复训练 |
| `--no-amp` | False | 禁用混合精度 |
| `--no-text-ratio` | 0.1 | 无文本训练的样本比例 |
| `--grad-accum` | 4 | 梯度累积步数 |
| `--max-samples` | None | 限制训练样本数（调试用） |

</details>

<details>
<summary><b>GPU 显存配置指南</b></summary>

| GPU 显存 | 推荐配置 |
|----------|---------|
| 8 GB | `batch_size=1`, `patch_size=[64,64,64]`, `grad_accum=4`, AMP 开启 |
| 12 GB | `batch_size=1`, `patch_size=[96,96,96]`, `grad_accum=4`, AMP 开启 |
| 24 GB | `batch_size=2`, `patch_size=[96,96,96]`, `grad_accum=2`, AMP 开启 |

梯度检查点默认开启（`gradient_checkpointing: true`），可节省约 30-50% 显存，训练速度降低约 20%。

</details>

---

## 评估

```bash
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pth
```

评估报告 BraTS 标准三区域指标：

| 区域 | 定义 |
|------|------|
| ET (Enhancing Tumor) | 强化肿瘤（class 3） |
| TC (Tumor Core) | 肿瘤核心（class 1 + 3） |
| WT (Whole Tumor) | 全肿瘤（class 1 + 2 + 3） |

指标包括 Dice Score（越高越好）和 HD95 Hausdorff 距离（越低越好）。

---

## 推理

### 单例推理

```bash
python inference.py \
    --checkpoint checkpoints/best.pth \
    --input /path/to/BraTS20_Training_001 \
    --tta
```

### 批量推理

```bash
python inference.py \
    --checkpoint checkpoints/best.pth \
    --input /path/to/cases_dir \
    --batch \
    --output ./predictions
```

### 自定义文本引导

```bash
python inference.py \
    --checkpoint checkpoints/best.pth \
    --input /path/to/case \
    --text "MRI shows left frontal lobe mass with enhancing component"
```

输出 NIfTI 格式分割结果，保留原始仿射变换矩阵。

---

## 模型架构详解

### 组件概览

| 组件 | 模块 | 参数量 | 关键设计 |
|------|------|--------|---------|
| 模态分组与编码入口 | Modality Grouping + PatchEmbed3D | ~0.5M | `T1+T1ce` / `T2+FLAIR` 物理分组 |
| 图像主干 | (3D DWConv → CrossScanBiMamba3D) ×4 | ~12M | 每阶段局部卷积 + 多尺度 6 方向扫描 |
| 文本编码器 | PubMedBERT + Projection + MambaLayer | ~110M (109.5M frozen) | 全局/序列文本双表征 |
| 双路跨模态融合 | MultiScaleFiLM + PixelTextCrossAttention | ~2M | 阶段级调制 + 像素-文本对齐 |
| 不确定性门控 | UncertaintyGating | ~0.3M | 抑制高不确定性区域噪声 |
| 解码与监督 | Decoder ×4 + Deep Supervision Heads | ~5M | 对称解码 + 多尺度辅助输出 |
| **总参数** | | **~130M** | 含 PubMedBERT |
| **总可训练参数** | | **~60M** | PubMedBERT 冻结 + 最后 2 层解冻 |

### CrossScan BiMamba3D 与 3D DWConv

每个编码阶段采用 `3D DWConv → CrossScan BiMamba3D` 的顺序：先提取局部纹理，再进行全局序列建模。CrossScan 通过三轴双向扫描增强 3D 上下文覆盖：

1. **D-H-W**（深度优先）：正向 + 反向
2. **H-W-D**（高度优先）：正向 + 反向
3. **W-D-H**（宽度优先）：正向 + 反向

6 个方向输出在多尺度分支中聚合后进入后续阶段。

### 双路融合策略

1. **多尺度 FiLM**：文本全局特征在编码器全部 4 个阶段通过 `output = gamma * features + beta` 调制图像特征。
2. **Pixel-Text Cross-Attention**：在像素 token 与文本 token 之间建立细粒度对齐，补充 FiLM 的全局调制能力。
3. **Uncertainty Gating**：在 CrossScan BiMamba3D 块内部，对多尺度特征进行不确定性门控，降低高噪声区域的特征聚合干扰。
4. **因果 Mamba 融合**：在瓶颈层将文本 token 置于图像 token 之前，利用因果扫描路径将文本上下文注入图像表征。

### 损失函数

```
L_total = L_main + lambda_ds * L_deep_supervision + L_edge + lambda_c * L_contrastive
L_main  = L_dice + L_ce

- L_dice:         类别加权 Dice（权重: [0.25, 3.0, 1.0, 4.0]）
- L_ce:           类别加权交叉熵
- L_deep_supervision: 多尺度辅助输出监督（解码中间层）
- L_edge:         3D Sobel 边缘加权惩罚，提升边界清晰度
- L_contrastive:  双向图像-文本对齐（batch_size > 1 时启用）
```

### 可复现信息

| 项目 | 设置 |
|------|------|
| 数据集 | BraTS2020 TextBraTS |
| 样本量与划分 | 369 例：220 训练 / 55 验证 / 94 测试 |
| 随机种子 | 42 |
| 硬件 | RTX 4060 Laptop 8GB（WSL2） |
| 输入 patch | 64³ |
| Batch Size | 1 |
| 梯度累积 | 4 |
| 训练加速 | bfloat16 AMP + Gradient Checkpointing |
| 当前状态 | v2 训练进行中（300 epochs） |

---

## 初步实验结果

> **注意**: 模型当前正在训练中，结果将在训练完成后更新。

---

## 项目结构

```
TextMamba3D/
├── configs/
│   ├── default.yaml              # BraTS2021 配置 (96³ patch)
│   └── textbrats.yaml            # TextBraTS 配置 (64³ patch, v2 优化)
├── data/
│   ├── brats_dataset.py          # BraTS2021 数据加载器
│   ├── brats_textbrats_dataset.py # TextBraTS 数据加载器（3-way split）
│   ├── text_generator.py         # 从分割 mask 自动生成诊断文本
│   └── transforms.py             # 3D 数据增强（裁剪/翻转/弹性/噪声）
├── models/
│   ├── mamba_block.py            # MambaBlock, BiMamba, CrossScanBiMamba3D
│   ├── encoder_3d.py             # PatchEmbed3D, PatchMerging3D, MambaEncoder3D
│   ├── text_encoder.py           # PubMedBERT + Mamba 文本编码器
│   ├── fusion.py                 # FiLM, MultiScaleFiLM, MambaFusion
│   ├── decoder_3d.py             # PatchExpanding3D, MambaDecoder3D
│   └── textmamba3d.py            # TextMamba3D 主模型
├── losses/
│   ├── dice_loss.py              # 类别加权 Dice 损失
│   ├── edge_loss.py              # 3D Sobel 边缘损失
│   ├── contrastive_loss.py       # 双向对比损失
│   └── __init__.py               # CombinedLoss
├── utils/
│   ├── metrics.py                # Dice, HD95（BraTS 区域指标）
│   ├── tta.py                    # 测试时增强
│   └── sliding_window.py         # 高斯加权滑动窗口
├── tests/                        # 214 个测试，7 个测试文件
├── scripts/
│   ├── download_pubmedbert.py    # PubMedBERT 预下载脚本
│   ├── prepare_brats.py          # 数据预处理脚本
│   └── verify_installation.py    # 安装验证脚本
├── train.py                      # 训练（bfloat16 AMP + 梯度裁剪 + 手动 LR）
├── evaluate.py                   # 评估（BraTS 区域指标 + TTA）
├── inference.py                  # 推理（滑动窗口 + TTA + 自定义文本）
├── smoke_test.py                 # 端到端冒烟测试脚本
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 已知限制与展望

- 当前仅在 BraTS2020 TextBraTS 单数据集上完成验证，尚未完成跨数据集泛化测试。
- 训练集仅 220 例，数据规模有限。
- 文本输入依赖人工撰写的临床描述，文本质量与书写风格会影响引导效果。
- mamba-ssm 需要 Linux/WSL2 环境，不支持 Windows 原生运行。

---

## 常见问题

<details>
<summary><b>WSL2 无法连接 HuggingFace / 网络不通</b></summary>

WSL2 网络问题导致无法在线下载 PubMedBERT。解决方法：

1. 在 Windows 上运行预下载脚本（Windows 网络正常的情况下）：
   ```bash
   cd TextMamba3D
   python scripts/download_pubmedbert.py
   ```
2. 下载完成后，WSL2 通过 `/mnt/e/...` 路径直接读取本地文件，无需联网。

如需修复 WSL2 网络本身，检查 DNS 和代理配置：
```bash
cat /etc/resolv.conf
sudo sh -c 'echo "nameserver 8.8.8.8" > /etc/resolv.conf'
```

</details>

<details>
<summary><b>mamba-ssm 安装失败</b></summary>

mamba-ssm 需要 CUDA 工具链编译。确认：

```bash
nvcc --version   # 需要 CUDA 11.8+
```

如果安装超时，尝试从源码安装：
```bash
pip install mamba-ssm --no-build-isolation
```

注意：Windows 原生不支持 mamba-ssm，必须在 Linux 或 WSL2 环境下安装和运行。

</details>

<details>
<summary><b>显存不足 (OOM)</b></summary>

1. 减小 `patch_size`：96 → 64（在 config YAML 中修改）
2. 确认 `gradient_checkpointing: true`（默认已开启）
3. 使用 `--grad-accum 8` 增大累积步数
4. 确认 AMP 已启用（默认开启，`--no-amp` 会禁用）

</details>

<details>
<summary><b>ImportError: GreedySearchDecoderOnlyOutput</b></summary>

transformers 版本过新导致。降级至测试通过的版本：

```bash
pip install transformers==4.38.0
```

</details>

<details>
<summary><b>训练启动后卡住不动</b></summary>

首次运行时 Mamba CUDA 内核需要即时编译（JIT），通常需要 1-2 分钟，之后会自动继续。

</details>

---

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{textmamba3d2026,
  title={TextMamba3D: Text-Guided 3D Medical Image Segmentation with Unified State Space Models},
  author={Lei, Yuxuan},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 致谢

本项目基于以下优秀工作：

- [Mamba](https://github.com/state-spaces/mamba) — State Space Model 架构
- [U-Mamba](https://github.com/bowang-lab/U-Mamba) — Mamba 医学图像分割应用
- [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) — 生物医学语言模型
- [BraTS 2020](https://www.synapse.org/#!Synapse:syn25829067) — 脑肿瘤分割挑战赛
- [TextBraTS](https://huggingface.co/datasets/Jupitern52/TextBraTS) — 文本引导脑肿瘤分割（MICCAI 2024）
- [MONAI](https://monai.io/) — 医学人工智能开放网络

---

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 开源。
