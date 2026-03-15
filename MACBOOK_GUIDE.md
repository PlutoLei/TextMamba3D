# TextMamba3D — MacBook 使用指南

> 本文件供 MacBook 端的 Claude Code 快速了解项目上下文和操作流程。
> Windows 端已完成跨平台审计并修复所有行尾问题（commit `7a3c828`）。

---

## 项目概述

TextMamba3D 是一个**文本引导的 3D 医学图像分割**模型，用于 BraTS 脑肿瘤分割。核心创新：Mamba (State-Space Model) 替代 Transformer + PubMedBERT 文本编码 + Sequential Cross-Attention 多尺度融合。

**技术栈：** PyTorch + mamba-ssm + transformers (PubMedBERT) + nibabel (NIfTI)

---

## 工作流：MacBook 编辑 → Colab 执行

```
MacBook (VS Code)          Colab (A100 40GB)
    │                           │
    │  VS Code Colab Plugin     │
    │ ─────────────────────────>│
    │  本地编辑 .ipynb           │  远程执行 cells
    │  本地 .py 同步到 /content/ │  GPU 训练/推理
    │                           │  Drive 存数据+checkpoint
```

**核心原则：MacBook 只做编辑，所有计算在 Colab 上执行。**

---

## 快速启动清单

### 前置条件

- [ ] MacBook 已安装 VS Code
- [ ] VS Code 已安装 Colab 插件（Google Colaboratory）
- [ ] 已 `git clone https://github.com/PlutoLei/TextMamba3D.git`
- [ ] Google Drive 上 `MyDrive/TextMamba3D/` 包含：
  - [x] `TextBraTS_data.zip` (3.2GB) — BraTS2020 训练数据
  - [x] `checkpoints/` — 训练 checkpoint 存储目录
  - [x] `TextMamba3D_code/` — 代码备份

### 训练步骤

1. **VS Code 连接 Colab**
   - 打开 `TextMamba3D_A100_V4.4.ipynb`
   - 连接到 Colab A100 runtime

2. **按顺序执行 Notebook Cells：**

   | Cell | 内容 | 说明 |
   |------|------|------|
   | 1 | Mount Drive + Install deps | 挂载 Drive，安装 mamba-ssm 等（pip cache 在 Drive） |
   | 2 | Verify code + Extract data | 验证本地代码已同步，解压 BraTS 数据到 runtime |
   | 4 | Patch fusion.py | 追加 SequentialCrossAttention + MultiScaleSeqCA |
   | 5 | Patch textmamba3d.py | 覆写为 v4.4 版本（MultiScaleSeqCA） |
   | 6 | Patch config | 写入 A100 训练配置（128³ patch, batch=4） |
   | 8-9 | Smoke test | 快速验证 pipeline 正确性 |
   | 11-12 | Training | 正式训练（~200 epochs） |
   | 14-15 | Evaluation | 滑窗推理 + 可视化 |

3. **训练产物自动保存到 Drive：**
   - `checkpoints/best.pth` — 最佳 with-text 模型
   - `checkpoints/best_no_text.pth` — 最佳 no-text 模型
   - `checkpoints/last.pth` — 最新 checkpoint
   - 每 10 个 epoch 自动同步到 Drive

---

## 关键文件说明

### 代码结构

```
TextMamba3D/
├── TextMamba3D_A100_V4.4.ipynb  ← 主 notebook（入口）
├── train.py                     ← 训练脚本（notebook 内部调用）
├── evaluate_full.py             ← 滑窗评估
├── inference.py                 ← 推理
├── models/
│   ├── textmamba3d.py           ← 主模型（notebook Cell 5 会覆写为 v4.4）
│   ├── encoder_3d.py            ← Mamba 3D 编码器
│   ├── decoder_3d.py            ← 层次解码器 + deep supervision
│   ├── text_encoder.py          ← PubMedBERT + Mamba 文本编码器
│   ├── fusion.py                ← 跨模态融合（notebook Cell 4 追加 SeqCA）
│   └── mamba_block.py           ← Mamba SSM 基础模块
├── data/
│   ├── brats_textbrats_dataset.py ← TextBraTS 数据加载器
│   └── transforms.py            ← 数据增强
├── losses/
│   └── __init__.py              ← CombinedLoss (Dice + CE + Edge + Contrastive)
└── configs/
    ├── textbrats_a100.yaml      ← A100 配置（Colab 用这个）
    ├── textbrats.yaml           ← 8GB 本地配置（Windows WSL 用）
    └── default.yaml             ← 默认配置
```

### 配置选择

| 场景 | 配置文件 | patch_size | batch | num_workers |
|------|---------|-----------|-------|-------------|
| **Colab A100** | `textbrats_a100.yaml` | 128³ | 4 | 4 |
| Windows 本地 | `textbrats.yaml` | 64³ | 1 | 0 |

---

## 版本注意事项

本地 `.py` 文件是 **v4.1 基线版本**。Notebook Cells 4-6 会自动将代码 patch 到 **v4.4**：

| 变更 | v4.1 (本地文件) | v4.4 (Notebook 覆写后) |
|------|----------------|----------------------|
| 融合模块 | `MultiScalePixelTextAttention` | `MultiScaleSeqCA` (Sequential Cross-Attention) |
| 文本引导策略 | 像素级注意力 | 两步 T2I→I2T 交叉注意力 |

**不要手动修改本地 `textmamba3d.py` 或 `fusion.py`**——notebook 会自动处理。

---

## 常见问题

| 问题 | 解决 |
|------|------|
| Cell 2 报 `FileNotFoundError` | VS Code Colab 插件未同步代码，重新连接 runtime |
| `TextBraTS_data.zip` not found | 确认 Drive 路径：`MyDrive/TextMamba3D/TextBraTS_data.zip` |
| mamba-ssm 安装失败 | 确认 runtime 类型为 GPU (A100)，非 CPU |
| 训练中断后恢复 | 执行 Cell 17 (Resume Training)，自动从 Drive 加载 last.pth |
| MacBook 本地 `pip install` 失败 | 正常——mamba-ssm 需要 CUDA，MacBook 不需要安装依赖 |

---

## 跨平台兼容性（已修复）

- [x] 所有 `.py` 文件行尾统一为 LF
- [x] `.gitattributes` 已配置跨平台行尾规范化
- [x] 所有路径使用 `os.path.join()` / `pathlib.Path`
- [x] 无硬编码 Windows 路径
- [x] Notebook 使用 `chr(10)` 确保 Colab 上写入 LF

---

*最后更新：2026-03-15 | crossfire audit commit: `7a3c828`*
