# BraTS 顶级方法的 TTA 与后处理策略研究

> 研究目标：梳理 BraTS2020 及后续挑战赛中顶级方法的 Test-Time Augmentation (TTA) 和 Post-Processing (PP) 策略，重点关注对 Enhancing Tumor (ET) Dice 的影响。

---

## 1. nnU-Net on BraTS2020（冠军方案, Isensee et al.）

**来源：** arXiv:2011.00848, BraTS2020 第一名

### 1.1 TTA 策略

nnU-Net 默认的 TTA 是沿三个轴进行镜像翻转（mirroring），在 3D 场景下产生 2^3 = 8 个预测，然后对 softmax 输出取平均。具体的 8 种组合为：

| 编号 | X 轴翻转 | Y 轴翻转 | Z 轴翻转 |
|------|---------|---------|---------|
| 1 | 否 | 否 | 否 |
| 2 | 是 | 否 | 否 |
| 3 | 否 | 是 | 否 |
| 4 | 否 | 否 | 是 |
| 5 | 是 | 是 | 否 |
| 6 | 是 | 否 | 是 |
| 7 | 否 | 是 | 是 |
| 8 | 是 | 是 | 是 |

**关键：nnU-Net 不使用旋转 TTA，也不使用多尺度 TTA**。仅使用镜像翻转。

在 BraTS2020 比赛的 Isensee 论文中，TTA 并未被单独列为消融实验的一项——论文重点消融的是 region-based training、batch size、data augmentation、batch normalization、batch Dice 等修改项。TTA 作为 nnU-Net 的标准推理流程默认开启。

### 1.2 滑动窗口推理

- **Patch 大小：** 128 x 128 x 128
- **重叠率（step_size）：** 0.5（即相邻 patch 重叠 50%）
- **Gaussian 加权：** 使用高斯重要性权重，中心体素权重高、边缘体素权重低，抑制拼接伪影

nnU-Net v2 代码中的默认参数：`tile_step_size=0.5, use_gaussian=True, use_mirroring=True`

### 1.3 Region-Based Training（BraTS 专属优化）

原始标签是 edema (ED)、necrosis (NCR)、enhancing tumor (ET) 三个类别，但 BraTS 评估的是三个部分重叠的**区域**：

| 区域 | 包含的标签 |
|------|----------|
| Whole Tumor (WT) | ED + NCR + ET |
| Tumor Core (TC) | NCR + ET |
| Enhancing Tumor (ET) | ET |

Isensee 的关键修改是将 softmax 替换为 sigmoid，直接以三个区域（WT/TC/ET）作为优化目标，每个区域独立用 binary cross-entropy 优化。这样网络直接学习区域层级关系，而非独立类别。

### 1.4 后处理：ET 替换规则

这是 BraTS 比赛中最重要的后处理技巧之一：

**核心逻辑：** 当预测的 ET 体素数量小于某个阈值时，将所有 ET 体素替换为 necrosis（NCR）。

**原理：** BraTS 评估规则中，如果 ground truth 没有 ET（部分 LGG 患者确实没有增强肿瘤），但模型预测了哪怕一个 ET 体素，Dice 就是 0、HD95 就是 373.13（最差值）。反之如果预测也为空，Dice 为 1。因此移除小体积的 ET 预测可以累积更多完美排名，即使偶尔移除了真正的 ET 预测，净收益也大于损失。

**替换后的标签：** ET → NCR（确保这些体素仍属于 Tumor Core）

**阈值：** 在 cross-validation 上优化。BraTS2021 冠军 KAIST 团队使用阈值 200 体素。BraTS2023 冠军使用的阈值为 ET=100, TC=150, WT=250 体素。

### 1.5 消融实验结果（Validation Set, Dice）

| 模型配置 | WT | TC | ET | Mean |
|---------|-----|-----|-----|------|
| BL (基线 nnU-Net) | 90.60 | 84.26 | 77.67 | 84.18 |
| BL* (batch size 5) | 90.93 | 83.70 | 76.64 | 83.76 |
| BL*+R (region-based) | 90.96 | 83.76 | 77.65 | 84.13 |
| BL*+R+DA (数据增强) | 90.90 | 84.61 | 78.67 | 84.73 |
| BL*+R+DA+BN | 91.24 | 85.04 | 79.32 | 85.20 |
| BL*+R+DA*+BN | 91.18 | 85.71 | 79.85 | **85.58** |

**最终竞赛成绩（测试集，25 模型集成）：** WT=88.95, TC=85.06, ET=82.03

从消融可以看出：
- Region-based training 对 ET 的贡献：77.67 → 77.65（几乎持平，但在排名上有帮助）
- 更强数据增强的贡献：77.65 → 78.67（+1.02）
- Batch Normalization 的贡献：78.67 → 79.32（+0.65）

---

## 2. SwinUNETR / UNETR（NVIDIA, BraTS2021）

**来源：** arXiv:2201.01266, MONAI 官方教程

### 2.1 TTA 策略

根据 MONAI 官方 SwinUNETR BraTS21 代码（`research-contributions/SwinUNETR/BRATS21/main.py`），**SwinUNETR 的官方实现不使用 TTA**。推理时仅使用：

- Sigmoid 激活 + 0.5 阈值二值化
- 无翻转、无旋转、无多尺度

### 2.2 滑动窗口推理

- **ROI 大小：** 128 x 128 x 128（也有使用 96 x 96 x 96 的版本）
- **重叠率：** 默认 0.5（50%），部分实验使用 0.6（60%）
- 使用 MONAI 的 `sliding_window_inference`

### 2.3 后处理

官方代码中**没有特殊后处理**，仅做 sigmoid + 阈值化。

### 2.4 性能

SwinUNETR 在 BraTS2021 挑战赛中排名第 7，平均 Dice 92.94%。作为单模型方案，不使用 TTA 和后处理的情况下已经有竞争力。

**TextBraTS（MICCAI 2025）中报告的 SwinUNETR 基线：** ET=81.0%

---

## 3. TransBTS（MICCAI 2021）

**来源：** arXiv:2103.04430

### 3.1 TTA 策略

TransBTS 使用了 TTA，但论文**没有具体说明使用了哪些增强类型**（未指明翻转轴、旋转角度等）。仅说"Test Time Augmentation (TTA) is utilized to further improve the performance"。

### 3.2 TTA 对 ET 的定量影响

| 数据集 | 设置 | ET Dice | WT Dice | TC Dice |
|--------|------|---------|---------|---------|
| BraTS2019 | w/o TTA | 78.36 | - | - |
| BraTS2019 | w/ TTA | 78.93 | 90.00 | 81.94 |
| BraTS2020 | w/o TTA | 78.50 | - | - |
| BraTS2020 | w/ TTA | 78.73 | 90.09 | 81.73 |

**TTA 对 ET 的提升约 0.2-0.6 个点**，提升幅度有限。

### 3.3 后处理

论文**未提及任何后处理**。

---

## 4. TextBraTS（MICCAI 2025, ET=83.3%）

**来源：** arXiv:2506.16784, GitHub: Jupitern52/TextBraTS

### 4.1 推理策略

TextBraTS 的论文和代码仓库中**没有提及 TTA、后处理、或特殊推理策略**。

- 所有数据 resample 到 128 x 128 x 128 进行训练和测试（全体积处理，非滑动窗口）
- 基于 SwinUNETR 架构 + BioBERT 文本编码
- 使用 bidirectional cross-attention 融合文本和图像特征

### 4.2 ET=83.3% 的来源

TextBraTS 的 ET Dice 83.3% 来自**文本引导的特征增强**，而非 TTA 或后处理：

| 方法 | ET Dice |
|------|---------|
| 3D-UNet | 80.4% |
| nnU-Net | 82.2% |
| SegResNet | 80.9% |
| SwinUNETR (基线) | 81.0% |
| Nestedformer | 82.6% |
| **TextBraTS (本文)** | **83.3%** |

### 4.3 文本模板的消融

| 文本格式 | 平均 Dice |
|---------|----------|
| 原始文本 | 83.0% |
| 仅位置模板 | 82.5% |
| 仅特征模板 | 82.2% |
| 完整模板 | **83.3%** |

**结论：** TextBraTS 的 ET 提升主要来自文本引导信号（+2.3 vs SwinUNETR 基线），不依赖 TTA 或后处理技巧。

---

## 5. BraTS 挑战赛冠军方案的通用策略总结

### 5.1 TTA 策略总结

| 策略 | 是否被冠军使用 | 预测数 | 说明 |
|------|-------------|--------|------|
| **三轴镜像翻转** | 几乎所有冠军 | 8 | 最标准的 TTA，nnU-Net 默认策略 |
| **旋转 TTA (90° 倍数)** | **否** | - | 主流 BraTS 方案均不使用旋转 TTA |
| **多尺度 TTA** | **极少** | - | 仅个别方案使用 zoom 1.125 等微小缩放 |
| **完整 TTA (翻转+旋转+缩放+噪声)** | 少数研究 | 20 | Wang 2019 使用，但非冠军方案 |

**为什么不使用旋转 TTA？**
- 脑部 MRI 有标准朝向（轴位/矢状位/冠状位），翻转是解剖学上合理的对称操作
- 旋转会引入插值伪影，且脑部肿瘤的空间关系对朝向敏感
- 翻转的 8 次预测已经提供了足够的集成效果，旋转带来的边际收益很小
- 旋转会显著增加推理时间（尤其对 3D 体积数据）

### 5.2 TTA 对 ET Dice 的定量影响

综合多个研究的数据：

| 来源 | 方法 | ET 提升 (Dice) |
|------|------|---------------|
| Wang 2019 (BraTS2018) | 3D UNet + 完整 TTA (20次) | +1.99 |
| Wang 2019 (BraTS2018) | Cascaded Net + 完整 TTA | +0.53 |
| TransBTS (BraTS2019) | TTA (细节未知) | +0.57 |
| TransBTS (BraTS2020) | TTA (细节未知) | +0.23 |

**结论：TTA 对 ET 的提升通常在 0.2-2.0 个 Dice 点**，取决于基础模型的强度。基础模型越强，TTA 的边际收益越小。

### 5.3 滑动窗口重叠率

| 方法/框架 | 重叠率 | Patch 大小 |
|---------|--------|-----------|
| nnU-Net (默认) | 0.5 (50%) | 128^3 |
| SwinUNETR (MONAI) | 0.5 (50%) | 128^3 或 96^3 |
| SwinUNETR (高精度) | 0.6 (60%) | 128^3 |
| BraTS2023 冠军 | 0.5 (50%) | 128^3 |

**结论：0.5 是标准配置**，配合 Gaussian 加权即可有效抑制拼接伪影。个别方案使用 0.6 以获取微小提升。

### 5.4 模型集成策略

| 挑战赛 | 冠军 | 集成策略 |
|-------|------|---------|
| BraTS2020 | Isensee | 3 种配置 x 5-fold = 25 模型，sigmoid 输出取平均 |
| BraTS2021 | KAIST | 2 种架构 x 5-fold = 10 模型 |
| BraTS2021 #2 | NVIDIA SegResNet | 5-fold 自适应子集选择 |
| BraTS2021 #7 | SwinUNETR | 2 x 5-fold = 10 模型 |
| BraTS2023 | Faking_it | 3 架构 x 3 数据策略 x 5-fold = 45 checkpoint |

**通用模式：** 5-fold cross-validation + 多架构集成 + softmax/sigmoid 输出平均

### 5.5 ET 专属后处理技巧

这是 BraTS 中最关键的后处理环节，总结如下：

#### (1) ET 体素数阈值替换

当预测的 ET 体素总数 < 阈值时，将所有 ET 替换为 NCR（necrosis）。

| 挑战赛 | 阈值 |
|-------|------|
| BraTS2018 (Isensee) | ~200 体素 |
| BraTS2021 (KAIST) | 200 体素 |
| BraTS2023 (Faking_it) | 100 体素 |

**原理：** BraTS 评估中，如果 GT 无 ET 但模型预测了 ET，Dice=0, HD95=373.13。移除小量 ET 可避免这类灾难性惩罚。替换为 NCR 而非直接删除，是为了保持 TC 区域的完整性。

#### (2) ET/WT 比例阈值（针对儿童肿瘤）

BraTS2023 PED 任务中，当 ET/WT 比例 < 0.04 时，将 ET 重标注为 NCR 或 ED。

#### (3) 连通域分析 + 最小体积过滤

按区域进行连通域分析，移除过小的独立组件：

| 区域 | BraTS2023 冠军阈值 |
|------|------------------|
| WT | 250 体素 |
| TC | 150 体素 |
| ET | 100 体素 |

BraTS2023 另一方案使用更精细的 FilterObjects 参数：
- WT: 大物体 ≥ 2000 体素 + 置信度 ≥ 0.85；中物体 100-2000 体素 + 置信度 ≥ 0.925
- ET: 大物体 70-95 体素 + 置信度 ≥ 0.71；中物体 + 置信度 ≥ 0.5
- TC: ≥ 350 体素

#### (4) 连通域分析（nnU-Net 默认）

nnU-Net 的标准后处理流程：
1. 将所有前景类视为一个整体，仅保留最大连通域（如果这能提高平均 Dice）
2. 对每个类别独立重复上述过程（如果能提高该类 Dice 而不降低其他类）
3. 仅在验证集 Dice 有改善时才应用

### 5.6 区域层级约束 (ET ⊂ TC ⊂ WT)

BraTS 的三个评估区域存在天然的层级包含关系。处理方式分为两种：

**方法一：Region-based training（主流）**
- 用 sigmoid 独立预测 WT/TC/ET 三个区域
- 训练时直接约束输出符合层级关系
- Isensee 的 nnU-Net BraTS 方案采用此方法

**方法二：后处理强制约束**
- 如果出现 ET 体素不在 TC 内的情况，强制将其归入 TC
- 如果出现 TC 体素不在 WT 内的情况，强制将其归入 WT
- 实践中 region-based training 已经很大程度避免了违反层级关系的预测

### 5.7 后处理的定量影响

BraTS2023 的一个方案提供了后处理的消融数据：

| 阶段 | PED ET Dice | MEN ET Dice |
|------|------------|------------|
| 集成输出（仅 TTA） | 0.466 | 0.833 |
| + 后处理 | **0.733** | **0.852** |
| 提升 | **+0.267** | +0.019 |

**对于 PED（儿童肿瘤），后处理对 ET Dice 的提升高达 26.7 个百分点！** 这是因为儿童肿瘤中很多案例没有 ET，阈值替换规则极为关键。对于成人胶质瘤（MEN），后处理的提升则较为温和。

---

## 6. 对 TextMamba3D 项目的启示

基于以上研究，为 TextMamba3D 提出以下推理策略建议：

### 6.1 TTA 策略建议

1. **必须实现三轴镜像翻转 TTA（8 次预测取平均）**——这是所有冠军方案的标配，实现简单且收益稳定
2. **不建议使用旋转 TTA**——主流方案均不使用，边际收益极小且引入插值伪影
3. **不建议使用多尺度 TTA**——3D 体积数据的多尺度推理计算代价极高

### 6.2 滑动窗口建议

1. **重叠率 0.5 即可**，配合 Gaussian 加权
2. 如果追求极致性能，可以提高到 0.6

### 6.3 后处理建议

1. **ET 阈值替换规则**（最关键）：ET 体素数 < 200 时全部替换为 NCR
2. **连通域分析**：对每个区域保留最大连通域，移除过小独立组件
3. **区域层级约束检查**：确保 ET ⊂ TC ⊂ WT

### 6.4 模型集成建议

1. 5-fold cross-validation 集成是基础
2. 如条件允许，可以集成多种架构（如 nnU-Net + SwinUNETR）

---

## 参考文献

1. Isensee et al., "nnU-Net for Brain Tumor Segmentation", BrainLes 2020 (arXiv:2011.00848)
2. Isensee et al., "No New-Net", BrainLes 2018 (arXiv:1809.10483)
3. Hatamizadeh et al., "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images" (arXiv:2201.01266)
4. Wang et al., "TransBTS: Multimodal Brain Tumor Segmentation Using Transformer", MICCAI 2021 (arXiv:2103.04430)
5. Wang et al., "Automatic Brain Tumor Segmentation using Convolutional Neural Networks with Test-Time Augmentation", BrainLes 2018 (arXiv:1810.07884)
6. TextBraTS, "Text-Guided Volumetric Brain Tumor Segmentation", MICCAI 2025 (arXiv:2506.16784)
7. BraTS2023 冠军, "How we won BraTS 2023 Adult Glioma challenge? Just faking it!" (arXiv:2402.17317)
8. Luu & Zagoya, "Model Ensemble for Brain Tumor Segmentation in Magnetic Resonance Imaging" (arXiv:2409.08232)
9. BraTS2023 解决方案集, "Advanced Tumor Segmentation in Medical Imaging: An Ensemble Approach" (arXiv:2403.09262)
10. BraTS2021 冠军 KAIST, GitHub: rixez/Brats21_KAIST_MRI_Lab
11. nnU-Net v2 官方仓库, GitHub: MIC-DKFZ/nnUNet
12. MONAI SwinUNETR BraTS21 教程, GitHub: Project-MONAI/research-contributions
