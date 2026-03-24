# ET 分割改进方法综合调研

> 三路并行搜索（arXiv + Idea2Paper KG + OpenAlex）去重汇总
> 基线：TextMamba3D V5.0, ET=79.1%, TC=85.6%, WT=89.7%, Mean=84.8% (BraTS2020)
> 目标：ET≥82%
> 日期：2026-03-24

## 已排除（与现有 18 篇论文重叠）

以下论文已在 `docs/Papers/` 中分析，不重复列入：

| 已有编号 | 论文 | 搜索源 |
|---------|------|--------|
| #07 TextBraTS | Swin UNETR + BioBERT SeqCA | Idea2Paper |
| #08 VSSD-UNet | Non-causal SSM (ICLR 2025) | Idea2Paper |
| #09 CKD-TransBTS | Clinical knowledge-driven modality pairing | Idea2Paper |
| #17 E2ENet | Dynamic sparse feature fusion | — |
| #18 SynthSeg | Label-to-image synthesis | Idea2Paper |
| BraTS 2023 Winner | GliGAN + 3-model ensemble (2402.17317) | arXiv, 已在 mem #7060 |
| Swin UNETR baseline | Idea2Paper, OpenAlex | 已在 mem #7042 |
| MedNeXt | BraTS 2024 SOTA | 已在 mem #7059 |
| SegMamba | Long-range Mamba 3D seg (373 cites) | OpenAlex |

---

## 新发现方法汇总（按优先级排序）

### P0: 零成本/极低成本推理时改进

| # | 方法 | 来源 | 关键技术 | ET 改进 | 是否需重训 |
|---|------|------|---------|---------|-----------|
| N1 | **ET/WT 比例重标记** | arXiv 2512.14937, 2411.17617, 2409.08232 | ET体积/WT体积 < 3-4% 时将 ET 改标为 NCR | +1.9% (0.794→0.813) | 否 |
| N2 | **Radiomic 聚类后处理** | arXiv 2512.14937 | 107 radiomic features→PCA→k-means→聚类特异阈值 | +1.9% | 否 |
| N3 | **模块化 JSON 后处理** | arXiv 2507.19626 (MIST) | per-label 可配置：小目标删除(100)、最大CC提取、hole filling、morphological closing | +1.8% (0.805→0.823) | 否 |
| N4 | **禁用 TTA 测试** | arXiv 2402.17317 (BraTS 2023冠军) | BraTS 2023冠军发现禁用 nnU-Net TTA 反而提升结果 | 待验证 | 否 |
| N5 | **概率校正** | Idea2Paper (SANet) | 推理时对输出概率做类别不平衡校正 | 待验证 | 否 |

### P1: 高优先级（预期 +1~3%，中等工程量）

| # | 方法 | 来源 | 关键技术 | ET 指标 | 是否需重训 |
|---|------|------|---------|---------|-----------|
| N6 | **STPF: Stick-Breaking + 拓扑约束** | arXiv 2507.08574 | 棒折断参数化保证 ET⊆TC⊆WT；视觉+语义+拓扑三先验融合；L_seg + 0.1L_hierarchy + 0.3L_continuity + 0.3L_topology | **ET=0.838 on BraTS2020** | 是 |
| N7 | **Boundary Loss (距离变换)** | Idea2Paper (W2966434031), arXiv 2302.03868 | 等值面 L2 距离→区域积分；处理极端类别不平衡；warm-up 退火调度 | +1~2% | 是 |
| N8 | **BDoU (Boundary Dice over Union)** | OpenAlex (MICCAI 2023, 36 cites) | 分离边界 vs 非边界区域的 Dice；插件式可直接替换 | 边界指标改善 | 是 |
| N9 | **CWCD (Contour-Weighted Compound Dice)** | arXiv 2407.06176 | 分离计算轮廓/非轮廓 Dice + 轮廓加权 CE；3x3x3 erosion 提取轮廓 | +3.76% avg over CEDL | 是 |
| N10 | **CAT: 双 Prompt 协调** | Idea2Paper (NeurIPS 2024) | 解剖 3D prompt + 文本 prompt 双路协调；ShareRefiner 融合 | 直接适用于 TextMamba3D 文本引导 | 是 |
| N11 | **Multi-modality 边界形状矫正** | OpenAlex (Pattern Recognition, 157 cites) | 多模态空间信息增强 + 显式边界形状矫正模块 | 高引用，BraTS 验证 | 是 |

### P2: 架构改进（预期 +1~2%，较大工程量）

| # | 方法 | 来源 | 关键技术 | ET 指标 | 是否需重训 |
|---|------|------|---------|---------|-----------|
| N12 | **DRBD-Mamba: Morton Z-order** | arXiv 2510.14383 | Morton Z-order 3D→1D 映射替代线性展开；双分辨率 Mamba；15x 效率提升 | **ET=87.97% on BraTS2023** | 是 |
| N13 | **UlikeMamba: Tri-scan 3D Mamba** | Idea2Paper (ICLR 2025) | 三正交方向扫描 + 3D depthwise conv + 多尺度 Mamba block | 3D 医学分割 SOTA | 是 |
| N14 | **CASCADE + MUTATION loss** | Idea2Paper (WACV 2023 + MIDL) | 注意力门控多尺度解码；MUTATION loss 多阶段特征混合实现隐式集成 | 架构无关技巧 | 是 |
| N15 | **ACTransU-Net: 两阶段级联** | OpenAlex (Phys Med Bio, 16 cites) | 粗-精两阶段：OmniDim 动态卷积(浅层) + 3D Swin-Transformer(深层) | Mean Dice=84.96% (BraTS2020) | 是 |
| N16 | **S2CA-Net: 形状-尺度协同感知** | OpenAlex (IEEE TMI, 37 cites) | 多尺度注意力可变形卷积 + CNN-Transformer 并行 + 局部-全局尺度混合器 | BraTS 2019/2020/MSD 优于对比方法 | 是 |
| N17 | **ET 专用解码头** | arXiv 2409.08232 | 不同模型/解码器负责不同区域；ET 用 nnU-Net 单独训练 | 区域定制化 | 是 |
| N18 | **nnUnetFormer** | OpenAlex (Phys Med Bio) | nnU-Net 深层嵌入 Transformer | **ET=0.872 on BraTS2021 (5-fold CV)** | 是 |

### P3: 数据增强（预期 +1~5%，需重训）

| # | 方法 | 来源 | 关键技术 | ET 指标 | 备注 |
|---|------|------|---------|---------|------|
| N19 | **GliGAN 实时插入** | arXiv 2509.24973 | 训练时动态 GliGAN 插入；SNFH→ET 类别替换(70%概率)；病灶缩放 0.1-0.3x | ET=0.790 (ensemble) | BraTS 2024/2025 冠军路线 |
| N20 | **Fourier 域适应合成** | OpenAlex (CMIG, 15 cites) | 条件 GAN + Fourier Domain Adaptation 减少域偏移 | +5% Dice over real-only | BraTS2020 验证 |
| N21 | **MRI 特异增强** | arXiv 2411.17617 | 运动伪影、尖峰伪影、偏置场畸变、各向异性模拟(各10%概率) | 提升鲁棒性 | 简单实现 |
| N22 | **自适应 Copy-Paste** | OpenAlex (KBS 2025, 24 cites) | 迭代伪标签 + 自适应 copy-paste；专为肿瘤分割设计 | 肿瘤分割验证 | ✅ 已实现基础版 |

### P4: 训练优化（低成本实验）

| # | 方法 | 来源 | 关键技术 | 预期效果 | 备注 |
|---|------|------|---------|---------|------|
| N23 | **GrokFast 梯度过滤** | arXiv 2411.17617 | 放大慢变梯度分量加速泛化 | 加速收敛 | 简单添加 |
| N24 | **Pixel-wise 调制 Dice** | arXiv 2506.15744 | 逐像素调制项→同时处理类别+难度不平衡 | 改善小区域 | 插件式替换 |
| N25 | **深度监督强化** | OpenAlex (DAUnet, 66 cites), arXiv 2507.23256 | 每层解码器深度监督 + 低分辨率层降权 | ET=0.797 (BraTS2024) | 已部分支持 |
| N26 | **简单概率平均 > 复杂融合** | arXiv 2402.17317 | BraTS 2023冠军：45 checkpoint 简单平均优于 STAPLE/CNN 融合 | +0.5~1% | 零训练成本 |

---

## 与已有方法的综合对比

| 策略类别 | 已尝试 | 新发现最优方案 | 差异分析 |
|---------|--------|-------------|---------|
| **损失函数** | Focal Tversky (失败) | STPF 拓扑约束 (ET=0.838) | FTL 全局重权 FP/FN，STPF 用拓扑约束防碎片化 + 棒折断保证层级 |
| **边界增强** | EdgeEnhance3D (无效) | Boundary Loss + BDoU | EE3D 加参数学边界，BL 用距离变换做区域积分无额外参数 |
| **后处理** | 基础 CC filter (min_size=50) | ET/WT 比例重标记 + radiomic 聚类 | ✅ 已实现高级版本，可追加 N1 比例检查 |
| **TTA** | 8-fold flip (SOTA标准) | 尝试禁用 TTA | BraTS 2023冠军发现 TTA 可能反向影响 ET |
| **数据增强** | Flip+Affine+Noise+Elastic | GliGAN 实时插入 + MRI 伪影模拟 | ✅ 已实现 Copy-Paste；GliGAN 需额外训练生成器 |
| **Mamba 改进** | 标准线性扫描 | Morton Z-order (DRBD-Mamba) + Tri-scan (UlikeMamba) | 空间局部性保持更好，ET=87.97% |
| **集成** | 未使用 | 多 checkpoint 概率平均 | 零训练成本，+0.5~1% |

---

## V5.3 推荐路线图

### 即时行动（无需重训，本周可完成）

1. **N1: ET/WT 比例重标记** — 在 `postprocess_brats_advanced()` 中添加 ET/WT ratio 检查 (~5行代码)
2. **N4: 无 TTA 对比测试** — 运行 `--no-tta` 评估对比
3. **N26: 多 checkpoint 平均** — 对 V5.0 训练过程保存的多个 checkpoint 做概率平均

### 短期实验（需重训，1-2周）

4. **N7: Boundary Loss** — 添加为辅助损失项，warm-up 退火
5. **N24: Pixel-wise 调制 Dice** — 插件式替换，测试 ET 效果
6. **Copy-Paste + ET Oversampling** — ✅ 已实现，在 V5.3 训练中启用

### 中期架构改进（2-4周）

7. **N6: Stick-Breaking 参数化** — 解码器输出层改为棒折断保证 ET⊆TC⊆WT
8. **N12: Morton Z-order** — 替换 Mamba 扫描序列化方式
9. **N17: ET 专用解码头** — 分离 ET 与 TC/WT 的梯度路径

### 参考但不急于实施

- N19 GliGAN（需训练生成器，工程量大）
- N15 ACTransU-Net（完整架构替换，不适合增量改进）
- N2 Radiomic 聚类（需 PyRadiomics 依赖，收益不确定）
