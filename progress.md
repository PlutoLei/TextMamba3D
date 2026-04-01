# Progress

## 2026-04-01 — V8.0 Stage 1 训练中 (H100)

### 当前任务
- [x] BraTS2021 数据下载 + 解压 + train/val split (1000/125/126)
- [x] BraTS2021_archive.zip 上传到 Drive
- [x] V8.0 configs 生成 (stage1 + stage2)
- [x] V8.0 notebook 修复 (cell type, data path)
- [x] batch_size 调整为 8 (H100 80GB)
- [ ] **V8.0 Stage 1 训练** (BraTS2021, 200 epoch, no text) — 重跑中
- [ ] V8.0 Stage 2 微调 (BraTS2020 + TextBraTS, 100 epoch)
- [ ] V8.0 评估 (text+TTA vs notext+TTA, 测量 text delta)

### 已完成的里程碑
- [x] V5.0 baseline: Mean=0.8479
- [x] V6.0 Bottleneck SeqCA: Mean=0.8469
- [x] V7.0 Boundary+Hierarchy Loss: Mean=0.8483, HD95_WT=1.92
- [x] 3-model ensemble: Mean=0.8511 (+0.32%)
- [x] Layer 0 推理优化: 无突破
- [x] Layer 1 超参搜索: 无突破
- [x] AutoResearch 框架实现
- [x] Crossfire code review (10 fixes)
- [x] gws 认证修复

### 下一步 (V8.0 完成后)
1. 分析 V8.0 的 text delta — 核心假设验证
2. 如果 text delta 显著: 论文 story = "预训练释放文本引导潜力"
3. 如果 text delta 不显著: 探索 Mamba-specific 融合策略
4. 多种子 ensemble (V8.0 不同 seed)
5. 开始写论文

## 2026-03-30 — V7.0 + Ensemble + V8.0 设计
- Boundary Loss + Hierarchy Loss 实现 (11 tests pass)
- V7.0 训练完成: Mean=0.8483, HD95_WT=1.92 (-34%)
- 3-model ensemble: Mean=0.8511
- V8.0 两阶段方案设计 + 实现

## 2026-03-28 — AutoResearch Layer 0 + Layer 1
- Layer 0 (23 experiments): 无突破
- Layer 1 (4 experiments): 无突破
- 确认 V5.0 是 295 样本天花板

## 2026-03-23 — Crossfire code review
- 10 fixes across 8 files
- FocalTverskyLoss 向量化 + 权重 bug 修复
- fusion.py LayerNorm 重复计算修复
- train.py SSM backend 打印修正

## 2026-03-19-22 — V5.1/V5.2/V5.3/V6.0
- Mamba3 尝试失败 (Triton crash)
- FTL/量化文本 fine-tune 失败 (V5.0 局部最优)
- Bottleneck SeqCA: TC/WT 提升但 ET 下降
