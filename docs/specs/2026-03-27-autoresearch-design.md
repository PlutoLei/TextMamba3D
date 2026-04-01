# AutoResearch: TextMamba3D 自主进化框架设计

## 概述

本地 Python orchestrator 驱动 Colab 实验循环，自动生成假设、执行实验、分析结果、迭代改进。三层级联：推理优化 → 超参搜索 → 模块 A/B 测试。架构级改动需用户审批。

## 目标

- 基线：V5.0 (Mamba2 + SeqCA) test Mean Dice 0.8479
- 成功条件：任何超过 0.8479 的改进
- 硬件约束：Google Colab A100 40GB，单卡

## 架构

```
Local Orchestrator (Mac)
├── experiments.json    — 实验队列
├── results.json        — 结果日志
├── Hypothesis Engine   — Claude API 生成下一轮假设
└── Notebook Writer     — 生成/修改 .ipynb cells
        │
        ▼ 写入 notebook
VS Code + Colab Runtime
        │
        ▼ 结果写入 Drive / git push
Result Collector
├── 解析结果 JSON/log
├── 判断是否有提升
├── Layer 0/1: 自动决策下一轮
└── Layer 2: 暂停等用户审批
```

## 三层级联

### Layer 0: 推理时优化（零训练成本）

用现有 best_v5.0.pth，只做推理。

| 实验 | 方法 | 预期 |
|------|------|------|
| L0-1 | ET/WT 比例重标记 (ratio<3-4%) | +1.9% ET |
| L0-2 | PP 参数 sweep (et_min_size, et_wt_ratio) | +1-2% ET |
| L0-3 | 禁用 TTA 对比 | 待验证 |
| L0-4 | 概率校正 (类别不平衡) | 待验证 |

### Layer 1: 超参搜索（需训练）

Random search，每次只跑 30 epoch 快速筛选。

| 维度 | 搜索范围 |
|------|---------|
| lr | [5e-5, 1e-4, 2e-4, 5e-4] |
| loss 权重 | dice_weight, ce_weight, edge_weight 组合 |
| augmentation | elastic on/off, modality_dropout on/off |
| no_text_ratio | [0.0, 0.1, 0.15, 0.3] |

### Layer 2: 模块 A/B 测试（需审批）

从文献调研 N6-N18 中选方向，优先级：
1. N1 ET/WT 比例重标记（已在 Layer 0）
2. N6 STPF 拓扑约束 (ET=0.838)
3. N7 Boundary Loss
4. N12 DRBD-Mamba Morton Z-order
5. N9 CWCD 轮廓加权 Dice

每个方向由 Codex 生成代码，Claude 审查，用户审批后执行。

## 关键决策逻辑

### fine-tune vs 从头训练

V5.2/V5.3 的教训：V5.0 已收敛到局部最优，fine-tune 跳不出来。

规则：
- 连续 3 次 fine-tune 无提升 → 自动切换从头训练
- fine-tune 时必须 reset optimizer state + 使用更高 lr (1e-4)
- 从头训练跑 200 epoch，early stopping patience=30

### 实验预算

- Layer 0: 0 GPU hours（纯推理）
- Layer 1 每次实验: ~1 GPU hour (30 epoch)
- Layer 2 每次实验: ~3-5 GPU hours (200 epoch)

## Orchestrator 实现

单个 Python 文件 `autoresearch.py`，功能：
1. `init()` — 初始化实验队列
2. `next_experiment()` — 从队列取下一个实验
3. `generate_notebook_cell()` — 生成可执行的 notebook cell
4. `collect_results()` — 从 Drive/git 解析实验结果
5. `analyze()` — 调用 Claude API 分析结果并生成假设
6. `report()` — 生成进度报告

## 文件结构

```
TextMamba3D/
├── autoresearch/
│   ├── orchestrator.py      — 主循环
│   ├── experiments.json     — 实验队列
│   ├── results.json         — 结果日志
│   ├── cell_templates/      — notebook cell 模板
│   └── hypotheses/          — Claude API 生成的假设记录
```
