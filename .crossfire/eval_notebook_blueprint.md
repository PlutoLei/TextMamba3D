# Blueprint: TextMamba3D V4.5 Evaluation Notebook

## 目标
创建独立 Colab notebook `TextMamba3D_A100_V4.5_eval.ipynb`，系统消融 TTA + ET后处理 + overlap 对 v4.5 的提升效果。

## 设计决策

### 效率优化：推理-后处理分离
- evaluate_full.py 的 `--save-preds` 保存 argmax .npy（每个 case ~9MB）
- 只跑 4 次推理，后处理从保存的 pred 离线计算（不需要重新推理）
- 省时：从 8+ 次推理降为 4 次

### 推理矩阵（4 runs）

| Run | text | TTA | overlap | save-preds dir | 预估时间 |
|-----|------|-----|---------|----------------|---------|
| A | with-text | OFF | 0.5 | preds/text_base | ~14 min |
| B | no-text | OFF | 0.5 | preds/notext_base | ~14 min |
| C | with-text | ON | 0.5 | preds/text_tta | ~112 min |
| D | no-text | ON | 0.5 | preds/notext_tta | ~112 min |

总推理时间：~252 min (~4.2h)

### 离线消融矩阵（从保存的 pred 计算，秒级）

| Config | 来源 | 后处理 | et_min_size |
|--------|------|--------|-------------|
| 1. Baseline | Run A/B | OFF | — |
| 2. +PP only | Run A/B | ON | 50 |
| 3. +TTA only | Run C/D | OFF | — |
| 4. +TTA+PP | Run C/D | ON | 50 |
| 5. +TTA+PP(best) | Run C/D | ON | grid search |

### Overlap 0.75 评估
- 放在最后一个 optional cell
- 仅跑 with-text + TTA（最有价值的 config）
- 预估 ~3.5h（可选，Colab 续命后跑）

### ET min_size grid search
- 在 Run A (no-TTA with-text) 上快速测试：{20, 50, 100, 200, 500}
- 选出最优后用于最终 TTA+PP 配置

## Notebook 结构（18 cells）

### Section 1: Setup (Cell 0-3)
- Cell 0 [md]: Title + 实验设计概述（推理矩阵 + 消融矩阵）
- Cell 1 [code]: Mount Drive + install packages
- Cell 2 [code]: Extract code + data（复用 v4.5 setup）
- Cell 3 [code]: Apply code patches（SeqCA + ET-enriched，复用 v4.5 cells 4-7）

### Section 2: Evaluation Helpers (Cell 4-5)
- Cell 4 [md]: Section header
- Cell 5 [code]: 定义 3 个 helper:
  - `run_eval(ckpt, use_text, tta, postprocess, overlap, save_dir)` → 调用 evaluate_full.py 子进程，解析 stdout 返回 dict
  - `offline_postprocess(pred_dir, gt_dataset, et_min_size)` → 加载 .npy pred + GT mask，应用 postprocess_et，计算 dice/hd95
  - `build_comparison_table(results_dict)` → pandas DataFrame 格式化

### Section 3: Phase A — Inference (Cell 6-10)
- Cell 6 [md]: "Phase A: Inference Runs (~4.2h total)"
- Cell 7 [code]: Run A — baseline with-text (no TTA, overlap=0.5, --save-preds)
- Cell 8 [code]: Run B — baseline no-text
- Cell 9 [code]: Run C — TTA with-text (~112 min)
- Cell 10 [code]: Run D — TTA no-text (~112 min)

### Section 4: Phase B — Offline Analysis (Cell 11-14)
- Cell 11 [md]: "Phase B: Offline Postprocessing + Grid Search"
- Cell 12 [code]: ET min_size grid search on Run A preds → 选最优 min_size
- Cell 13 [code]: Apply postprocessing to all 4 runs with optimal min_size → 计算 8 configs 的 dice
- Cell 14 [code]: Build full comparison table (pandas) + delta computation

### Section 5: Visualization (Cell 15-16)
- Cell 15 [md]: "Results"
- Cell 16 [code]:
  - Grouped bar chart: ET/TC/WT/Mean × 5 configs
  - Delta chart: text guidance delta across configs
  - 保存为 v4.5_eval_ablation.png

### Section 6: Optional — Overlap 0.75 (Cell 17)
- Cell 17 [code]: Run TTA + overlap=0.75 (with-text only), 对比 overlap=0.5

## 约束
- 不修改 evaluate_full.py 或任何现有源码
- 所有 patches 内联到 notebook（与 v4.5 训练 notebook 相同的 patch cells）
- DRIVE_CKPT 路径: /content/drive/MyDrive/TextMamba3D/checkpoints/best_v4.5.pth
- Config: configs/textbrats_v7.yaml
- 95 test cases, split="test"
- A100 40GB 内存约束下 sw_batch_size=2
- Predictions 保存到 /content/TextMamba3D/preds/（临时目录）

## 文件清单
- 新建: `TextMamba3D/TextMamba3D_A100_V4.5_eval.ipynb` (唯一输出)
- 只读: `evaluate_full.py`, `configs/textbrats_v7.yaml`, `utils/metrics.py`, `utils/sliding_window.py`
- 复用 patches: v4.5 notebook cells 4-7 (SeqCA + ET-enriched patches)
