# Blueprint LOCKED: TextMamba3D V4.5 Evaluation Notebook
> Debate resolved: 10/10 issues addressed. This version is frozen.

## 目标
创建独立 Colab notebook `TextMamba3D_A100_V4.5_eval.ipynb`，系统消融：
1. TTA (8-fold flip)
2. ET 后处理 (connected-component filter)
3. v4.4 + v4.5 Ensemble
4. (Optional) overlap 0.75

## 核心设计决策

### D1: 直接 Python 调用，不解析 stdout
不用 subprocess 调 evaluate_full.py，而是 import 其函数（sliding_window_inference, sliding_window_inference_tta, postprocess_et, load_model）直接在 notebook 中调用。结构化结果，无解析风险。（修复 Debate #3）

### D2: 推理-后处理分离 + Drive 持久化
- 推理保存 softmax probs（float16 压缩）到 Drive，不只是 argmax
- 后处理从 saved probs 离线计算
- 保存路径: `/content/drive/MyDrive/TextMamba3D/eval_preds/v45/`（修复 #6）
- 为什么存 softmax 而非 argmax：ensemble 需要概率平均，postprocess 可以在 argmax 上做

### D3: val set grid search，test set 只评最终 config
- ET min_size grid search 在 split="val"（55 cases）上做
- 选出最优 min_size 后，只在 test set 跑一次最终评估（修复 #1）
- Grid search 只计算 Dice，不算 HD95（修复 #10）

### D4: 逐 case 跳过已完成推理
- 每个 case 推理前检查 .npy 是否已存在
- Colab 中断后恢复只需重跑 cell，自动跳过已完成的 case（修复 #5）

### D5: Patch 生效验证
- Patch 后 assert `from models.textmamba3d import TextMamba3D; hasattr(model, 'multi_scale_attn')` 类型为 MultiScaleSeqCA（修复 #2）
- Checkpoint load 后验证 state_dict key 包含 SeqCA 特征键

### D6: Dataset 重建对齐
- 离线后处理时重建 TextBraTSDataset(split="test", transform=None) 获取 GT
- 按 case_name 匹配（dataset.cases[i] 的 basename == .npy 文件名前缀）（修复 #4）

### D7: Ensemble 策略
- 加载 v4.4 和 v4.5 两个 checkpoint
- 对每个 case：两个模型分别 sliding_window_inference → softmax probs 平均 → argmax → 可选后处理
- Ensemble 不存 intermediate probs（VRAM 允许单 case 两次推理）
- 需要 v4.4 checkpoint: `/content/drive/MyDrive/TextMamba3D/checkpoints/best_v4.4.pth`
- v4.4 使用 textbrats_a100.yaml config（与 v4.5 的 textbrats_v7.yaml 架构相同，只是无 ET-enriched）

## 推理矩阵

### Phase A: 单模型推理（4 runs）
| Run | Model | text | TTA | overlap | save dir | 预估时间 |
|-----|-------|------|-----|---------|----------|---------|
| A | v4.5 | with | OFF | 0.5 | v45_text_base/ | ~14 min |
| B | v4.5 | no | OFF | 0.5 | v45_notext_base/ | ~14 min |
| C | v4.5 | with | TTA | 0.5 | v45_text_tta/ | ~112 min |
| D | v4.5 | no | TTA | 0.5 | v45_notext_tta/ | ~112 min |

### Phase B: val set grid search (~5 min)
- 对 Run A 的 val 子集做 min_size ∈ {20, 50, 100, 200, 500} sweep
- 只算 Dice（不算 HD95），选出 best_min_size

### Phase C: Ensemble 推理（2 runs，可选）
| Run | Models | text | TTA | overlap | 预估时间 |
|-----|--------|------|-----|---------|---------|
| E | v4.4+v4.5 | with | OFF | 0.5 | ~28 min |
| F | v4.4+v4.5 | no | OFF | 0.5 | ~28 min |

### Phase D: 离线消融 + 最终结果
从 saved probs 计算 10 个 config 的 Dice：

| Config | 来源 | PP | 备注 |
|--------|------|-----|------|
| 1. v4.5 Baseline | A/B | OFF | 现有结果复现 |
| 2. v4.5 +PP | A/B | ON (best_min_size) | |
| 3. v4.5 +TTA | C/D | OFF | |
| 4. v4.5 +TTA+PP | C/D | ON (best_min_size) | ★ 预期最佳单模型 |
| 5. Ensemble base | E/F | OFF | |
| 6. Ensemble +PP | E/F | ON (best_min_size) | ★ 预期最佳整体 |

HD95 只算 config 4 和 6。

## Notebook 结构（20 cells）

### Section 1: Setup (Cell 0-4)
- Cell 0 [md]: Title + 实验设计 + 推理矩阵表
- Cell 1 [code]: Mount Drive + install packages
- Cell 2 [code]: Extract code + data
- Cell 3 [code]: Apply code patches (SeqCA + ET-enriched) — 复用 v4.5 notebook cells 4-7
- Cell 4 [code]: Patch 验证 assert（SeqCA 类型检查 + checkpoint key 检查）

### Section 2: Helpers (Cell 5-6)
- Cell 5 [md]: "Evaluation Functions"
- Cell 6 [code]: 定义 helpers:
  - `eval_single_model(model, dataset, tta, overlap, save_dir, resume=True)` → dict of per-case results + save probs
  - `eval_ensemble(model_a, model_b, dataset, overlap, save_dir)` → dict
  - `offline_postprocess(pred_dir, dataset, min_size)` → dice dict
  - `grid_search_min_size(pred_dir, dataset, sizes)` → best size + table
  - `build_table(results)` → pandas DataFrame

### Section 3: Phase A — Single Model Inference (Cell 7-11)
- Cell 7 [md]: "Phase A: V4.5 Inference (~4.2h)"
- Cell 8 [code]: Run A — v4.5 with-text baseline
- Cell 9 [code]: Run B — v4.5 no-text baseline
- Cell 10 [code]: Run C — v4.5 with-text TTA
- Cell 11 [code]: Run D — v4.5 no-text TTA

### Section 4: Phase B — Val Set Grid Search (Cell 12-13)
- Cell 12 [md]: "Phase B: ET Post-Processing Grid Search (val set)"
- Cell 13 [code]: Grid search min_size on val set using Run A preds → 输出 best_min_size

### Section 5: Phase C — Ensemble (Cell 14-15)
- Cell 14 [md]: "Phase C: V4.4 + V4.5 Ensemble"
- Cell 15 [code]: Run E + F — ensemble inference

### Section 6: Results (Cell 16-19)
- Cell 16 [md]: "Results"
- Cell 17 [code]: 离线消融：apply postprocess to all runs, compute Dice + HD95 for best configs
- Cell 18 [code]: Build comparison table + delta computation + print
- Cell 19 [code]: Visualization (grouped bar chart + delta chart) → save v4.5_eval_ablation.png

## 约束
- 不修改 evaluate_full.py 或任何仓库源码文件
- Patches 内联到 notebook（与 v4.5 训练 notebook 相同）
- DRIVE paths:
  - v4.5 ckpt: /content/drive/MyDrive/TextMamba3D/checkpoints/best_v4.5.pth
  - v4.4 ckpt: /content/drive/MyDrive/TextMamba3D/checkpoints/best_v4.4.pth
  - Predictions: /content/drive/MyDrive/TextMamba3D/eval_preds/
- Config: configs/textbrats_v7.yaml (v4.5), configs/textbrats_a100.yaml (v4.4)
- Test cases: 动态获取 len(dataset)，不硬编码
- A100 40GB, sw_batch_size=2
- Softmax probs 保存为 float16 .npy（每 case ~28MB × 4 classes × float16 ≈ 14MB 压缩后）
