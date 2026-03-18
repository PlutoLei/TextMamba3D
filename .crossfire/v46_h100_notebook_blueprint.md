# Blueprint: TextMamba3D V4.6 H100 Training Notebook

## 目标
创建 `TextMamba3D_H100_V4.6.ipynb` — H100 80GB 优化的 V4.6 训练 notebook。
同时创建 `configs/textbrats_v8_h100.yaml`。

## H100 vs A100 配置差异

| 参数 | A100 (v8) | H100 (v8_h100) |
|------|-----------|----------------|
| batch_size | 4 | 8 |
| gradient_checkpointing | true | false |
| sw_batch_size | 2 | 4 |
| num_workers | 4 | 8 |
| experiment.name | *_a100_* | *_h100_* |

## Notebook 结构 (20 cells)

### Section 1: Setup (Cell 0-2)
- Cell 0 [md]: Title — "TextMamba3D — H100 Training Pipeline (v4.6)"
- Cell 1 [code]: Mount Drive (复用 V4.5 cell 1)
- Cell 2 [code]: Extract code + data (复用 V4.5 cell 2，改 DRIVE_BASE)

### Section 2: Code Patches (Cell 3-8)
- Cell 3 [md]: "## Code Patches (V4.5 + V4.6)"
- Cell 4 [code]: [v4.4-1] Append SeqCA + MultiScaleSeqCA to fusion.py (复用 V4.5 cell 4)
- Cell 5 [code]: [v4.6-1] Append RMSNorm + CrossScaleSkipAttention + TextScaleGate + MultiScaleTextGate to fusion.py (NEW)
- Cell 6 [code]: [v4.6-2] Overwrite decoder_3d.py with V4.6 version (NEW — includes CrossScaleSkipAttention integration)
- Cell 7 [code]: [v4.6-3] Overwrite textmamba3d.py with V4.6 version (修改 V4.5 cell 5 — imports MultiScaleTextGate, 新参数)
- Cell 8 [code]: [v4.5-1] Patch ET-enriched dataset (复用 V4.5 cell 6)

### Section 3: Config (Cell 9-10)
- Cell 9 [code]: [v4.6-4] Create configs/textbrats_v8_h100.yaml (H100 优化)
- Cell 10 [code]: Patch verification — assert V4.6 modules exist + checkpoint compatibility check

### Section 4: ET Preprocessing (Cell 11)
- Cell 11 [code]: ET-enriched text preprocessing (复用 V4.5 cell 9)

### Section 5: Training (Cell 12-14)
- Cell 12 [md]: "## Training (H100 80GB)"
- Cell 13 [code]: Checkpoint sync setup (复用 V4.5 cell 14，改路径为 v4.6)
- Cell 14 [code]: Run training — `python train.py --config configs/textbrats_v8_h100.yaml`

### Section 6: Evaluation (Cell 15-17)
- Cell 15 [md]: "## Evaluation"
- Cell 16 [code]: Full-volume eval with-text + no-text (改 config 为 v8_h100)
- Cell 17 [code]: Results visualization

### Section 7: Resume (Cell 18-19)
- Cell 18 [md]: "## Resume Training"
- Cell 19 [code]: Resume after disconnect

## 关键约束
- 所有 V4.6 模块内联 patch（不依赖本地 feat 分支）
- decoder_3d.py 完整覆盖（包含 CrossScaleSkipAttention 集成）
- textmamba3d.py 完整覆盖（包含 TextScaleGate + use_cross_scale_skip 转发）
- train.py 需要内联 patch 添加 3 个新 config 字段转发
- H100 不需要 gradient_checkpointing
