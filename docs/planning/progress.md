# Progress Log

## Session: 2026-03-04

### Crossfire Pipeline: 本地 vs GitHub 比对与更新

| 阶段 | 状态 | 耗时 |
|------|------|------|
| Phase 0a EXPLORE | ✅ | 发现 5 commits + 11 files 差异 |
| Phase 0b PLAN | ✅ | 5-step 蓝图 |
| Phase 0c DEBATE | ✅ | Codex 质询 10 点，采纳 9 点 |
| Phase 0d LOCK | ✅ | 蓝图冻结 |
| Phase 1 EXECUTE | ✅ | git push + Codex README + paperbanana 架构图 |
| Phase 2 REVIEW | ✅ | Claude 初审 + Codex 终审（5 issues, all fixed） |
| Phase 3 REPORT | ✅ | 本文件 |

### 提交记录
1. `249a039` — Phase 5 training optimizations (12 files)
2. `fc8cc16` — README update + architecture diagram + docs (9 files)

### GitHub 设置更新
- Description: "Text-guided 3D brain tumor segmentation..."
- Topics: 7 个精准标签

### Codex 终审修复项
1. "无注意力机制" → 修正为 SSM 为主干 + 轻量级 Cross-Attention
2. 参数量补充冻结策略说明（默认解冻 BERT 最后 2 层）
3. 目录锚点修复（项目结构从 details 提升为 ## section）
4. Uncertainty Gating 位置修正（CrossScan 块内）
5. 中英混排统一（samples→例, Last updated→最后更新）

---

## Session: 2026-03-04 (NaN Loss Bugfix)

### Crossfire L2 Bugfix: 修复训练 NaN Loss

| 阶段 | 状态 | 详情 |
|------|------|------|
| Phase 0a EXPLORE | ✅ | 发现 5 个叠加根因（DS+AMP 爆炸 85%、Dice 空类 70%、Edge 除零 65%、无 NaN 检测 60%、GradScaler 30%） |
| Phase 0b PLAN | ✅ | 6 Fix 蓝图 |
| Phase 0c DEBATE | ✅ | Codex 质询 7 点，全部采纳（发现 clamp 不修 NaN、双重加权、union 判据无效等关键问题） |
| Phase 0d LOCK | ✅ | 蓝图 v2 冻结 |
| Phase 1 EXECUTE | ✅ | Codex 实现 4 文件修改 |
| Phase 2 REVIEW | ✅ | Claude 初审（清理 EdgeLoss 死属性）+ Codex 终审（修复 scheduler 保存不完整） |
| Phase 3 REPORT | ✅ | 本记录 |

### 修改文件
| 文件 | 修改 |
|------|------|
| `losses/__init__.py` | `_sanitize_loss` 方法，isfinite+where+clamp，DS 每步 sanitize |
| `losses/edge_loss.py` | 去 sqrt 偏置，per-sample 归一化，移除 edge_weight 属性 |
| `losses/dice_loss.py` | 空类 target_i 判据，smooth=1e-4，weight_sum 安全回退 |
| `train.py` | NaN 检测+skip，GradScaler(2^14)，所有 ckpt 保存 scheduler/scaler，resume 恢复 |

### 测试
- **214 passed**, 7 warnings (mamba_ssm 第三方), 0 failed

---

## Session: 2026-03-08 (文本注入修复)

### Crossfire L2 Research Template: 替换 FiLM+MambaFusion 为 Bottleneck Cross-Attention

| 阶段 | 状态 | 详情 |
|------|------|------|
| Phase 0a EXPLORE | ✅ | 读取 research_summary.md（6 篇论文），代码审计发现 PixelTextCrossAttention 可复用 |
| Phase 0b PLAN | ✅ | 蓝图：移除 MultiScaleFiLM + MambaFusion，替换为 bottleneck cross-attn |
| Phase 0c DEBATE | ✅ | Codex 质询发现 6 个客观问题（contrastive 错误计算、text-free 基线污染等），3 个主观建议 |
| Phase 0d LOCK | ✅ | 蓝图冻结 |
| Phase 1 EXECUTE | ✅ | 4 文件修改，7/7 测试通过 |
| Phase 2 REVIEW | ✅ | Claude 初审 CLEAN PASS，修复 2 个 dead code（unused import + variable） |
| Phase 3 REPORT | ✅ | 本记录 |

### 修改文件
| 文件 | 修改 |
|------|------|
| `models/textmamba3d.py` | 移除 MultiScaleFiLM + MambaFusion，新增 PixelTextCrossAttention at bottleneck；text-free 绕过 fusion；contrastive 从 fused 计算；清理 dead import/variable |
| `models/__init__.py` | 新增 PixelTextCrossAttention export |
| `train.py` | no-text 批次跳过 feature extraction (`need_features and use_text`) |
| `tests/test_models.py` | test_return_features_no_text 期望 None features |

### 研究依据
- TextBraTS (MICCAI 2025): bottleneck-only cross-attn +1.5% Dice
- Neural Field Conditioning (2023): Cross-Attention > FiLM > Concatenation
- FiLM 仅适合低维条件信号，不适合 256-dim BERT

### DEBATE 关键发现
1. contrastive 在 no-text 批次错误计算（向 null embed 对齐）→ 跳过
2. text-free 路径通过 default_text_embed + fusion → 绕过
3. contrastive 监督 raw bottleneck → 改为 fused bottleneck

### 预期效果
- 参数量净减 ~0.7M
- with-text val_dice > without-text val_dice（消除 -0.38% delta）
- 待 A100 训练 200 epochs 验证

### 测试
- **7/7 TextMamba3D 测试通过**（1 个 pre-existing CrossScan3D 失败，不相关）
