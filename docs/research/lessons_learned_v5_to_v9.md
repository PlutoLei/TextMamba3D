# TextMamba3D 失败经验教训总结 (V5.0 → V9.1)

> 更新日期: 2026-04-04
> 跨越 10+ 个实验版本，6 种融合方法，3 种训练策略

---

## 一句话总结

**同域预训练让视觉 backbone 学到了文本能提供的所有信息，文本变成了冗余模态。问题不在融合方法，而在训练策略。**

---

## 实验全景

| 版本 | 融合方法 | 训练方式 | Mean Dice | Text Delta | 结论 |
|------|---------|---------|-----------|-----------|------|
| V2 | FiLM + MambaFusion + PixelTextCA | 从零 (295 cases) | 0.6848 | -0.19% | 过度注入 + 浅层污染 |
| V3 | PixelTextCA (bottleneck only) | 从零 | 0.8815 | -0.36% | 注入点太窄 |
| V4.1 | PixelTextCA (multi-scale) | 从零 | 0.8316 | -0.02% | 方向错 (Image=Query) |
| V4.3 | PWAM (乘法融合) | 从零 | 0.8319 | -0.26% | 乘法放大噪声 |
| **V4.4** | **SeqCA (Text=Query)** | **从零** | **0.8343** | **+0.55%** | **首次正向** |
| V5.0 | SeqCA + Copy-Paste | 从零 | 0.8479 | +0.67% | 最佳从零训练 |
| **V8.0** | SeqCA | **BraTS2021 预训练** | **0.8753** | **0.00%** | 预训练杀死文本 |
| V9.0 | SeqCA + freeze/dropout/align | 预训练微调 | 0.8723 | -0.02% | 4 个修复全无效 |
| V9.1 | **ConcatScan (Mamba 原生)** | 预训练微调 | 0.8719 | -0.01% | 换融合也无效 |

---

## 教训 1: 同域预训练让辅助模态失效

**现象:** V5.0 从零训练时 text delta = +0.55%。V8.0 用 BraTS2021 预训练后 delta 降为 0%。

**原因:** BraTS2021 预训练直接教会了 backbone 脑瘤分割的所有特征。文本能提供的信息（肿瘤位置、形态、类型）backbone 已经从图像中学到了。对比 TextBraTS (SwinUNETR)，它用 ImageNet 跨域预训练，backbone 对脑瘤一无所知，文本才能填补知识缺口 (+1.5% delta)。

**文献支撑:**
- Modality Competition (ICML 2022): 已收敛的模态压制未收敛的模态
- Representation Collapse (ICML 2025 Spotlight): 融合层神经元被强模态占满
- A Theory of Multimodal Learning (NeurIPS 2023): 理论上多模态应该帮助，但假设联合训练

**教训:** 预训练数据集与下游任务越近，辅助模态越冗余。跨域预训练反而给辅助模态留了空间。

---

## 教训 2: 融合方法不是瓶颈

**现象:** 在预训练 backbone 上试了 6 种融合方法，text delta 全为零：

| 融合方法 | 类型 | Text Delta |
|---------|------|-----------|
| SeqCA (Text=Query) | Cross-attention | 0.00% |
| + freeze warmup 10 epoch | 优化策略 | 0.00% |
| + 非零初始化 | 优化策略 | 0.00% |
| + vision dropout 15% | 优化策略 | 0.00% |
| + InfoNCE alignment loss | 对齐损失 | 0.00% |
| ConcatScan (Mamba 原生) | SSM 原生扫描 | 0.00% |

**教训:** 当 backbone 已经"知道答案"时，无论用什么方法把文本注入进去，模型都不会用它。问题在上游（训练策略），不在下游（融合方法）。

---

## 教训 3: 多个修复同时上线会互相干扰

**现象:** V9.0 同时上了 4 个修复（freeze + non-zero init + dropout + alignment），绝对性能反而下降了 0.3%（0.8753 → 0.8723）。V9.1 单独换了 ConcatScan，也下降了 0.34%。

**原因:**
- Alignment loss 和 vision dropout 引入了训练不稳定性（HIGH LOSS 频繁出现）
- freeze 只有 10 epoch 太短，解冻后 backbone 立刻重新主导
- 多个变量同时改，无法判断哪个有效哪个有害

**教训:** Red Line #6（每次只改一个变量）被违反了。即使每个修复单独看都有文献支撑，组合使用时可能互相冲突。

---

## 教训 4: 10 epoch 冻结不等于真正的冻结

**现象:** V9.0 冻结 backbone 10 epoch，但 text delta 仍为零。

**原因:** BLIP-2 (ICML 2023) 的成功是**永久冻结** backbone + 轻量级 Q-Former 桥接。10 epoch 冻结太短，解冻后 backbone 梯度立刻压制文本。文本分支在 10 epoch 内还没学到足够有意义的表征。

**教训:** 冻结策略需要足够长——要么整个 Stage 2 完全冻结，要么永久冻结 + 独立桥接模块。

---

## 教训 5: Mamba 特征与 cross-attention 确实不兼容，但这不是当前的主要矛盾

**现象:** SeqCA (cross-attention) 在从零训练时 delta = +0.55%，在预训练 backbone 上 delta = 0%。ConcatScan (Mamba 原生) 在预训练 backbone 上 delta 也 = 0%。

**分析:**
- 从零训练时 cross-attention 能工作（+0.55%），说明不兼容问题存在但不致命
- 预训练后 cross-attention 和 Mamba 原生扫描都不工作（0%），说明当前主要矛盾是训练策略，不是融合方法
- Mamba 特征与 cross-attention 的不兼容是**次要矛盾**，在解决训练策略问题后可能需要处理

**教训:** 区分主要矛盾和次要矛盾。当前主要矛盾是同域预训练导致的模态冗余，不是融合方法。

---

## 教训 6: Colab 训练的工程脆弱性

**现象:**
- V8.0 第一次训练 77 epoch 时断电，checkpoint 未保存（`DRIVE_CKPT_DIR` 未设置）
- V8.0 第二次训练到 epoch 162，以为只到 84（因为看到的是 log 的一部分）
- V9.0 的 `last.pth` 被多个 stage 共用，导致 resume 时可能用错 checkpoint
- VS Code 回退 NotebookEdit 的修改，必须用 `python3 + json.dump` 写 notebook

**教训:**
- `DRIVE_CKPT_DIR` 环境变量**必须在 Setup cell 设置**，不能等到训练 cell
- 每个 stage 必须有**专属 checkpoint 文件名**（`last_stage1.pth`、`best_V8.0.pth`）
- 后台自动同步守护线程是必要的（每 2-3 分钟 + 写入安全检查）
- Notebook 修改必须通过 Python 脚本，不能依赖 NotebookEdit

---

## 教训 7: 数据泄漏风险需要提前规划

**现象:** BraTS2021 包含 ~365/369 的 BraTS2020 病例（重新匿名化了 ID）。V8.0 在 BraTS2021 上预训练后在 BraTS2020 上评估，存在数据泄漏。

**缓解:**
- Text delta 测量不受影响（相对比较，两组用同一个泄漏的 backbone）
- 绝对 Dice 偏高，需要消融实验（剔除重叠后重跑）
- BrainSegFounder (MedIA 2024) 做了同样的事并发表，但审稿人可能还是会问

**教训:** 跨年数据集的重叠在设计实验时就要考虑，不能等到写论文再处理。

---

## 已建立的红线 (从失败中提炼)

| # | 红线 | 来源 |
|---|------|------|
| 1 | 禁止单向因果 SSM 融合 | V2 MambaFusion 失败 |
| 2 | 避免浅层注入 (Stage 0-1) | V2 FiLM 失败 |
| 3 | 禁止伪文本嵌入 | V2 default_text_embed 污染基线 |
| 4 | 必须多尺度融合 | V3 单 bottleneck 不够 |
| 5 | 加法 > 乘法 | V4.3 PWAM 噪声放大 |
| 6 | **每次只改一个变量** | V4.3 + V9.0 多变量同时改无法调试 |
| 7 | Text=Query > Image=Query | V4.4 SeqCA 成功验证 |
| 8 | **同域预训练会杀死辅助模态** | V8.0/V9.0/V9.1 全部验证 |
| 9 | **冻结 backbone 要么足够长要么不冻** | V9.0 10 epoch 太短 |
| 10 | **先解决训练策略再调融合方法** | V9.0/V9.1 融合方法不是瓶颈 |

---

## 下一步方向

基于以上教训，最有希望的方向是：

1. **3-stage 训练** — Stage 1 视觉预训练 (已有) → Stage 2 完全冻结 backbone 只训文本+融合 → Stage 3 联合微调
2. **从零训练 ConcatScan** — 验证文本在弱 backbone 下确实有用（快速实验）
3. **文本条件化预训练** — 在 Stage 1 就引入文本（生成 BraTS2021 的文本描述），防止 backbone 学成"不需要文本"的表征
