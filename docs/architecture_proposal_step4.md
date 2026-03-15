# TextMamba3D Step 4 Architecture Proposal: Text Guidance Effectiveness

> 基于 research_summary_step4.md，2026-03-13

## 1. 设计目标

将文本引导从当前的 -0.02%（接近中性）提升到显著正向贡献（目标 +1~3% Dice），通过同时解决：

1. **文本内容冗余** — 当前文本只是 mask 的自然语言翻译
2. **注入机制可绕过** — additive cross-attention 允许模型忽略文本

## 2. 系统架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                     │
│                                                         │
│  ┌──────────────┐    ┌──────────────────────────────┐   │
│  │   3D Volume   │    │  Knowledge-Enriched Text     │   │
│  │  (4-ch MRI)   │    │  (LLM-generated + expert)    │   │
│  └──────┬───────┘    └──────────┬───────────────────┘   │
│         │                       │                        │
│         ▼                       ▼                        │
│  ┌──────────────┐    ┌──────────────────────────────┐   │
│  │ Mamba Encoder │    │  PubMedBERT (last 2 unfrozen)│   │
│  │              │    └──────────┬───────────────────┘   │
│  │  Stage 0 ────┤              │                        │
│  │    ↓         │    ┌─────────▼─────────┐              │
│  │  Stage 1 ◄───┼────┤  PWAM + FiLM Gate │ (×)         │
│  │    ↓         │    └───────────────────┘              │
│  │  Stage 2 ◄───┼────┤  PWAM + FiLM Gate │ (×)         │
│  │    ↓         │    └───────────────────┘              │
│  │  Stage 3 ◄───┼────┤  PWAM + FiLM Gate │ (×)         │
│  │              │    └───────────────────┘              │
│  └──────┬───────┘                                       │
│         │                                                │
│         ▼                                                │
│  ┌──────────────┐    ┌──────────────────────────────┐   │
│  │   Decoder     │    │  Text-Conditioned 3D Conv    │   │
│  │  (FPN-style)  │    │  → Per-voxel BCE (L_t2v)    │   │
│  └──────┬───────┘    └──────────┬───────────────────┘   │
│         │                       │                        │
│         ▼                       ▼                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │  L_total = L_dice + L_ce + λ₁·L_t2v + λ₂·L_nec  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 3. 模块详细设计

### 3.1 Module A: Knowledge-Enriched Text Generation (离线预处理)

**目标：** 为每个 BraTS 样本生成超越 mask 信息的临床文本。

**输入：** segmentation mask + 原始 expert text

**输出：** enriched text（英文，~200-400 tokens）

**生成流程：**

```python
# Step 1: 从 mask 提取形态学特征
features = {
    "location": get_lobe_from_centroid(mask),        # 已有: text_generator.py
    "volume_et": compute_volume(mask == 4),           # 已有
    "volume_tc": compute_volume(mask >= 1),           # 已有
    "volume_wt": compute_volume(mask >= 1),           # 已有
    "enhancing_ratio": vol_et / vol_tc,               # 新增
    "necrosis_ratio": vol_ncr / vol_tc,               # 新增
    "boundary_regularity": surface_to_volume(mask),   # 已有
    "eloquent_proximity": distance_to_motor_cortex(centroid),  # 新增
}

# Step 2: LLM prompt 模板
prompt = f"""Given a brain MRI with the following tumor characteristics:
- Location: {features['location']}
- Enhancing tumor volume: {features['volume_et']:.1f} cm³
- Enhancing ratio: {features['enhancing_ratio']:.2f}
- Necrosis ratio: {features['necrosis_ratio']:.2f}
- Boundary: {'irregular' if features['boundary_regularity'] > threshold else 'regular'}
- Distance to eloquent cortex: {features['eloquent_proximity']:.1f} mm

Generate a clinical description covering:
1. WHO grade estimation with reasoning
2. Differential diagnosis (GBM vs metastasis vs lymphoma)
3. Functional area involvement and clinical implications
4. Histological features suggested by imaging pattern

Keep under 300 words. Use standard neuroradiology terminology."""

# Step 3: LLM 生成（离线，一次性）
enriched_text = llm_generate(prompt)  # GPT-4 / Claude
```

**与原始 expert text 的结合方式：**

```
final_text = expert_text + " [SEP] " + enriched_text
```

PubMedBERT 的 `[SEP]` token 自然分隔两段，前段是影像学描述，后段是临床推断。

**数据量：** 369 个样本 × 1 次 LLM 调用 = 369 次调用（离线完成，成本可忽略）。

### 3.2 Module B: PWAM — Pixel-Word Attention Module (借鉴 LAVT)

**目标：** 在 Mamba encoder 内部做乘性文本注入，结构上不可绕过。

**替换目标：** 替换 v4 的 `MultiScalePixelTextAttention`（additive cross-attention）。

**实现：**

```python
class PWAM3D(nn.Module):
    """Pixel-Word Attention Module adapted for 3D Mamba encoder.

    From LAVT (CVPR 2022), adapted for 3D volumes + Mamba SSM.
    """
    def __init__(self, vis_dim, lang_dim=768):
        super().__init__()
        # Cross-attention: Q=visual, K/V=language
        self.q_proj = nn.Sequential(
            nn.Conv1d(vis_dim, vis_dim, 1),
            nn.InstanceNorm1d(vis_dim)  # Critical: +2.07 oIoU in LAVT
        )
        self.k_proj = nn.Conv1d(lang_dim, vis_dim, 1)
        self.v_proj = nn.Conv1d(lang_dim, vis_dim, 1)
        self.out_proj = nn.Sequential(
            nn.Conv1d(vis_dim, vis_dim, 1),
            nn.InstanceNorm1d(vis_dim)
        )

        # Multiplicative fusion
        self.vis_proj = nn.Sequential(
            nn.Conv1d(vis_dim, vis_dim, 1), nn.GELU(), nn.Dropout(0.1)
        )
        self.mm_proj = nn.Sequential(
            nn.Conv1d(vis_dim, vis_dim, 1), nn.GELU(), nn.Dropout(0.1)
        )

        # Language gate (Tanh bounds to [-1, +1])
        self.gate = nn.Sequential(
            nn.Linear(vis_dim, vis_dim, bias=False),
            nn.ReLU(),
            nn.Linear(vis_dim, vis_dim, bias=False),
            nn.Tanh()
        )

    def forward(self, vis, lang_tokens, lang_mask):
        """
        vis:         [B, N, C]  (N = D*H*W flattened voxels)
        lang_tokens: [B, T, 768] (T = text token count)
        lang_mask:   [B, 1, T]  (padding mask)
        """
        B, N, C = vis.shape

        # Step 1: Pixel-Word cross-attention
        Q = self.q_proj(vis.permute(0,2,1))           # [B, C, N]
        K = self.k_proj(lang_tokens.permute(0,2,1))   # [B, C, T]
        V = self.v_proj(lang_tokens.permute(0,2,1))   # [B, C, T]

        attn = torch.bmm(Q.permute(0,2,1), K) / (C ** 0.5)  # [B, N, T]
        attn = attn + (1e4 * lang_mask - 1e4)          # mask padding
        attn = F.softmax(attn, dim=-1)
        lang_feat = self.out_proj(torch.bmm(attn, V.permute(0,2,1)).permute(0,2,1))
        # lang_feat: [B, C, N]

        # Step 2: Multiplicative fusion (key mechanism)
        vis_feat = self.vis_proj(vis.permute(0,2,1))   # [B, C, N]
        fused = self.mm_proj(vis_feat * lang_feat)      # element-wise multiply
        # fused: [B, C, N] → [B, N, C]
        fused = fused.permute(0,2,1)

        # Step 3: Gated residual
        gate = self.gate(fused)                         # [B, N, C]
        output = gate * fused + vis                     # gated residual

        return output
```

**插入位置：** Mamba encoder 的 Stage 1, 2, 3 之后（Stage 0 排除，token 数 32K 过多）。

**参数估算（embed_dim=48 的 TextMamba3D）：**

| Stage | vis_dim | 参数量 |
|-------|---------|--------|
| 1 | 96 | ~150K |
| 2 | 192 | ~450K |
| 3 | 384 | ~1.5M |
| **合计** | | **~2.1M** |

相比 v4 的 ~38M trainable，增加 ~5.5%，非常轻量。

### 3.3 Module C: Text-to-Voxel Contrastive Loss (借鉴 CRIS)

**目标：** 提供逐体素的文本依赖训练信号。

**核心发现：** CRIS 的 "contrastive loss" 实际上是 per-pixel BCE with text-conditioned conv kernel。3D 适配只需 Conv2d → Conv3d。

```python
class TextToVoxelLoss(nn.Module):
    """Text-conditioned per-voxel BCE loss.

    From CRIS (CVPR 2022), adapted for 3D.
    Text global feature → 3D conv kernel → convolve over voxel features → BCE.
    """
    def __init__(self, vis_dim, text_dim=768, kernel_size=1):
        super().__init__()
        self.vis_up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.vis_proj = nn.Conv3d(vis_dim, vis_dim, 1)

        k = kernel_size
        self.text_to_kernel = nn.Linear(text_dim, vis_dim * k * k * k + 1)  # +1 for bias
        self.k = k
        self.vis_dim = vis_dim

    def forward(self, voxel_features, text_global, gt_mask):
        """
        voxel_features: [B, C, D, H, W] (decoder output)
        text_global:    [B, 768] (PubMedBERT [CLS] token)
        gt_mask:        [B, D', H', W'] (ground truth, may need resize)
        """
        B = voxel_features.shape[0]
        x = self.vis_proj(self.vis_up(voxel_features))  # [B, C, D, H, W]

        # Text → dynamic 3D conv kernel
        kernel_params = self.text_to_kernel(text_global)  # [B, C*k³+1]
        weight = kernel_params[:, :-1].reshape(B, 1, self.vis_dim, self.k, self.k, self.k)
        bias = kernel_params[:, -1]  # [B]

        # Per-sample grouped convolution
        # Reshape for grouped conv: [1, B*C, D, H, W]
        x_grouped = x.reshape(1, B * self.vis_dim, *x.shape[2:])
        w_grouped = weight.reshape(B, self.vis_dim, self.k, self.k, self.k)

        logits = []
        for i in range(B):
            l = F.conv3d(x[i:i+1], w_grouped[i:i+1], bias=bias[i:i+1],
                        padding=self.k//2)  # [1, 1, D, H, W]
            logits.append(l)
        logits = torch.cat(logits, dim=0)  # [B, 1, D, H, W]

        # Resize GT mask to match
        gt_resized = F.interpolate(
            gt_mask.unsqueeze(1).float(), size=logits.shape[2:],
            mode='nearest'
        )
        gt_binary = (gt_resized > 0).float()  # foreground vs background

        return F.binary_cross_entropy_with_logits(logits, gt_binary)
```

**kernel_size 选择：** 用 `k=1` 替代 CRIS 的 `k=3`，将参数从 27C 降到 C。3D 场景中 3×3×3=27 倍膨胀过大。

**无额外可学习参数：** 仅 `text_to_kernel` 一个线性层（768 → C+1）。

### 3.4 Module D: Text-Necessity Loss (新提出)

**目标：** 显式训练信号，惩罚模型忽略文本。

**灵感：** Classifier-Free Guidance (ICLR 2025) 的训练范式——同时训练有条件和无条件版本。

```python
class TextNecessityLoss(nn.Module):
    """Force the model to depend on text by comparing
    text-guided prediction vs text-masked prediction.
    """
    def __init__(self, margin=0.02):
        self.margin = margin  # 要求有文本时至少好 margin

    def forward(self, pred_with_text, pred_without_text, gt_mask):
        """
        pred_with_text:    [B, num_classes, D, H, W]
        pred_without_text: [B, num_classes, D, H, W]  (text replaced with zeros)
        gt_mask:           [B, D, H, W]
        """
        dice_with = compute_dice(pred_with_text, gt_mask)
        dice_without = compute_dice(pred_without_text, gt_mask)

        # Hinge loss: penalize when text-guided is not better by margin
        loss = F.relu(self.margin - (dice_with - dice_without))
        return loss.mean()
```

**训练策略：**
- 每个 batch 前向传播两次：一次正常（有文本），一次 text-masked（零向量替代 text embedding）
- `L_nec` 仅在 warmup 之后启用（前 30 epoch 不用，让模型先学基础分割）
- 训练开销 ×2 前向，但梯度只回传有文本版本

### 3.5 Module E: Embedding Perturbation (简化版 RLEG)

**目标：** 增加 text embedding 的训练时多样性。

**简化理由：** RLEG 原版需要预训练 12 层 diffusion Transformer，369 样本不足。简化为直接在 PubMedBERT 输出上施加可控扰动。

```python
class EmbeddingPerturbation(nn.Module):
    """Simplified RLEG: Gaussian noise + learned direction perturbation."""

    def __init__(self, embed_dim=768, num_directions=8):
        super().__init__()
        # Learned perturbation directions (like PCA basis)
        self.directions = nn.Parameter(torch.randn(num_directions, embed_dim) * 0.01)
        self.scale = nn.Parameter(torch.ones(num_directions) * 0.1)

    def forward(self, text_emb, training=True):
        """
        text_emb: [B, T, 768] (PubMedBERT output)
        Returns: [B, T, 768] (perturbed, only during training)
        """
        if not training:
            return text_emb

        # Random combination of learned directions
        coeffs = torch.randn(text_emb.shape[0], self.directions.shape[0],
                            device=text_emb.device)  # [B, num_directions]
        perturbation = torch.einsum('bn,nd->bd', coeffs * self.scale, self.directions)
        # perturbation: [B, 768] → broadcast to [B, 1, 768]

        return text_emb + perturbation.unsqueeze(1)
```

**参数量：** 8 × 768 + 8 = ~6.2K（可忽略）。

## 4. 训练策略

### 4.1 Loss 函数

```
L_total = L_dice + L_ce + λ₁ · L_t2v + λ₂ · L_nec
```

| Loss | 权重 | 启用时机 | 来源 |
|------|------|---------|------|
| L_dice | 1.0 | Epoch 0+ | 标准 |
| L_ce | 1.0 | Epoch 0+ | 标准 |
| L_t2v (text-to-voxel) | 0.1 | Epoch 0+ | CRIS |
| L_nec (text-necessity) | 0.05 | Epoch 30+ | 新提出 |

### 4.2 训练流程

```
Phase A (Epoch 0-30): 基础训练
  - L_total = L_dice + L_ce + 0.1 × L_t2v
  - PWAM + FiLM Gate 学习文本-视觉对齐
  - Embedding Perturbation 开启
  - PubMedBERT last 2 layers unfrozen

Phase B (Epoch 30-300): 强化文本依赖
  - L_total = L_dice + L_ce + 0.1 × L_t2v + 0.05 × L_nec
  - Text-Necessity Loss 启用，强制文本贡献
  - 每个 batch 双前向（with/without text）
```

### 4.3 消融实验设计

| 实验 | PWAM | L_t2v | L_nec | Enriched Text | Emb Perturb | 版本 |
|------|------|-------|-------|---------------|-------------|------|
| A: v4 baseline | ❌ (additive CA) | ❌ | ❌ | ❌ | ❌ | v4 |
| B: +PWAM only | ✅ | ❌ | ❌ | ❌ | ❌ | v5a |
| C: +PWAM +L_t2v | ✅ | ✅ | ❌ | ❌ | ❌ | v5b |
| D: +PWAM +L_t2v +L_nec | ✅ | ✅ | ✅ | ❌ | ❌ | v5c |
| E: +Enriched Text | ✅ | ✅ | ✅ | ✅ | ❌ | v5d |
| F: Full (all modules) | ✅ | ✅ | ✅ | ✅ | ✅ | v5 |

每个实验记录 `Dice(with text)` 和 `Dice(without text)`，核心指标是 `Delta = Dice(with) - Dice(without)`。

## 5. 文件变更清单

| 文件 | 变更类型 | 内容 |
|------|---------|------|
| `models/textmamba3d.py` | 修改 | 替换 MultiScalePixelTextAttention → PWAM3D |
| `models/pwam.py` | 新建 | PWAM3D + FiLM Gate 实现 |
| `losses/text_voxel_loss.py` | 新建 | TextToVoxelLoss (CRIS-style per-voxel BCE) |
| `losses/text_necessity_loss.py` | 新建 | TextNecessityLoss (hinge on dice delta) |
| `models/embedding_perturbation.py` | 新建 | EmbeddingPerturbation (simplified RLEG) |
| `data/text_enrichment.py` | 新建 | LLM prompt 模板 + 离线生成脚本 |
| `data/enriched_texts/` | 新建 | 369 个增强文本文件 |
| `train.py` | 修改 | 双前向逻辑 + L_nec 调度 |
| `configs/textbrats_v5.yaml` | 新建 | v5 配置（λ₁, λ₂, warmup epochs 等） |
| `losses/__init__.py` | 修改 | 注册新 loss |

## 6. 技术选型理由

| 决策 | 选择 | 理由 | 备选方案 | 为何不选 |
|------|------|------|---------|---------|
| 注入机制 | PWAM (LAVT) | 乘性融合 + 门控残差，架构无关 | Text-as-Query (SEEM) | 需要重写 decoder，改动量大 |
| 对齐 loss | Per-voxel BCE (CRIS) | 简单有效，无需正负样本采样 | InfoNCE contrastive | 需要大 batch，3D 医学不适合 |
| 文本增强 | LLM 知识注入 | 引入超越 mask 的临床知识 | Paraphrase | 不增加信息量，只换措辞 |
| Embedding 增强 | Gaussian + learned directions | 369 样本不足以训练 diffusion | 完整 RLEG | 需预训练 diffusion model |
| Text-necessity | Hinge loss on dice delta | 直接优化目标指标 | Dropout regularization | 不提供显式训练信号 |

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| PWAM 在 Mamba 中 cross-attention 计算量大 | 低 | Stage 1-3 token 数 4K-500，远小于 Stage 0 的 32K | 已排除 Stage 0 |
| 双前向训练过慢 | 中 | 训练时间 ×1.5~2 | L_nec 仅 Phase B 启用；text-masked forward 可复用 encoder 缓存 |
| LLM 生成文本含幻觉 | 中 | 错误的分级/鉴别信息 | 限制为从 mask 形态可推导的知识；人工抽查 10% |
| Enriched text 过长超出 max_len=256 | 低 | 截断丢失信息 | 增大 max_len 到 512；或只 concat 最关键部分 |
| 过拟合（新增 ~2.1M 参数） | 中 | 训练集过拟合 | 总参数 ~40M，增加 5.5%，可接受；PWAM 有 dropout 0.1 |
