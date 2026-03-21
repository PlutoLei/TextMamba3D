# Mamba-3: Improved Sequence Modeling using State Space Principles

**Paper:** Mamba-3: Improved Sequence Modeling using State Space Principles
**Authors:** Aakash Lahoti*, Kevin Y. Li*, Berlin Chen*, Caitlin Wang*, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu
**Venue:** arXiv 2026 (2603.15569)
**Analysis Depth:** Standard

---

## Quick Summary / 快速概要

**EN:** Mamba-3 introduces three core improvements to the SSM framework: (1) exponential-trapezoidal discretization that implicitly replaces short causal convolutions, (2) complex-valued state transitions enabling state-tracking capabilities Mamba-2 lacks, and (3) MIMO formulation that increases decoding FLOPs up to 4x with similar wall-clock latency. At 1.5B scale, Mamba-3 MIMO improves average downstream accuracy by +2.2 over Transformers and achieves comparable perplexity to Mamba-2 with half the state size.

**ZH:** Mamba-3 对 SSM 框架引入三大核心改进：(1) 指数梯形离散化，隐式替代短因果卷积；(2) 复数值状态转移，解决 Mamba-2 缺失的状态追踪能力；(3) MIMO 公式将解码 FLOPs 提升至 4 倍而延迟几乎不变。在 1.5B 规模下，Mamba-3 MIMO 平均下游准确率比 Transformer 高 2.2 个百分点，用一半的状态大小达到 Mamba-2 同等困惑度。

## Files / 文件

| File | Description |
|------|-------------|
| `Mamba3_arXiv2026.pdf` | Original paper |

## Key Contributions / 核心贡献

1. **Exponential-Trapezoidal Discretization** -- Second-order discretization that implicitly introduces width-2 data-dependent convolution on SSM input, making external short causal convolution optional
2. **Complex-valued SSM** -- Enables state-tracking capabilities (parity, modular arithmetic) that real-valued SSMs cannot solve; equivalent to data-dependent RoPE on B, C projections
3. **MIMO Formulation** -- Switches state update from outer-product to matrix multiplication, increasing arithmetic intensity to utilize idle GPU tensor cores during decoding without increasing latency

## Relevance to TextMamba3D / 与 TextMamba3D 的关联

Mamba-3 is the direct successor to the SSM backbone used in TextMamba3D (`mamba_ssm.Mamba`). The complex-valued SSM is especially relevant: TextMamba3D's cross-scan along 3 spatial axes (DHW, HWD, WDH) involves spatial rotations that real-valued states cannot encode, potentially contributing to the TC Dice regression observed in V4.4→V4.5. Upgrading to Mamba-3's complex-valued SSM is a candidate V5.0 direction for improving cross-scan spatial coherence. The hybrid architecture validation (Mamba:Attention = 5:1) also aligns with TextMamba3D's "predominantly Mamba + frozen PubMedBERT" design.

---

*Added on 2026-03-19. GitHub: https://github.com/state-spaces/mamba*
