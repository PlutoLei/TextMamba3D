# tests/test_v46_modules.py
import torch
import pytest


def test_cross_scale_skip_attention_output_shape():
    """CrossScaleSkipAttention should output [B, L_target, D_target]."""
    from models.fusion import CrossScaleSkipAttention

    B = 2
    target = torch.randn(B, 64, 192)
    candidates = [
        torch.randn(B, 4096, 48),
        torch.randn(B, 512, 96),
        torch.randn(B, 64, 192),
    ]
    candidate_dims = [48, 96, 192]

    module = CrossScaleSkipAttention(
        target_dim=192,
        candidate_dims=candidate_dims,
    )

    out = module(target, candidates)
    assert out.shape == (B, 64, 192), f"Expected (2,64,192), got {out.shape}"


def test_cross_scale_skip_attention_zero_init():
    """At initialization, output should be near-zero (identity-preserving)."""
    from models.fusion import CrossScaleSkipAttention

    B = 2
    target = torch.randn(B, 64, 192)
    candidates = [torch.randn(B, 512, 96), torch.randn(B, 64, 192)]

    module = CrossScaleSkipAttention(
        target_dim=192,
        candidate_dims=[96, 192],
    )

    out = module(target, candidates)
    assert out.abs().max() < 1e-5, f"Expected near-zero output at init, got max={out.abs().max()}"


def test_cross_scale_skip_attention_single_candidate():
    """Should work with a single candidate (last decoder stage)."""
    from models.fusion import CrossScaleSkipAttention

    B = 2
    target = torch.randn(B, 4096, 48)
    candidates = [torch.randn(B, 4096, 48)]

    module = CrossScaleSkipAttention(target_dim=48, candidate_dims=[48])
    out = module(target, candidates)
    assert out.shape == (B, 4096, 48)


def test_cross_scale_skip_attention_gradients():
    """Verify gradients flow correctly through the module."""
    from models.fusion import CrossScaleSkipAttention

    module = CrossScaleSkipAttention(target_dim=64, candidate_dims=[32, 64])
    target = torch.randn(2, 16, 64)
    candidates = [
        torch.randn(2, 8, 32, requires_grad=True),
        torch.randn(2, 16, 64, requires_grad=True),
    ]
    out = module(target, candidates)
    loss = out.sum()
    loss.backward()
    assert candidates[0].grad is not None
    assert candidates[1].grad is not None
    assert not torch.isnan(module.out_proj.weight.grad).any()
    assert module.pseudo_query.grad is not None
