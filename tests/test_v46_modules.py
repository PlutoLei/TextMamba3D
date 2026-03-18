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


def test_text_scale_gate_output_shape():
    """TextScaleGate should output same shape as input."""
    from models.fusion import TextScaleGate

    B, L, D = 2, 512, 96
    raw = torch.randn(B, L, D)
    fused = torch.randn(B, L, D)

    gate = TextScaleGate(feat_dim=D)
    out = gate(raw, fused)
    assert out.shape == (B, L, D), f"Expected ({B},{L},{D}), got {out.shape}"


def test_text_scale_gate_init_near_fused():
    """At init, gate ≈ 0.88, so output ≈ 0.88*fused + 0.12*raw."""
    from models.fusion import TextScaleGate

    B, L, D = 2, 64, 192
    raw = torch.zeros(B, L, D)
    fused = torch.ones(B, L, D)

    gate = TextScaleGate(feat_dim=D)
    out = gate(raw, fused)
    # sigmoid(2) ≈ 0.8808
    assert 0.85 < out.mean().item() < 0.92, f"Expected ~0.88, got {out.mean():.4f}"


def test_text_scale_gate_bypass_when_same():
    """When raw == fused, output should equal both (gate*x + (1-gate)*x = x)."""
    from models.fusion import TextScaleGate

    B, L, D = 2, 64, 192
    x = torch.randn(B, L, D)

    gate = TextScaleGate(feat_dim=D)
    out = gate(x, x)
    assert torch.allclose(out, x, atol=1e-5), "When raw==fused, output should equal input"


def test_multi_scale_text_gate():
    """MultiScaleTextGate should apply gates at each scale independently."""
    from models.fusion import MultiScaleTextGate

    B = 2
    stage_dims = [96, 192, 384]
    raw_features = [torch.randn(B, 512, 96), torch.randn(B, 64, 192), torch.randn(B, 8, 384)]
    fused_features = [torch.randn(B, 512, 96), torch.randn(B, 64, 192), torch.randn(B, 8, 384)]

    gate = MultiScaleTextGate(stage_dims=stage_dims)
    outputs = gate(raw_features, fused_features)

    assert len(outputs) == 3
    for out, raw in zip(outputs, raw_features):
        assert out.shape == raw.shape
