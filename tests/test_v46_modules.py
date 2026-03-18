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


def test_decoder_with_cross_scale_skip():
    """Decoder with cross-scale skip should produce same output shape."""
    from models.decoder_3d import MambaDecoder3D

    B = 1
    # img_size=(64,64,64), patch_size=(4,4,4) => base_spatial=(16,16,16)
    # stage0: dim=48, L=16^3=4096; stage1: dim=96, L=8^3=512;
    # stage2: dim=192, L=4^3=64; stage3 (bottleneck): dim=384, L=2^3=8
    decoder = MambaDecoder3D(
        img_size=(64, 64, 64),
        patch_size=(4, 4, 4),
        out_channels=4,
        embed_dim=48,
        depths=[2, 2, 2, 2],
        use_cross_scale_skip=True,
    )

    features = [
        torch.randn(B, 4096, 48),
        torch.randn(B, 512, 96),
        torch.randn(B, 64, 192),
        torch.randn(B, 8, 384),
    ]

    out = decoder(features)
    assert out.shape == (B, 4, 64, 64, 64), f"Expected (1,4,64,64,64), got {out.shape}"


def test_decoder_backward_compat():
    """Decoder without cross-scale skip should work identically to v4.5."""
    from models.decoder_3d import MambaDecoder3D

    B = 1
    decoder = MambaDecoder3D(
        img_size=(64, 64, 64),
        patch_size=(4, 4, 4),
        out_channels=4,
        embed_dim=48,
        depths=[2, 2, 2, 2],
        use_cross_scale_skip=False,
    )

    features = [
        torch.randn(B, 4096, 48),
        torch.randn(B, 512, 96),
        torch.randn(B, 64, 192),
        torch.randn(B, 8, 384),
    ]

    out = decoder(features)
    assert out.shape == (B, 4, 64, 64, 64)


def test_textmamba3d_with_text_gate():
    """TextMamba3D with use_text_gate=True should produce same output shape."""
    from models.textmamba3d import TextMamba3D

    model = TextMamba3D(
        img_size=(32, 32, 32),
        in_channels=4,
        out_channels=4,
        embed_dim=48,
        depths=[1, 1, 1, 1],
        text_embed_dim=64,
        text_max_len=16,
        text_depth=1,
        use_pretrained_text=False,
        use_text_gate=True,
        use_cross_scale_skip=True,
    )

    img = torch.randn(1, 4, 32, 32, 32)
    out = model(img, use_text=False)
    assert out.shape == (1, 4, 32, 32, 32)


def test_textmamba3d_v46_backward_compat():
    """TextMamba3D without new features should have no text_gate."""
    from models.textmamba3d import TextMamba3D

    model = TextMamba3D(
        img_size=(32, 32, 32),
        in_channels=4,
        out_channels=4,
        embed_dim=48,
        depths=[1, 1, 1, 1],
        text_embed_dim=64,
        text_max_len=16,
        text_depth=1,
        use_pretrained_text=False,
        use_text_gate=False,
        use_cross_scale_skip=False,
    )

    assert model.text_gate is None
    assert not model.decoder.use_cross_scale_skip


@pytest.mark.slow
def test_v46_full_forward_pass():
    """Full V4.6 model forward pass with text input."""
    from models.textmamba3d import TextMamba3D

    model = TextMamba3D(
        img_size=(32, 32, 32),
        in_channels=4,
        out_channels=4,
        embed_dim=48,
        depths=[1, 1, 1, 1],
        text_embed_dim=64,
        text_max_len=32,
        text_depth=1,
        use_text_gate=True,
        use_cross_scale_skip=True,
        use_pretrained_text=False,
    )
    model.eval()

    img = torch.randn(1, 4, 32, 32, 32)
    text_ids = torch.randint(0, 1000, (1, 32))
    mask = torch.ones(1, 32)

    with torch.no_grad():
        out_text = model(img, text_ids, mask, use_text=True)
        assert out_text.shape == (1, 4, 32, 32, 32)

        out_no_text = model(img, use_text=False)
        assert out_no_text.shape == (1, 4, 32, 32, 32)

    assert not torch.allclose(out_text, out_no_text, atol=1e-3), \
        "Text and no-text outputs should differ"


@pytest.mark.slow
def test_v46_gradient_flow():
    """Verify gradients flow through both new modules."""
    from models.textmamba3d import TextMamba3D

    model = TextMamba3D(
        img_size=(32, 32, 32),
        in_channels=4,
        out_channels=4,
        embed_dim=48,
        depths=[1, 1, 1, 1],
        text_embed_dim=64,
        text_max_len=16,
        text_depth=1,
        use_text_gate=True,
        use_cross_scale_skip=True,
        use_pretrained_text=False,
    )

    img = torch.randn(1, 4, 32, 32, 32)
    text_ids = torch.randint(0, 1000, (1, 16))
    mask = torch.ones(1, 16)

    out = model(img, text_ids, mask, use_text=True)
    loss = out.sum()
    loss.backward()

    # TextScaleGate gradients
    for i, gate in enumerate(model.text_gate.gates):
        assert gate.gate_proj.weight.grad is not None, f"No gradient for text gate {i}"
        assert gate.gate_proj.weight.grad.abs().sum() > 0, f"Zero gradient for text gate {i}"

    # CrossScaleSkipAttention gradients
    for i, attn in enumerate(model.decoder.cross_scale_attns):
        assert attn.pseudo_query.grad is not None, f"No gradient for pseudo_query {i}"


@pytest.mark.slow
def test_v46_with_contrastive_features():
    """V4.6 model should return features for contrastive loss."""
    from models.textmamba3d import TextMamba3D

    model = TextMamba3D(
        img_size=(32, 32, 32),
        in_channels=4,
        out_channels=4,
        embed_dim=48,
        depths=[1, 1, 1, 1],
        text_embed_dim=64,
        text_max_len=16,
        text_depth=1,
        use_text_gate=True,
        use_cross_scale_skip=True,
        use_pretrained_text=False,
    )

    img = torch.randn(1, 4, 32, 32, 32)
    text_ids = torch.randint(0, 1000, (1, 16))
    mask = torch.ones(1, 16)

    seg, img_g, text_g, pix = model(img, text_ids, mask, return_features=True)
    assert seg.shape == (1, 4, 32, 32, 32)
    assert img_g is not None
    assert text_g is not None
    assert pix is not None
