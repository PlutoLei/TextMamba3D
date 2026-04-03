import pytest
import torch
from models.concat_fusion import MambaConcatFusion, MultiScaleConcatFusion


def _has_mamba3():
    try:
        from mamba_ssm import Mamba3
        return True
    except ImportError:
        return False


requires_mamba3 = pytest.mark.skipif(
    not _has_mamba3(), reason="mamba_ssm with Mamba3 not installed"
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# =========================================================================
# CPU tests (use_mamba3=False falls back to MLP, tests fusion logic)
# =========================================================================

def test_concat_fusion_shape():
    fusion = MambaConcatFusion(feat_dim=96, text_dim=256, num_prompts=8,
                               use_mamba3=False)
    img = torch.randn(2, 512, 96)
    text = torch.randn(2, 32, 256)
    out = fusion(img, text)
    assert out.shape == img.shape


def test_compressor_text_matters():
    """TextPromptCompressor produces different prompts for different texts."""
    from models.concat_fusion import TextPromptCompressor
    torch.manual_seed(42)
    comp = TextPromptCompressor(text_dim=256, feat_dim=96, num_prompts=8)
    text_a = torch.randn(2, 32, 256)
    text_b = torch.randn(2, 32, 256)
    prompts_a = comp(text_a)
    prompts_b = comp(text_b)
    diff = (prompts_a - prompts_b).abs().max().item()
    assert diff > 0.01, f"Different text must produce different prompts, diff={diff}"


@requires_mamba3
@requires_cuda
def test_concat_fusion_text_matters():
    """With real Mamba SSM, different text inputs produce different outputs.

    Note: MLP fallback processes tokens independently (no cross-token
    interaction), so text prompts can't influence image tokens. This test
    requires the real Mamba SSM for sequential scan to work.
    """
    torch.manual_seed(42)
    fusion = MambaConcatFusion(feat_dim=96, text_dim=256, num_prompts=8,
                               use_mamba3=True).cuda()
    img = torch.randn(2, 512, 96).cuda()
    text_a = torch.randn(2, 32, 256).cuda()
    text_b = torch.randn(2, 32, 256).cuda()
    out_a = fusion(img, text_a)
    out_b = fusion(img, text_b)
    assert not torch.allclose(out_a, out_b, atol=1e-4)


def test_multiscale_concat_fusion():
    fusion = MultiScaleConcatFusion(
        stage_dims=[96, 192, 384], text_dim=256, num_prompts=8,
        use_mamba3=False,
    )
    features = [
        torch.randn(2, 4096, 96),
        torch.randn(2, 512, 192),
        torch.randn(2, 64, 384),
    ]
    text = torch.randn(2, 32, 256)
    out = fusion(features, text)
    assert len(out) == 3
    for o, f in zip(out, features):
        assert o.shape == f.shape


def test_multiscale_accepts_num_heads_kwarg():
    # Must not raise TypeError when num_heads is passed (drop-in compatibility)
    fusion = MultiScaleConcatFusion(
        stage_dims=[96, 192], text_dim=256, num_heads=4,
        use_mamba3=False,
    )
    assert len(fusion.layers) == 2


def test_concat_fusion_gradient():
    fusion = MambaConcatFusion(feat_dim=96, text_dim=256, num_prompts=8,
                               use_mamba3=False)
    img = torch.randn(2, 64, 96, requires_grad=True)
    text = torch.randn(2, 16, 256, requires_grad=True)
    out = fusion(img, text)
    out.sum().backward()
    assert img.grad is not None
    assert text.grad is not None


def test_concat_fusion_small_init():
    """Output should be near-identity at initialization (residual dominates)."""
    torch.manual_seed(42)
    fusion = MambaConcatFusion(feat_dim=96, text_dim=256, num_prompts=8,
                               use_mamba3=False)
    img = torch.randn(2, 64, 96)
    text = torch.randn(2, 16, 256)
    out = fusion(img, text)
    diff = (out - img).abs().mean()
    assert diff < 1.0, f"Output should be near-identity at init, diff={diff}"


# =========================================================================
# GPU + Mamba3 tests (only run on training server)
# =========================================================================

@requires_mamba3
@requires_cuda
def test_concat_fusion_mamba3_shape():
    fusion = MambaConcatFusion(feat_dim=96, text_dim=256, num_prompts=8,
                               use_mamba3=True).cuda()
    img = torch.randn(2, 512, 96).cuda()
    text = torch.randn(2, 32, 256).cuda()
    out = fusion(img, text)
    assert out.shape == img.shape


@requires_mamba3
@requires_cuda
def test_concat_fusion_mamba3_gradient():
    fusion = MambaConcatFusion(feat_dim=96, text_dim=256, num_prompts=8,
                               use_mamba3=True).cuda()
    img = torch.randn(2, 64, 96, requires_grad=True).cuda()
    text = torch.randn(2, 16, 256, requires_grad=True).cuda()
    out = fusion(img, text)
    out.sum().backward()
    assert img.grad is not None
    assert text.grad is not None
