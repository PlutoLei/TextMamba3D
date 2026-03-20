# tests/test_mamba3.py
"""Tests for Mamba-3 SSM integration (V5.0)."""
import pytest
import torch


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
# _create_ssm factory
# =========================================================================

class TestCreateSSM:
    """Test _create_ssm factory with use_mamba3 flag."""

    def test_create_ssm_mamba1_default(self):
        """Default _create_ssm returns Mamba (or fallback MLP)."""
        from models.mamba_block import _create_ssm
        ssm = _create_ssm(dim=96, d_state=16, d_conv=4, expand=2)
        assert ssm is not None

    def test_create_ssm_mamba3_flag_false(self):
        """use_mamba3=False returns Mamba-1 SSM."""
        from models.mamba_block import _create_ssm
        ssm = _create_ssm(dim=96, d_state=16, d_conv=4, expand=2, use_mamba3=False)
        assert ssm is not None

    @requires_mamba3
    @requires_cuda
    def test_create_ssm_mamba3_returns_mamba3(self):
        """use_mamba3=True returns Mamba3 instance."""
        from mamba_ssm import Mamba3
        from models.mamba_block import _create_ssm
        ssm = _create_ssm(dim=48, d_state=64, d_conv=4, expand=2,
                          use_mamba3=True, headdim=48)
        assert isinstance(ssm, Mamba3)

    @requires_mamba3
    @requires_cuda
    def test_create_ssm_mamba3_forward_shape(self):
        """Mamba3 SSM preserves (B, L, D) shape."""
        from models.mamba_block import _create_ssm
        ssm = _create_ssm(dim=48, d_state=64, d_conv=4, expand=2,
                          use_mamba3=True, headdim=48).cuda()
        x = torch.randn(2, 100, 48).cuda()
        out = ssm(x)
        assert out.shape == (2, 100, 48)

    def test_auto_headdim(self):
        """_auto_headdim picks valid headdim for common dimensions."""
        from models.mamba_block import _auto_headdim
        assert 96 % _auto_headdim(96) == 0
        assert 192 % _auto_headdim(192) == 0
        assert 384 % _auto_headdim(384) == 0
        assert 768 % _auto_headdim(768) == 0

    def test_auto_headdim_prime_returns_1(self):
        """_auto_headdim returns 1 for prime d_inner that no preferred value divides."""
        from models.mamba_block import _auto_headdim
        assert _auto_headdim(97) == 1  # prime, no preferred hd divides it

    def test_invalid_headdim_raises(self, monkeypatch):
        """Explicit headdim that doesn't divide d_inner raises ValueError."""
        import models.mamba_block as mb
        # Monkeypatch Mamba3 availability so we reach the validation code
        monkeypatch.setattr(mb, 'MAMBA3_AVAILABLE', True)
        monkeypatch.setattr(mb, 'Mamba3', lambda **kw: None)  # dummy
        with pytest.raises(ValueError, match="does not divide d_inner"):
            mb._create_ssm(dim=48, d_state=16, d_conv=4, expand=2,
                           use_mamba3=True, headdim=37)  # 37 does not divide 96

    def test_use_mamba3_without_library_raises(self, monkeypatch):
        """use_mamba3=True when Mamba3 not installed raises ImportError."""
        import models.mamba_block as mb
        monkeypatch.setattr(mb, 'MAMBA3_AVAILABLE', False)
        with pytest.raises(ImportError, match="use_mamba3=True but"):
            mb._create_ssm(dim=48, d_state=16, d_conv=4, expand=2,
                           use_mamba3=True)

    def test_mamba3_constructor_error_propagates(self, monkeypatch):
        """Mamba3 constructor errors propagate instead of silent fallback."""
        import models.mamba_block as mb

        def bad_mamba3(**kw):
            raise RuntimeError("CUDA kernel compilation failed")

        monkeypatch.setattr(mb, 'MAMBA3_AVAILABLE', True)
        monkeypatch.setattr(mb, 'Mamba3', bad_mamba3)
        with pytest.raises(RuntimeError, match="CUDA kernel"):
            mb._create_ssm(dim=48, d_state=16, d_conv=4, expand=2,
                           use_mamba3=True, headdim=48)

    def test_mamba1_narrow_exception(self, monkeypatch):
        """Mamba-1 fallback only catches TypeError/RuntimeError, not all Exception."""
        import models.mamba_block as mb

        def bad_mamba(**kw):
            raise ValueError("unexpected value error")

        monkeypatch.setattr(mb, 'MAMBA_AVAILABLE', True)
        monkeypatch.setattr(mb, 'Mamba', bad_mamba)
        # ValueError should NOT be caught by the narrowed except clause
        with pytest.raises(ValueError, match="unexpected value error"):
            mb._create_ssm(dim=48, d_state=16, d_conv=4, expand=2,
                           use_mamba3=False)


# =========================================================================
# CrossScan blocks with Mamba3
# =========================================================================

class TestCrossScanMamba3D:
    """Test CrossScanBiMamba3DBlock/Layer with Mamba3."""

    @requires_mamba3
    @requires_cuda
    def test_crossscan_block_mamba3_shape(self):
        """CrossScanBiMamba3DBlock with Mamba3 preserves shape."""
        from models.mamba_block import CrossScanBiMamba3DBlock
        block = CrossScanBiMamba3DBlock(
            dim=48, spatial_dims=(8, 8, 8), d_state=64,
            use_mamba3=True, headdim=48,
        ).cuda()
        x = torch.randn(1, 512, 48).cuda()
        out = block(x)
        assert out.shape == (1, 512, 48)

    @requires_mamba3
    @requires_cuda
    def test_crossscan_layer_mamba3_shape(self):
        """CrossScanBiMamba3DLayer with Mamba3 preserves shape."""
        from models.mamba_block import CrossScanBiMamba3DLayer
        layer = CrossScanBiMamba3DLayer(
            dim=48, depth=2, spatial_dims=(8, 8, 8), d_state=64,
            use_mamba3=True, headdim=48,
        ).cuda()
        x = torch.randn(1, 512, 48).cuda()
        out = layer(x)
        assert out.shape == (1, 512, 48)

    @requires_mamba3
    @requires_cuda
    def test_crossscan_block_mamba3_gradient(self):
        """Mamba3 CrossScan block produces valid gradients."""
        from models.mamba_block import CrossScanBiMamba3DBlock
        block = CrossScanBiMamba3DBlock(
            dim=48, spatial_dims=(4, 4, 4), d_state=64,
            use_mamba3=True, headdim=48,
        ).cuda()
        x = torch.randn(1, 64, 48, requires_grad=True).cuda()
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_crossscan_block_default_no_mamba3(self):
        """Default CrossScanBiMamba3DBlock uses Mamba-1 (backward compat)."""
        from models.mamba_block import CrossScanBiMamba3DBlock
        block = CrossScanBiMamba3DBlock(dim=48, spatial_dims=(4, 4, 4), d_state=16)
        x = torch.randn(1, 64, 48)
        out = block(x)
        assert out.shape == (1, 64, 48)


# =========================================================================
# Full model with Mamba3
# =========================================================================

class TestTextMamba3DWithMamba3:
    """End-to-end TextMamba3D with Mamba3 backend."""

    @requires_mamba3
    @requires_cuda
    def test_full_model_mamba3_forward(self):
        """Full TextMamba3D forward pass with Mamba3."""
        from models.textmamba3d import TextMamba3D
        model = TextMamba3D(
            img_size=(32, 32, 32), in_channels=4, out_channels=4,
            embed_dim=48, depths=[1, 1, 1, 1], patch_size=(4, 4, 4),
            d_state=64, use_pretrained_text=False,
            use_mamba3=True, headdim=48,
        ).cuda()
        img = torch.randn(1, 4, 32, 32, 32).cuda()
        text = torch.randint(0, 100, (1, 16)).cuda()
        out = model(img, text)
        assert out.shape == (1, 4, 32, 32, 32)

    @requires_mamba3
    @requires_cuda
    def test_full_model_mamba3_no_text(self):
        """Forward without text still works with Mamba3."""
        from models.textmamba3d import TextMamba3D
        model = TextMamba3D(
            img_size=(32, 32, 32), in_channels=4, out_channels=4,
            embed_dim=48, depths=[1, 1, 1, 1], patch_size=(4, 4, 4),
            d_state=64, use_pretrained_text=False,
            use_mamba3=True, headdim=48,
        ).cuda()
        img = torch.randn(1, 4, 32, 32, 32).cuda()
        out = model(img, use_text=False)
        assert out.shape == (1, 4, 32, 32, 32)

    @requires_mamba3
    @requires_cuda
    def test_full_model_mamba3_gradient_flow(self):
        """Gradients flow through entire Mamba3 model."""
        from models.textmamba3d import TextMamba3D
        model = TextMamba3D(
            img_size=(32, 32, 32), in_channels=4, out_channels=4,
            embed_dim=48, depths=[1, 1, 1, 1], patch_size=(4, 4, 4),
            d_state=64, use_pretrained_text=False,
            use_mamba3=True, headdim=48,
        ).cuda()
        img = torch.randn(1, 4, 32, 32, 32).cuda()
        text = torch.randint(0, 100, (1, 16)).cuda()
        out = model(img, text)
        loss = out.sum()
        loss.backward()
        enc_block = model.img_encoder.stages[0].blocks[0]
        ssm_params = list(enc_block.dhw_fwd.parameters())
        assert len(ssm_params) > 0
        assert ssm_params[0].grad is not None

    def test_full_model_mamba1_backward_compat(self):
        """TextMamba3D without use_mamba3 still works (backward compat)."""
        from models.textmamba3d import TextMamba3D
        model = TextMamba3D(
            img_size=(32, 32, 32), in_channels=4, out_channels=4,
            embed_dim=48, depths=[1, 1, 1, 1], patch_size=(4, 4, 4),
            d_state=16, use_pretrained_text=False,
        )
        img = torch.randn(1, 4, 32, 32, 32)
        text = torch.randint(0, 100, (1, 16))
        out = model(img, text)
        assert out.shape == (1, 4, 32, 32, 32)
