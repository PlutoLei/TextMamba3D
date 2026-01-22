# tests/test_models.py
import torch
import pytest


class TestMambaBlock:
    def test_mamba_block_output_shape(self):
        """Test MambaBlock maintains input shape."""
        from models.mamba_block import MambaBlock

        block = MambaBlock(dim=96, d_state=16, d_conv=4, expand=2)
        x = torch.randn(2, 1000, 96)  # [B, L, D]
        out = block(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_mamba_layer_output_shape(self):
        """Test MambaLayer with multiple blocks."""
        from models.mamba_block import MambaLayer

        layer = MambaLayer(dim=96, depth=2, d_state=16)
        x = torch.randn(2, 1000, 96)
        out = layer(x)
        assert out.shape == x.shape
