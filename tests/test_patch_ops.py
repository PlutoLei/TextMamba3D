# tests/test_patch_ops.py
"""Tests for PatchMerging3D and PatchExpanding3D.

PatchMerging3D downsamples 2x in each spatial dimension (halves D,H,W; doubles C).
PatchExpanding3D upsamples 2x in each spatial dimension (doubles D,H,W; maps to out_dim).
"""
import torch
import pytest


# =========================================================================
# PatchMerging3D
# =========================================================================

class TestPatchMerging3D:
    """PatchMerging3D: downsample 2x in each spatial dimension."""

    def test_output_shape_basic(self):
        """PatchMerging3D halves spatial dims and doubles channels."""
        from models.encoder_3d import PatchMerging3D

        D, H, W = 8, 8, 8
        dim = 32
        merge = PatchMerging3D(dim=dim, spatial_dims=(D, H, W))
        x = torch.randn(2, D * H * W, dim)
        out = merge(x)

        expected_seq_len = (D // 2) * (H // 2) * (W // 2)  # 4*4*4 = 64
        expected_dim = dim * 2  # 64
        assert out.shape == (2, expected_seq_len, expected_dim)

    def test_output_shape_non_cubic(self):
        """Works with non-cubic spatial dimensions."""
        from models.encoder_3d import PatchMerging3D

        D, H, W = 6, 8, 4
        dim = 48
        merge = PatchMerging3D(dim=dim, spatial_dims=(D, H, W))
        x = torch.randn(1, D * H * W, dim)
        out = merge(x)

        expected_seq_len = (D // 2) * (H // 2) * (W // 2)  # 3*4*2 = 24
        expected_dim = dim * 2  # 96
        assert out.shape == (1, expected_seq_len, expected_dim)

    def test_gradient_flow(self):
        """Gradients flow through PatchMerging3D."""
        from models.encoder_3d import PatchMerging3D

        D, H, W = 4, 4, 4
        dim = 32
        merge = PatchMerging3D(dim=dim, spatial_dims=(D, H, W))
        x = torch.randn(1, D * H * W, dim, requires_grad=True)
        out = merge(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_reduction_layer_sizes(self):
        """Internal reduction: 8*dim -> 2*dim."""
        from models.encoder_3d import PatchMerging3D

        dim = 48
        merge = PatchMerging3D(dim=dim, spatial_dims=(4, 4, 4))
        assert merge.reduction.in_features == 8 * dim
        assert merge.reduction.out_features == 2 * dim

    def test_batch_independence(self):
        """Different batch elements produce different outputs."""
        from models.encoder_3d import PatchMerging3D

        D, H, W = 4, 4, 4
        dim = 32
        merge = PatchMerging3D(dim=dim, spatial_dims=(D, H, W))

        x = torch.randn(2, D * H * W, dim)
        # Ensure the two batch elements are different
        x[0] = x[0] * 2.0 + 1.0
        out = merge(x)
        assert not torch.allclose(out[0], out[1], atol=1e-5), \
            "Different batch inputs should produce different outputs"


# =========================================================================
# PatchExpanding3D
# =========================================================================

class TestPatchExpanding3D:
    """PatchExpanding3D: upsample 2x in each spatial dimension."""

    def test_output_shape_basic(self):
        """PatchExpanding3D doubles spatial dims."""
        from models.decoder_3d import PatchExpanding3D

        D, H, W = 3, 3, 3
        dim = 64
        out_dim = 32
        expand = PatchExpanding3D(dim=dim, out_dim=out_dim, spatial_dims=(D, H, W))
        x = torch.randn(2, D * H * W, dim)
        out = expand(x)

        expected_seq_len = (D * 2) * (H * 2) * (W * 2)  # 6*6*6 = 216
        assert out.shape == (2, expected_seq_len, out_dim)

    def test_output_shape_non_cubic(self):
        """Works with non-cubic spatial dimensions."""
        from models.decoder_3d import PatchExpanding3D

        D, H, W = 2, 3, 4
        dim = 96
        out_dim = 48
        expand = PatchExpanding3D(dim=dim, out_dim=out_dim, spatial_dims=(D, H, W))
        x = torch.randn(1, D * H * W, dim)
        out = expand(x)

        expected_seq_len = (D * 2) * (H * 2) * (W * 2)  # 4*6*8 = 192
        assert out.shape == (1, expected_seq_len, out_dim)

    def test_gradient_flow(self):
        """Gradients flow through PatchExpanding3D."""
        from models.decoder_3d import PatchExpanding3D

        D, H, W = 3, 3, 3
        dim = 64
        out_dim = 32
        expand = PatchExpanding3D(dim=dim, out_dim=out_dim, spatial_dims=(D, H, W))
        x = torch.randn(1, D * H * W, dim, requires_grad=True)
        out = expand(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_expand_layer_sizes(self):
        """Internal expand: dim -> 8*out_dim."""
        from models.decoder_3d import PatchExpanding3D

        dim = 64
        out_dim = 32
        expand = PatchExpanding3D(dim=dim, out_dim=out_dim, spatial_dims=(3, 3, 3))
        assert expand.expand.in_features == dim
        assert expand.expand.out_features == 8 * out_dim

    def test_merging_expanding_roundtrip_shape(self):
        """PatchMerging followed by PatchExpanding restores original sequence length."""
        from models.encoder_3d import PatchMerging3D
        from models.decoder_3d import PatchExpanding3D

        D, H, W = 6, 6, 6
        dim = 32
        seq_len = D * H * W  # 216

        # Merge: (6,6,6) dim=32 -> (3,3,3) dim=64
        merge = PatchMerging3D(dim=dim, spatial_dims=(D, H, W))
        # Expand: (3,3,3) dim=64 -> (6,6,6) dim=32
        expand = PatchExpanding3D(
            dim=dim * 2, out_dim=dim,
            spatial_dims=(D // 2, H // 2, W // 2),
        )

        x = torch.randn(1, seq_len, dim)
        merged = merge(x)
        restored = expand(merged)
        assert restored.shape == x.shape, \
            f"Roundtrip shape mismatch: {restored.shape} vs {x.shape}"
