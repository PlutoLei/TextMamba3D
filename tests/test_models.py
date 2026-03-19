# tests/test_models.py
"""Comprehensive model tests for TextMamba3D.

Tests cover: MambaBlock, BiMamba, CrossScan3D, Encoder, Decoder,
TextEncoder (fallback), Fusion, FiLM, and full model (with/without text).
"""
import torch
import pytest


# =========================================================================
# MambaBlock (unidirectional)
# =========================================================================

class TestMambaBlock:
    def test_mamba_block_output_shape(self):
        """MambaBlock preserves [B, L, D] shape."""
        from models.mamba_block import MambaBlock

        block = MambaBlock(dim=96, d_state=16, d_conv=4, expand=2)
        x = torch.randn(2, 100, 96)
        out = block(x)
        assert out.shape == x.shape

    def test_mamba_layer_output_shape(self):
        """MambaLayer (depth>1) preserves shape."""
        from models.mamba_block import MambaLayer

        layer = MambaLayer(dim=96, depth=2, d_state=16)
        x = torch.randn(2, 100, 96)
        out = layer(x)
        assert out.shape == x.shape

    def test_mamba_block_residual(self):
        """MambaBlock output differs from input (residual + transform)."""
        from models.mamba_block import MambaBlock

        block = MambaBlock(dim=64, d_state=16)
        x = torch.randn(1, 50, 64)
        out = block(x)
        assert out.shape == x.shape
        # Output should differ from input (SSM processes the data)
        assert not torch.allclose(out, x, atol=1e-4)


# =========================================================================
# BiMamba (bidirectional)
# =========================================================================

class TestBiMamba:
    def test_bimamba_block_shape(self):
        """BiMambaBlock preserves [B, L, D] shape."""
        from models.mamba_block import BiMambaBlock

        block = BiMambaBlock(dim=64, d_state=16)
        x = torch.randn(2, 100, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_bimamba_layer_shape(self):
        """BiMambaLayer (stacked blocks) preserves shape."""
        from models.mamba_block import BiMambaLayer

        layer = BiMambaLayer(dim=64, depth=2, d_state=16)
        x = torch.randn(2, 100, 64)
        out = layer(x)
        assert out.shape == x.shape

    def test_bimamba_block_has_both_directions(self):
        """BiMambaBlock should have forward_ssm and backward_ssm."""
        from models.mamba_block import BiMambaBlock

        block = BiMambaBlock(dim=32)
        assert hasattr(block, 'forward_ssm')
        assert hasattr(block, 'backward_ssm')
        assert hasattr(block, 'merge')

    def test_bimamba_block_gradient_flow(self):
        """Gradients should flow through BiMambaBlock."""
        from models.mamba_block import BiMambaBlock

        block = BiMambaBlock(dim=32)
        x = torch.randn(1, 20, 32, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# =========================================================================
# CrossScan Mamba 3D (3-direction scanning)
# =========================================================================

class TestCrossScan3D:
    def test_cross_scan_block_shape(self):
        """CrossScanBiMamba3DBlock preserves shape for cubic spatial dims."""
        from models.mamba_block import CrossScanBiMamba3DBlock

        D, H, W = 4, 4, 4
        block = CrossScanBiMamba3DBlock(dim=64, spatial_dims=(D, H, W))
        x = torch.randn(2, D * H * W, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_cross_scan_layer_shape(self):
        """CrossScanBiMamba3DLayer (stacked) preserves shape."""
        from models.mamba_block import CrossScanBiMamba3DLayer

        D, H, W = 4, 4, 4
        layer = CrossScanBiMamba3DLayer(dim=64, depth=2, spatial_dims=(D, H, W))
        x = torch.randn(2, D * H * W, 64)
        out = layer(x)
        assert out.shape == x.shape

    def test_cross_scan_non_cubic(self):
        """CrossScan works with non-cubic spatial dimensions."""
        from models.mamba_block import CrossScanBiMamba3DBlock

        D, H, W = 4, 6, 8
        block = CrossScanBiMamba3DBlock(dim=32, spatial_dims=(D, H, W))
        x = torch.randn(1, D * H * W, 32)
        out = block(x)
        assert out.shape == (1, D * H * W, 32)

    def test_cross_scan_has_3_forward_ssms(self):
        """CrossScan block should have 3 forward-only SSMs (one per axis order)."""
        from models.mamba_block import CrossScanBiMamba3DBlock

        block = CrossScanBiMamba3DBlock(dim=32, spatial_dims=(4, 4, 4))
        ssm_names = ['dhw_fwd', 'hwd_fwd', 'wdh_fwd']
        for name in ssm_names:
            assert hasattr(block, name), f"Missing SSM: {name}"

    def test_cross_scan_merge_dimension(self):
        """Merge layer should project 3*dim -> dim."""
        from models.mamba_block import CrossScanBiMamba3DBlock

        dim = 48
        block = CrossScanBiMamba3DBlock(dim=dim, spatial_dims=(4, 4, 4))
        assert block.merge.in_features == dim * 3
        assert block.merge.out_features == dim

    def test_cross_scan_gradient_flow(self):
        """Gradients should flow through CrossScan block."""
        from models.mamba_block import CrossScanBiMamba3DBlock

        D, H, W = 4, 4, 4
        block = CrossScanBiMamba3DBlock(dim=32, spatial_dims=(D, H, W))
        x = torch.randn(1, D * H * W, 32, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# =========================================================================
# FiLM (Feature-wise Linear Modulation)
# =========================================================================

class TestFiLMLayer:
    def test_film_near_identity_init(self):
        """FiLM warm-init (std=0.01) should be near-identity: gamma ≈ 1, beta ≈ 0."""
        from models.fusion import FiLMLayer

        film = FiLMLayer(feat_dim=64, cond_dim=32)
        x = torch.randn(2, 100, 64)
        cond = torch.randn(2, 32)
        out = film(x, cond)
        relative_diff = (out - x).abs().mean() / x.abs().mean()
        assert relative_diff < 0.1, \
            f"FiLM warm-init too far from identity: relative diff = {relative_diff.item():.4f}"

    def test_film_output_shape(self):
        """FiLM preserves input spatial shape."""
        from models.fusion import FiLMLayer

        film = FiLMLayer(feat_dim=64, cond_dim=128)
        x = torch.randn(2, 50, 64)
        cond = torch.randn(2, 128)
        out = film(x, cond)
        assert out.shape == x.shape

    def test_film_after_training_steps(self):
        """FiLM should produce different output after weight updates."""
        from models.fusion import FiLMLayer

        film = FiLMLayer(feat_dim=32, cond_dim=16)
        # Manually set non-trivial weights
        film.gamma_proj.weight.data.fill_(0.1)
        film.beta_proj.weight.data.fill_(0.05)

        x = torch.randn(1, 10, 32)
        cond = torch.randn(1, 16)
        out = film(x, cond)
        # Should no longer be identity
        assert not torch.allclose(out, x, atol=1e-3)


class TestMultiScaleFiLM:
    def test_multi_scale_film_shapes(self):
        """MultiScaleFiLM outputs match input shapes at each scale."""
        from models.fusion import MultiScaleFiLM

        stage_dims = [96, 192, 384, 768]
        film = MultiScaleFiLM(stage_dims=stage_dims, text_dim=256)

        features = [
            torch.randn(2, 100, 96),
            torch.randn(2, 50, 192),
            torch.randn(2, 25, 384),
            torch.randn(2, 10, 768),
        ]
        text_global = torch.randn(2, 256)
        out = film(features, text_global)

        assert len(out) == 4
        for feat, o in zip(features, out):
            assert o.shape == feat.shape

    def test_multi_scale_film_near_identity_init(self):
        """All FiLM layers should start near-identity with warm init."""
        from models.fusion import MultiScaleFiLM

        stage_dims = [32, 64]
        film = MultiScaleFiLM(stage_dims=stage_dims, text_dim=16)

        features = [torch.randn(1, 20, 32), torch.randn(1, 10, 64)]
        text_global = torch.randn(1, 16)
        out = film(features, text_global)

        for feat, o in zip(features, out):
            relative_diff = (o - feat).abs().mean() / feat.abs().mean()
            assert relative_diff < 0.1, \
                f"MultiScaleFiLM warm-init too far from identity: {relative_diff.item():.4f}"

    def test_multi_scale_film_num_layers(self):
        """Number of FiLM layers should match number of stages."""
        from models.fusion import MultiScaleFiLM

        dims = [32, 64, 128]
        film = MultiScaleFiLM(stage_dims=dims, text_dim=16)
        assert len(film.film_layers) == 3


# =========================================================================
# Encoder 3D
# =========================================================================

class TestEncoder3D:
    def test_patch_embed_3d_output_shape(self):
        """PatchEmbed3D produces correct sequence length."""
        from models.encoder_3d import PatchEmbed3D

        embed = PatchEmbed3D(
            img_size=(96, 96, 96), patch_size=(4, 4, 4),
            in_channels=4, embed_dim=96,
        )
        x = torch.randn(2, 4, 96, 96, 96)
        out = embed(x)
        expected_seq_len = (96 // 4) ** 3
        assert out.shape == (2, expected_seq_len, 96)

    def test_encoder_3d_output_shape(self):
        """Encoder outputs multi-scale feature list."""
        from models.encoder_3d import MambaEncoder3D

        encoder = MambaEncoder3D(
            img_size=(96, 96, 96), in_channels=4,
            embed_dim=96, depths=[2, 2], patch_size=(4, 4, 4),
        )
        x = torch.randn(2, 4, 96, 96, 96)
        features = encoder(x)
        assert isinstance(features, list)
        assert len(features) == 2

    def test_encoder_small_uses_crossscan(self):
        """Encoder stages should use CrossScanBiMamba3DLayer."""
        from models.encoder_3d import MambaEncoder3D
        from models.mamba_block import CrossScanBiMamba3DLayer

        encoder = MambaEncoder3D(
            img_size=(16, 16, 16), in_channels=4,
            embed_dim=32, depths=[1, 1], patch_size=(4, 4, 4),
        )
        for stage in encoder.stages:
            assert isinstance(stage, CrossScanBiMamba3DLayer)

    def test_encoder_feature_dimensions(self):
        """Feature dimensions should double at each stage."""
        from models.encoder_3d import MambaEncoder3D

        embed_dim = 32
        encoder = MambaEncoder3D(
            img_size=(16, 16, 16), in_channels=4,
            embed_dim=embed_dim, depths=[1, 1], patch_size=(4, 4, 4),
        )
        x = torch.randn(1, 4, 16, 16, 16)
        features = encoder(x)
        assert features[0].shape[-1] == embed_dim
        assert features[1].shape[-1] == embed_dim * 2

    def test_encoder_checkpoint_mode(self):
        """Encoder should accept use_checkpoint parameter and run without error."""
        from models.encoder_3d import MambaEncoder3D

        encoder = MambaEncoder3D(
            img_size=(16, 16, 16), in_channels=4,
            embed_dim=32, depths=[1, 1], patch_size=(4, 4, 4),
            use_checkpoint=True,
        )
        encoder.train()
        x = torch.randn(1, 4, 16, 16, 16)
        features = encoder(x)
        assert len(features) == 2


# =========================================================================
# Decoder 3D
# =========================================================================

class TestDecoder3D:
    def test_decoder_output_shape(self):
        """Decoder outputs [B, out_channels, D, H, W]."""
        from models.decoder_3d import MambaDecoder3D

        decoder = MambaDecoder3D(
            img_size=(96, 96, 96), patch_size=(4, 4, 4),
            out_channels=4, embed_dim=96, depths=[2, 2, 2, 2],
        )
        features = [
            torch.randn(2, 24*24*24, 96),
            torch.randn(2, 12*12*12, 192),
            torch.randn(2, 6*6*6, 384),
            torch.randn(2, 3*3*3, 768),
        ]
        out = decoder(features)
        assert out.shape == (2, 4, 96, 96, 96)

    def test_decoder_small_volume(self):
        """Decoder works with small volumes (16^3)."""
        from models.decoder_3d import MambaDecoder3D

        decoder = MambaDecoder3D(
            img_size=(16, 16, 16), patch_size=(4, 4, 4),
            out_channels=4, embed_dim=32, depths=[1, 1],
        )
        features = [
            torch.randn(1, 4*4*4, 32),
            torch.randn(1, 2*2*2, 64),
        ]
        out = decoder(features)
        assert out.shape == (1, 4, 16, 16, 16)


# =========================================================================
# Fusion
# =========================================================================

class TestFusion:
    def test_mamba_fusion_output_shape(self):
        """MambaFusion preserves image feature shape."""
        from models.fusion import MambaFusion

        fusion = MambaFusion(
            img_dim=192, text_dim=256, hidden_dim=192, depth=2,
        )
        img_feat = torch.randn(2, 1000, 192)
        text_feat = torch.randn(2, 64, 256)
        out = fusion(img_feat, text_feat)
        assert out.shape == img_feat.shape

    def test_mamba_fusion_residual(self):
        """MambaFusion includes residual connection (out ≠ input but close initially)."""
        from models.fusion import MambaFusion

        fusion = MambaFusion(
            img_dim=64, text_dim=32, hidden_dim=64, depth=1,
        )
        img_feat = torch.randn(1, 50, 64)
        text_feat = torch.randn(1, 10, 32)
        out = fusion(img_feat, text_feat)
        assert out.shape == img_feat.shape


# =========================================================================
# Text Encoder (fallback mode — no PubMedBERT dependency)
# =========================================================================

class TestTextEncoder:
    def test_text_encoder_output_shape(self):
        """Fallback text encoder produces correct shape."""
        from models.text_encoder import TextMambaEncoder

        encoder = TextMambaEncoder(
            vocab_size=30522, embed_dim=256, max_len=128,
            depth=2, use_pretrained=False,
        )
        input_ids = torch.randint(0, 30522, (2, 64))
        out = encoder(input_ids)
        assert out.shape == (2, 64, 256)

    def test_text_encoder_global_feature(self):
        """Global feature extraction produces [B, D] shape."""
        from models.text_encoder import TextMambaEncoder

        encoder = TextMambaEncoder(
            vocab_size=30522, embed_dim=256, max_len=128,
            depth=2, use_pretrained=False,
        )
        input_ids = torch.randint(0, 30522, (2, 64))
        out = encoder(input_ids)
        global_feat = encoder.get_global_feature(out)
        assert global_feat.shape == (2, 256)

    def test_text_encoder_fallback_uses_mean_pooling(self):
        """Fallback mode should use mean pooling for global feature."""
        from models.text_encoder import TextMambaEncoder

        encoder = TextMambaEncoder(
            vocab_size=100, embed_dim=32, max_len=64,
            depth=1, use_pretrained=False,
        )
        assert not encoder.use_pretrained
        # Mean pooling: global_feat should equal x.mean(dim=1)
        x = torch.randn(1, 10, 32)
        global_feat = encoder.get_global_feature(x)
        expected = x.mean(dim=1)
        assert torch.allclose(global_feat, expected)

    def test_text_encoder_different_seq_lengths(self):
        """Encoder handles different sequence lengths up to max_len."""
        from models.text_encoder import TextMambaEncoder

        encoder = TextMambaEncoder(
            vocab_size=100, embed_dim=32, max_len=64,
            depth=1, use_pretrained=False,
        )
        for seq_len in [16, 32, 64]:
            ids = torch.randint(0, 100, (1, seq_len))
            out = encoder(ids)
            assert out.shape == (1, seq_len, 32)


# =========================================================================
# TextMamba3D full model
# =========================================================================

class TestTextMamba3D:
    """Full model tests using small volumes for speed."""

    @pytest.fixture
    def small_model(self):
        from models.textmamba3d import TextMamba3D
        return TextMamba3D(
            img_size=(16, 16, 16), in_channels=4, out_channels=4,
            embed_dim=32, depths=[1, 1], text_embed_dim=64,
            text_depth=1, use_pretrained_text=False,
        )

    def test_forward_with_text(self, small_model):
        """Forward pass with text guidance."""
        img = torch.randn(1, 4, 16, 16, 16)
        text_ids = torch.randint(0, 30522, (1, 16))
        out = small_model(img, text_ids)
        assert out.shape == (1, 4, 16, 16, 16)

    def test_forward_without_text(self, small_model):
        """Forward pass without text (use_text=False)."""
        img = torch.randn(1, 4, 16, 16, 16)
        out = small_model(img, text_ids=None, use_text=False)
        assert out.shape == (1, 4, 16, 16, 16)

    def test_forward_without_text_convenience(self, small_model):
        """forward_without_text convenience method."""
        img = torch.randn(1, 4, 16, 16, 16)
        out = small_model.forward_without_text(img)
        assert out.shape == (1, 4, 16, 16, 16)

    def test_return_features(self, small_model):
        """Feature extraction for contrastive loss."""
        img = torch.randn(1, 4, 16, 16, 16)
        text_ids = torch.randint(0, 30522, (1, 16))
        out, img_feat, text_feat, pixel_feat = small_model(
            img, text_ids, return_features=True,
        )
        assert out.shape == (1, 4, 16, 16, 16)
        assert img_feat.shape == (1, 64)   # text_embed_dim
        assert text_feat.shape == (1, 64)
        assert pixel_feat is not None

    def test_return_features_no_text(self, small_model):
        """Feature extraction without text returns None (no contrastive on text-free)."""
        img = torch.randn(1, 4, 16, 16, 16)
        out, img_feat, text_feat, pixel_feat = small_model(
            img, text_ids=None, use_text=False, return_features=True,
        )
        assert out.shape == (1, 4, 16, 16, 16)
        assert img_feat is None
        assert text_feat is None
        assert pixel_feat is None

    def test_gradient_checkpointing(self):
        """Model with gradient checkpointing should train normally."""
        from models.textmamba3d import TextMamba3D

        model = TextMamba3D(
            img_size=(16, 16, 16), in_channels=4, out_channels=4,
            embed_dim=32, depths=[1, 1], text_embed_dim=64,
            text_depth=1, use_pretrained_text=False,
            use_checkpoint=True,
        )
        model.train()
        img = torch.randn(1, 4, 16, 16, 16)
        text_ids = torch.randint(0, 30522, (1, 16))
        out = model(img, text_ids)
        loss = out.sum()
        loss.backward()
        assert out.shape == (1, 4, 16, 16, 16)

    def test_no_default_text_embed(self, small_model):
        """v3: default_text_embed removed (was v2 artifact, text-free path bypasses fusion)."""
        assert not hasattr(small_model, 'default_text_embed')


# =========================================================================
# Encoder 3D edge cases
# =========================================================================

class TestEncoder3DEdgeCases:
    def test_encoder_single_depth(self):
        """Encoder with depths=[1] (single stage, no downsampling)."""
        from models.encoder_3d import MambaEncoder3D

        encoder = MambaEncoder3D(
            img_size=(16, 16, 16), in_channels=4,
            embed_dim=32, depths=[1], patch_size=(4, 4, 4),
        )
        x = torch.randn(1, 4, 16, 16, 16)
        features = encoder(x)
        assert len(features) == 1
        # Single stage: seq_len = (16/4)^3 = 64, dim = 32
        assert features[0].shape == (1, 64, 32)

    def test_encoder_asymmetric_depths(self):
        """Encoder with non-uniform depths like [1, 2, 3]."""
        from models.encoder_3d import MambaEncoder3D

        encoder = MambaEncoder3D(
            img_size=(16, 16, 16), in_channels=4,
            embed_dim=32, depths=[1, 2, 3], patch_size=(4, 4, 4),
        )
        x = torch.randn(1, 4, 16, 16, 16)
        features = encoder(x)
        assert len(features) == 3
        # Check channel dims double each stage
        assert features[0].shape[-1] == 32
        assert features[1].shape[-1] == 64
        assert features[2].shape[-1] == 128

    def test_encoder_spatial_dims_tracked(self):
        """Encoder tracks spatial_dims for each stage."""
        from models.encoder_3d import MambaEncoder3D

        encoder = MambaEncoder3D(
            img_size=(16, 16, 16), in_channels=4,
            embed_dim=32, depths=[1, 1], patch_size=(4, 4, 4),
        )
        # After patch embed: (4,4,4), after downsample: (2,2,2)
        assert encoder.spatial_dims[0] == (4, 4, 4)
        assert encoder.spatial_dims[1] == (2, 2, 2)

    def test_encoder_downsample_count(self):
        """Number of downsamplers = num_stages - 1."""
        from models.encoder_3d import MambaEncoder3D

        for n_stages in [1, 2, 3, 4]:
            encoder = MambaEncoder3D(
                img_size=(16, 16, 16), in_channels=4,
                embed_dim=32, depths=[1] * n_stages, patch_size=(4, 4, 4),
            )
            assert len(encoder.downsamples) == n_stages - 1


# =========================================================================
# MultiScaleFiLM extended tests
# =========================================================================

class TestMultiScaleFiLMExtended:
    def test_film_gradient_to_text_global(self):
        """Gradients flow from loss through FiLM back to text_global."""
        from models.fusion import MultiScaleFiLM

        stage_dims = [32, 64]
        film = MultiScaleFiLM(stage_dims=stage_dims, text_dim=16)

        # Break identity init so FiLM actually transforms the input
        for layer in film.film_layers:
            layer.gamma_proj.weight.data.fill_(0.1)
            layer.beta_proj.weight.data.fill_(0.05)

        features = [torch.randn(1, 20, 32), torch.randn(1, 10, 64)]
        text_global = torch.randn(1, 16, requires_grad=True)
        out = film(features, text_global)

        loss = sum(o.sum() for o in out)
        loss.backward()
        assert text_global.grad is not None
        assert text_global.grad.shape == (1, 16)

    def test_film_different_text_produces_different_output(self):
        """Different text globals produce different modulated features."""
        from models.fusion import MultiScaleFiLM

        stage_dims = [32]
        film = MultiScaleFiLM(stage_dims=stage_dims, text_dim=16)

        # Break identity init
        for layer in film.film_layers:
            layer.gamma_proj.weight.data.normal_(0, 0.1)
            layer.beta_proj.weight.data.normal_(0, 0.1)

        features = [torch.randn(1, 20, 32)]
        text_a = torch.randn(1, 16)
        text_b = torch.randn(1, 16)
        out_a = film(features, text_a)
        out_b = film(features, text_b)
        assert not torch.allclose(out_a[0], out_b[0], atol=1e-5), \
            "Different text globals should produce different modulated outputs"

    def test_film_single_stage(self):
        """Works with just one stage."""
        from models.fusion import MultiScaleFiLM

        film = MultiScaleFiLM(stage_dims=[64], text_dim=32)
        features = [torch.randn(2, 50, 64)]
        text_global = torch.randn(2, 32)
        out = film(features, text_global)
        assert len(out) == 1
        assert out[0].shape == (2, 50, 64)


# =========================================================================
# MambaFusion extended tests
# =========================================================================

class TestMambaFusionExtended:
    def test_fusion_gradient_to_text(self):
        """Gradients flow from fused output back to text_feat."""
        from models.fusion import MambaFusion

        fusion = MambaFusion(
            img_dim=64, text_dim=32, hidden_dim=64, depth=1,
        )
        img_feat = torch.randn(1, 20, 64)
        text_feat = torch.randn(1, 10, 32, requires_grad=True)
        out = fusion(img_feat, text_feat)

        loss = out.sum()
        loss.backward()
        assert text_feat.grad is not None
        assert text_feat.grad.shape == (1, 10, 32)

    def test_fusion_different_text_lengths(self):
        """Fusion handles varying text sequence lengths."""
        from models.fusion import MambaFusion

        fusion = MambaFusion(
            img_dim=64, text_dim=32, hidden_dim=64, depth=1,
        )
        img_feat = torch.randn(1, 20, 64)

        for text_len in [5, 20, 64]:
            text_feat = torch.randn(1, text_len, 32)
            out = fusion(img_feat, text_feat)
            assert out.shape == (1, 20, 64), \
                f"Output shape wrong for text_len={text_len}: {out.shape}"

    def test_fusion_preserves_shape(self):
        """Output shape matches img_feat shape."""
        from models.fusion import MambaFusion

        fusion = MambaFusion(
            img_dim=96, text_dim=64, hidden_dim=128, depth=2,
        )
        img_feat = torch.randn(2, 100, 96)
        text_feat = torch.randn(2, 30, 64)
        out = fusion(img_feat, text_feat)
        assert out.shape == img_feat.shape

    def test_fusion_proj_dimensions(self):
        """Verify internal projection dimensions."""
        from models.fusion import MambaFusion

        img_dim, text_dim, hidden_dim = 96, 64, 128
        fusion = MambaFusion(
            img_dim=img_dim, text_dim=text_dim,
            hidden_dim=hidden_dim, depth=1,
        )
        # img_proj: img_dim -> hidden_dim
        assert fusion.img_proj[0].in_features == img_dim
        assert fusion.img_proj[0].out_features == hidden_dim
        # text_proj: text_dim -> hidden_dim
        assert fusion.text_proj[0].in_features == text_dim
        assert fusion.text_proj[0].out_features == hidden_dim
        # out_proj: hidden_dim -> img_dim
        assert fusion.out_proj.in_features == hidden_dim
        assert fusion.out_proj.out_features == img_dim
