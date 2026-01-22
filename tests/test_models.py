# tests/test_models.py
import torch
import pytest


class TestTextEncoder:
    def test_text_encoder_output_shape(self):
        """Test text Mamba encoder."""
        from models.text_encoder import TextMambaEncoder

        encoder = TextMambaEncoder(
            vocab_size=30522,
            embed_dim=256,
            max_len=128,
            depth=2,
        )
        # Simulated token ids
        input_ids = torch.randint(0, 30522, (2, 64))
        out = encoder(input_ids)

        assert out.shape == (2, 64, 256)

    def test_text_encoder_global_feature(self):
        """Test global feature extraction."""
        from models.text_encoder import TextMambaEncoder

        encoder = TextMambaEncoder(
            vocab_size=30522,
            embed_dim=256,
            max_len=128,
            depth=2,
        )
        input_ids = torch.randint(0, 30522, (2, 64))
        out = encoder(input_ids)
        global_feat = encoder.get_global_feature(out)

        assert global_feat.shape == (2, 256)


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


class TestEncoder3D:
    def test_patch_embed_3d_output_shape(self):
        """Test 3D patch embedding."""
        from models.encoder_3d import PatchEmbed3D

        embed = PatchEmbed3D(
            img_size=(96, 96, 96),
            patch_size=(4, 4, 4),
            in_channels=4,
            embed_dim=96,
        )
        x = torch.randn(2, 4, 96, 96, 96)  # [B, C, D, H, W]
        out = embed(x)
        # Expected: [B, (96/4)^3, 96] = [B, 13824, 96]
        expected_seq_len = (96 // 4) ** 3
        assert out.shape == (2, expected_seq_len, 96)

    def test_encoder_3d_output_shape(self):
        """Test full 3D Mamba encoder."""
        from models.encoder_3d import MambaEncoder3D

        encoder = MambaEncoder3D(
            img_size=(96, 96, 96),
            in_channels=4,
            embed_dim=96,
            depths=[2, 2],
            patch_size=(4, 4, 4),
        )
        x = torch.randn(2, 4, 96, 96, 96)
        features = encoder(x)

        # Should return multi-scale features
        assert isinstance(features, list)
        assert len(features) == 2


class TestFusion:
    def test_mamba_fusion_output_shape(self):
        """Test Mamba fusion module."""
        from models.fusion import MambaFusion

        fusion = MambaFusion(
            img_dim=192,
            text_dim=256,
            hidden_dim=192,
            depth=2,
        )

        img_feat = torch.randn(2, 1000, 192)   # [B, N, D_img]
        text_feat = torch.randn(2, 64, 256)    # [B, M, D_text]

        out = fusion(img_feat, text_feat)

        # Output should match image feature shape
        assert out.shape == img_feat.shape


class TestDecoder3D:
    def test_decoder_output_shape(self):
        """Test 3D Mamba decoder."""
        from models.decoder_3d import MambaDecoder3D

        decoder = MambaDecoder3D(
            img_size=(96, 96, 96),
            patch_size=(4, 4, 4),
            out_channels=4,
            embed_dim=96,
            depths=[2, 2, 2, 2],
        )

        # Simulated encoder features (4 stages)
        features = [
            torch.randn(2, 24*24*24, 96),    # Stage 1
            torch.randn(2, 12*12*12, 192),   # Stage 2
            torch.randn(2, 6*6*6, 384),      # Stage 3
            torch.randn(2, 3*3*3, 768),      # Stage 4 (bottleneck)
        ]

        out = decoder(features)

        # Output should be [B, out_channels, D, H, W]
        assert out.shape == (2, 4, 96, 96, 96)


class TestTextMamba3D:
    def test_full_model_forward(self):
        """Test complete TextMamba3D model."""
        from models.textmamba3d import TextMamba3D

        model = TextMamba3D(
            img_size=(96, 96, 96),
            in_channels=4,
            out_channels=4,
            embed_dim=96,
            depths=[2, 2, 2, 2],
            text_embed_dim=256,
            text_depth=2,
        )

        img = torch.randn(2, 4, 96, 96, 96)
        text_ids = torch.randint(0, 30522, (2, 64))

        out = model(img, text_ids)

        assert out.shape == (2, 4, 96, 96, 96)

    def test_model_get_features_for_contrastive(self):
        """Test feature extraction for contrastive loss."""
        from models.textmamba3d import TextMamba3D

        model = TextMamba3D(
            img_size=(96, 96, 96),
            in_channels=4,
            out_channels=4,
            embed_dim=96,
            depths=[2, 2, 2, 2],
            text_embed_dim=256,
            text_depth=2,
        )

        img = torch.randn(2, 4, 96, 96, 96)
        text_ids = torch.randint(0, 30522, (2, 64))

        out, img_feat, text_feat = model(img, text_ids, return_features=True)

        assert img_feat.shape == (2, 256)  # Global image feature
        assert text_feat.shape == (2, 256)  # Global text feature
