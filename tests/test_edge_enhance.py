# tests/test_edge_enhance.py
import torch


class TestEdgeEnhance3D:
    def test_output_shape(self):
        from models.edge_enhance import EdgeEnhance3D
        ee = EdgeEnhance3D(channels=96, spatial_dims=(32, 32, 32))
        x = torch.randn(2, 32*32*32, 96)
        out = ee(x)
        assert out.shape == x.shape

    def test_near_identity_at_init(self):
        from models.edge_enhance import EdgeEnhance3D
        ee = EdgeEnhance3D(channels=48, spatial_dims=(8, 8, 8))
        x = torch.randn(1, 512, 48) + 5.0  # Shift away from zero for stable ratio
        out = ee(x)
        # With zero weights and bias=-3, conv outputs constant -3
        # sigmoid(-3) ~ 0.047, so out ~ x * 1.047
        assert torch.allclose(out, x * 1.047, atol=0.15)

    def test_backward_pass(self):
        from models.edge_enhance import EdgeEnhance3D
        ee = EdgeEnhance3D(channels=96, spatial_dims=(8, 8, 8))
        x = torch.randn(1, 512, 96, requires_grad=True)
        out = ee(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_different_stages(self):
        from models.edge_enhance import EdgeEnhance3D
        for channels, spatial in [(48, (32, 32, 32)), (96, (16, 16, 16)), (192, (8, 8, 8))]:
            N = spatial[0] * spatial[1] * spatial[2]
            ee = EdgeEnhance3D(channels=channels, spatial_dims=spatial)
            x = torch.randn(1, N, channels)
            out = ee(x)
            assert out.shape == (1, N, channels)

    def test_parameter_count(self):
        from models.edge_enhance import EdgeEnhance3D
        ee = EdgeEnhance3D(channels=192, spatial_dims=(8, 8, 8))
        num_params = sum(p.numel() for p in ee.parameters())
        assert num_params < 50000


class TestEdgeEnhanceIntegration:
    def test_decoder_with_edge_enhance(self):
        from models.decoder_3d import MambaDecoder3D

        decoder = MambaDecoder3D(
            img_size=(128, 128, 128),
            patch_size=(4, 4, 4),
            out_channels=4,
            embed_dim=48,
            depths=[2, 2, 2, 2],
            use_edge_enhance=True,
        )
        assert decoder.edge_enhances is not None
        assert len(decoder.edge_enhances) == 3

    def test_decoder_without_edge_enhance_backward_compatible(self):
        from models.decoder_3d import MambaDecoder3D

        decoder = MambaDecoder3D(
            img_size=(128, 128, 128),
            patch_size=(4, 4, 4),
            out_channels=4,
            embed_dim=48,
            depths=[2, 2, 2, 2],
        )
        assert decoder.edge_enhances is None
