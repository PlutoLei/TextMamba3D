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
        x = torch.randn(1, 512, 48)
        out = ee(x)
        ratio = out.mean() / x.mean()
        assert 0.9 < ratio.item() < 1.2

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
