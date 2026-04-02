import torch
from models.fusion import SequentialCrossAttention

def test_seqca_nonzero_init():
    seqca = SequentialCrossAttention(feat_dim=96, text_dim=256, num_heads=4)
    w = seqca.i2t_out.weight
    assert w.abs().sum() > 0, "i2t_out.weight should not be zero-initialized"
    assert w.abs().mean() < 0.1, "i2t_out.weight should be small magnitude"

def test_seqca_output_near_identity():
    torch.manual_seed(42)
    seqca = SequentialCrossAttention(feat_dim=96, text_dim=256, num_heads=4)
    x = torch.randn(2, 64, 96)
    text = torch.randn(2, 16, 256)
    out = seqca(x, text)
    # With small init, output should be close to input (residual dominates)
    diff = (out - x).abs().mean()
    assert diff < 1.0, f"Output should be near-identity at init, diff={diff}"
