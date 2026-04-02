import torch
from losses.alignment_loss import InfoNCELoss

def test_infonce_positive_pairs():
    loss_fn = InfoNCELoss(temperature=0.07)
    feats = torch.randn(4, 256)
    loss_same = loss_fn(feats, feats)
    loss_rand = loss_fn(feats, torch.randn(4, 256))
    assert loss_same < loss_rand

def test_infonce_gradient_flows():
    loss_fn = InfoNCELoss(temperature=0.07)
    img = torch.randn(4, 256, requires_grad=True)
    txt = torch.randn(4, 256, requires_grad=True)
    loss = loss_fn(img, txt)
    loss.backward()
    assert img.grad is not None
    assert txt.grad is not None

def test_infonce_batch_size_one():
    loss_fn = InfoNCELoss(temperature=0.07)
    img = torch.randn(1, 256)
    txt = torch.randn(1, 256)
    loss = loss_fn(img, txt)
    assert loss.isfinite()
