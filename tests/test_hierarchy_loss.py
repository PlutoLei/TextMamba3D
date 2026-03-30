import torch
from losses.hierarchy_loss import HierarchyLoss

def test_hierarchy_no_violation():
    loss_fn = HierarchyLoss()
    pred = torch.zeros(1, 4, 4, 4, 4)
    pred[0, 0] = 0.1
    pred[0, 2] = 2.0
    pred[0, 1] = 1.0
    pred[0, 3] = 0.5
    loss = loss_fn(pred)
    assert loss.item() < 0.01

def test_hierarchy_with_violation():
    loss_fn = HierarchyLoss()
    # Pass pre-computed "probabilities" (requires_grad=False skips softmax).
    # Set channel 1 negative so p_tc = ch1 + ch3 < p_et = ch3, creating a
    # genuine ET > TC violation.
    pred = torch.zeros(1, 4, 4, 4, 4)
    pred[0, 0] = 0.1
    pred[0, 1] = -1.0   # negative makes p_tc < p_et
    pred[0, 2] = -1.0   # negative makes p_wt < p_tc possible too
    pred[0, 3] = 5.0
    loss = loss_fn(pred)
    assert loss.item() > 0.01

def test_hierarchy_grad_flows():
    loss_fn = HierarchyLoss()
    pred = torch.randn(2, 4, 4, 4, 4, requires_grad=True)
    loss = loss_fn(pred)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
