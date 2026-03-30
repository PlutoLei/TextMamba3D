import numpy as np
import torch
from losses.boundary_loss import compute_distance_map, BoundaryLoss

def test_compute_distance_map_shape():
    mask = np.zeros((16, 16, 16), dtype=np.int64)
    mask[4:12, 4:12, 4:12] = 1
    mask[6:10, 6:10, 6:10] = 3
    dist = compute_distance_map(mask, num_classes=4)
    assert dist.shape == (4, 16, 16, 16)
    assert dist.dtype == np.float32

def test_compute_distance_map_signs():
    mask = np.zeros((16, 16, 16), dtype=np.int64)
    mask[4:12, 4:12, 4:12] = 1
    dist = compute_distance_map(mask, num_classes=4)
    assert dist[1, 8, 8, 8] < 0
    assert dist[1, 0, 0, 0] > 0
    assert dist[0, 0, 0, 0] < 0
    assert dist[0, 8, 8, 8] > 0

def test_compute_distance_map_empty_class():
    mask = np.zeros((16, 16, 16), dtype=np.int64)
    dist = compute_distance_map(mask, num_classes=4)
    assert dist[1].min() > 0
    assert dist[2].min() > 0
    assert dist[3].min() > 0

def test_boundary_loss_forward():
    pred = torch.softmax(torch.randn(2, 4, 8, 8, 8), dim=1)
    dist = torch.randn(2, 4, 8, 8, 8)
    loss_fn = BoundaryLoss(include_background=False)
    loss = loss_fn(pred, dist)
    assert loss.shape == ()
    assert torch.isfinite(loss)

def test_boundary_loss_perfect_prediction():
    mask = np.zeros((8, 8, 8), dtype=np.int64)
    mask[2:6, 2:6, 2:6] = 1
    dist = compute_distance_map(mask, num_classes=4)
    dist_t = torch.from_numpy(dist).unsqueeze(0)
    perfect = torch.zeros(1, 4, 8, 8, 8)
    perfect[0, 0] = torch.from_numpy((mask == 0).astype(np.float32))
    perfect[0, 1] = torch.from_numpy((mask == 1).astype(np.float32))
    random_pred = torch.softmax(torch.randn(1, 4, 8, 8, 8), dim=1)
    loss_fn = BoundaryLoss(include_background=False)
    loss_perfect = loss_fn(perfect, dist_t)
    loss_random = loss_fn(random_pred, dist_t)
    assert loss_perfect < loss_random
