import torch
import numpy as np
from losses import CombinedLoss
from losses.boundary_loss import compute_distance_map

def test_combined_loss_with_boundary():
    criterion = CombinedLoss(
        use_boundary=True,
        boundary_max_weight=1.0,
        use_hierarchy=True,
        hierarchy_weight=0.1,
    )
    pred = torch.randn(2, 4, 8, 8, 8)
    target = torch.randint(0, 4, (2, 8, 8, 8))
    dist_maps = []
    for i in range(2):
        dm = compute_distance_map(target[i].numpy(), num_classes=4)
        dist_maps.append(torch.from_numpy(dm))
    dist_maps = torch.stack(dist_maps)
    result = criterion(pred, target, distance_map=dist_maps, epoch=50, total_epochs=80)
    assert 'total' in result
    assert 'boundary' in result
    assert 'hierarchy' in result
    assert torch.isfinite(result['total'])

def test_combined_loss_alpha_annealing():
    criterion = CombinedLoss(use_boundary=True, boundary_max_weight=1.0)
    pred = torch.randn(1, 4, 8, 8, 8)
    target = torch.randint(0, 4, (1, 8, 8, 8))
    dm = compute_distance_map(target[0].numpy(), num_classes=4)
    dist_maps = torch.from_numpy(dm).unsqueeze(0)
    r0 = criterion(pred, target, distance_map=dist_maps, epoch=0, total_epochs=80)
    r80 = criterion(pred, target, distance_map=dist_maps, epoch=80, total_epochs=80)
    # epoch 0: alpha=0, boundary contribution should be 0
    assert abs(r0.get('boundary', torch.tensor(0.0)).item()) < 1e-6
    # epoch 80: alpha=1.0, boundary should be nonzero
    assert abs(r80['boundary'].item()) > 1e-6

def test_combined_loss_without_boundary():
    criterion = CombinedLoss()
    pred = torch.randn(2, 4, 8, 8, 8)
    target = torch.randint(0, 4, (2, 8, 8, 8))
    result = criterion(pred, target)
    assert 'total' in result
    assert 'boundary' not in result
    assert 'hierarchy' not in result
