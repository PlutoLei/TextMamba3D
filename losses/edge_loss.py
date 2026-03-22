# losses/edge_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    """Edge-enhanced loss for sharper boundaries."""

    def __init__(self):
        super().__init__()
        self.register_buffer('sobel_kernels', self._create_sobel_kernels())

    def _create_sobel_kernels(self) -> torch.Tensor:
        kernels = torch.zeros(3, 1, 3, 3, 3, dtype=torch.float32)
        neg_plane = torch.tensor(
            [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
            dtype=torch.float32,
        )
        pos_plane = -neg_plane

        kernels[0, 0, 0, :, :] = neg_plane
        kernels[0, 0, 2, :, :] = pos_plane

        kernels[1, 0, :, 0, :] = neg_plane
        kernels[1, 0, :, 2, :] = pos_plane

        kernels[2, 0, :, :, 0] = neg_plane
        kernels[2, 0, :, :, 2] = pos_plane

        return kernels / 16.0

    def get_edge_mask(self, target: torch.Tensor) -> torch.Tensor:
        target_float = target.float().unsqueeze(1)
        grads = F.conv3d(target_float, self.sobel_kernels, padding=1)
        edge = torch.sqrt(torch.clamp_min(grads.square().sum(dim=1, keepdim=True), 0.0))
        edge = edge / torch.clamp_min(edge.amax(dim=(-3, -2, -1), keepdim=True), 1e-4)
        return edge

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return only the extra edge-weighted penalty (no base CE).

        This avoids double-counting CE when used with CombinedLoss,
        which already has a standalone CE term.
        """
        edge_mask = self.get_edge_mask(target)  # [B, 1, D, H, W]
        edge_weight_map = edge_mask.squeeze(1)  # [B, D, H, W]
        loss = F.cross_entropy(pred.float(), target, reduction='none')  # [B, D, H, W]
        edge_bonus = (loss * edge_weight_map).mean()
        return edge_bonus
