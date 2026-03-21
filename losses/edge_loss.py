# losses/edge_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    """Edge-enhanced loss for sharper boundaries."""

    def __init__(self):
        super().__init__()
        self.register_buffer('sobel_x', self._create_sobel_kernel(0))
        self.register_buffer('sobel_y', self._create_sobel_kernel(1))
        self.register_buffer('sobel_z', self._create_sobel_kernel(2))

    def _create_sobel_kernel(self, axis: int) -> torch.Tensor:
        kernel = torch.zeros(1, 1, 3, 3, 3)
        if axis == 0:
            kernel[0, 0, 0, :, :] = torch.tensor([[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]])
            kernel[0, 0, 2, :, :] = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        elif axis == 1:
            kernel[0, 0, :, 0, :] = torch.tensor([[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]])
            kernel[0, 0, :, 2, :] = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        else:
            kernel[0, 0, :, :, 0] = torch.tensor([[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]])
            kernel[0, 0, :, :, 2] = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        return kernel / 16.0

    def get_edge_mask(self, target: torch.Tensor) -> torch.Tensor:
        target_float = target.float().unsqueeze(1)

        # Sobel kernels may be bf16 after model.to(bf16) — force fp32 for conv
        gx = F.conv3d(target_float, self.sobel_x.float(), padding=1)
        gy = F.conv3d(target_float, self.sobel_y.float(), padding=1)
        gz = F.conv3d(target_float, self.sobel_z.float(), padding=1)
        edge = torch.sqrt(torch.clamp_min(gx**2 + gy**2 + gz**2, 0.0))
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
