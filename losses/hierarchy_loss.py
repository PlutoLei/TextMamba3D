"""Hierarchy Loss for BraTS tumor regions.

Soft constraint ensuring ET ⊆ TC ⊆ WT at every voxel.
From STPF (arXiv:2507.08574), extracted as standalone loss term.

BraTS regions from softmax:
  WT = class 1 + 2 + 3 (all tumor)
  TC = class 1 + 3 (tumor core)
  ET = class 3 (enhancing tumor)

Hierarchy: p(ET) <= p(TC) <= p(WT) at every voxel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchyLoss(nn.Module):
    """Penalize violations of ET <= TC <= WT hierarchy."""

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, D, H, W] raw logits or softmax probabilities

        Returns:
            scalar hierarchy violation penalty
        """
        probs = F.softmax(pred, dim=1) if pred.requires_grad else pred

        # BraTS region probabilities
        p_et = probs[:, 3]              # ET = class 3
        p_tc = probs[:, 1] + probs[:, 3]  # TC = class 1 + 3
        p_wt = probs[:, 1] + probs[:, 2] + probs[:, 3]  # WT = 1 + 2 + 3

        # Hierarchy violations: ET > TC or TC > WT
        violation_et_tc = F.relu(p_et - p_tc)  # should be 0 if ET <= TC
        violation_tc_wt = F.relu(p_tc - p_wt)  # should be 0 if TC <= WT

        return (violation_et_tc.mean() + violation_tc_wt.mean())
