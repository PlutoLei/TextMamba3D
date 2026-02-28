# losses/__init__.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dice_loss import DiceLoss, BRATS_CLASS_WEIGHTS
from .edge_loss import EdgeLoss
from .contrastive_loss import ContrastiveLoss


class CombinedLoss(nn.Module):
    """Combined loss for TextMamba3D.

    Supports class-weighted Dice and CE to handle BraTS class imbalance.
    Skips expensive computation when a loss weight is zero.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        edge_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        temperature: float = 0.07,
        class_weights: list[float] | None = None,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.edge_weight = edge_weight
        self.contrastive_weight = contrastive_weight

        self.dice_loss = DiceLoss(class_weights=class_weights)
        self.edge_loss = EdgeLoss()
        self.contrastive_loss = ContrastiveLoss(temperature)

        # CE class weights (normalized for cross_entropy)
        if class_weights is not None:
            self.register_buffer(
                'ce_class_weights',
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.ce_class_weights = None

    def _zero(self, pred: torch.Tensor) -> torch.Tensor:
        """Return a zero scalar on the same device as pred."""
        return pred.new_zeros(())

    def forward(self, pred, target, img_feat=None, text_feat=None) -> dict:
        losses = {}

        # Only compute losses with nonzero weights
        losses['dice'] = self.dice_loss(pred, target) if self.dice_weight else self._zero(pred)
        losses['ce'] = (
            F.cross_entropy(pred, target, weight=self.ce_class_weights)
            if self.ce_weight else self._zero(pred)
        )
        losses['edge'] = self.edge_loss(pred, target) if self.edge_weight else self._zero(pred)

        if self.contrastive_weight and img_feat is not None and text_feat is not None:
            losses['contrastive'] = self.contrastive_loss(img_feat, text_feat)
        else:
            losses['contrastive'] = self._zero(pred)

        losses['total'] = (
            self.dice_weight * losses['dice'] +
            self.ce_weight * losses['ce'] +
            self.edge_weight * losses['edge'] +
            self.contrastive_weight * losses['contrastive']
        )
        return losses
