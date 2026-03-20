# losses/__init__.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dice_loss import DiceLoss, BRATS_CLASS_WEIGHTS
from .edge_loss import EdgeLoss
from .contrastive_loss import ContrastiveLoss, ForegroundContrastiveLoss


class CombinedLoss(nn.Module):
    """Combined loss for TextMamba3D.

    Supports class-weighted Dice and CE to handle BraTS class imbalance.
    Skips expensive computation when a loss weight is zero.
    """

    # Deep supervision weights for aux heads (modify class constant to change)
    DS_WEIGHTS = [0.2, 0.1, 0.05]

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        edge_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        temperature: float = 0.07,
        feat_dim: int = 384,
        text_dim: int = 256,
        class_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.edge_weight = edge_weight
        self.contrastive_weight = contrastive_weight

        self.dice_loss = DiceLoss(class_weights=class_weights)
        self.edge_loss = EdgeLoss()
        self.contrastive_loss = ContrastiveLoss(temperature)
        self.fg_contrastive_loss = ForegroundContrastiveLoss(
            temperature=temperature,
            feat_dim=feat_dim,
            text_dim=text_dim,
        )

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

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        img_feat: torch.Tensor | None = None,
        text_feat: torch.Tensor | None = None,
        pixel_feat: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        mask_orig: torch.Tensor | None = None,
        aux_preds: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        losses = {}

        # All loss computation in fp32 (bf16 backward kernels missing for some ops)
        pred = pred.float()

        ce_w = self.ce_class_weights.float() if self.ce_class_weights is not None else None
        losses['dice'] = (
            self.dice_loss(pred, target) if self.dice_weight else self._zero(pred)
        )
        losses['ce'] = (
            F.cross_entropy(pred, target, weight=ce_w)
            if self.ce_weight else self._zero(pred)
        )
        losses['edge'] = (
            self.edge_loss(pred, target) if self.edge_weight else self._zero(pred)
        )

        contrastive_mask = mask if mask is not None else mask_orig
        if (
            self.contrastive_weight
            and pixel_feat is not None
            and text_feat is not None
            and contrastive_mask is not None
        ):
            losses['contrastive'] = self.fg_contrastive_loss(
                pixel_feat,
                text_feat,
                contrastive_mask,
            )
        elif self.contrastive_weight and img_feat is not None and text_feat is not None:
            losses['contrastive'] = self.contrastive_loss(img_feat, text_feat)
        else:
            losses['contrastive'] = self._zero(pred)

        total = (
            self.dice_weight * losses['dice'] +
            self.ce_weight * losses['ce'] +
            self.edge_weight * losses['edge'] +
            self.contrastive_weight * losses['contrastive']
        )
        # Clamp finite values only — NaN/Inf pass through for training loop detection
        losses['total'] = total.clamp(0.0, 20.0) if total.isfinite() else total

        # Deep supervision: aux heads at intermediate decoder stages
        if aux_preds:
            ds_loss = self._zero(pred)
            target_size = target.shape[1:]  # (D, H, W)
            for w, aux in zip(self.DS_WEIGHTS, aux_preds):
                aux_up = F.interpolate(aux, size=target_size, mode='trilinear', align_corners=False).float()
                aux_dice = self.dice_loss(aux_up, target)
                aux_ce = F.cross_entropy(aux_up, target, weight=ce_w)
                ds_loss = ds_loss + w * (aux_dice + aux_ce)
            losses['deep_supervision'] = ds_loss
            total_with_ds = losses['total'] + ds_loss
            losses['total'] = total_with_ds.clamp(0.0, 20.0) if total_with_ds.isfinite() else total_with_ds
        else:
            losses['deep_supervision'] = self._zero(pred)

        return losses
