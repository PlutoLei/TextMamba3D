# losses/__init__.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dice_loss import DiceLoss, BRATS_CLASS_WEIGHTS
from .focal_tversky_loss import FocalTverskyLoss
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
        # V5.2: Focal Tversky Loss
        use_ftl: bool = False,
        ftl_alpha: float = 0.3,
        ftl_beta: float = 0.7,
        ftl_gamma: float = 1.33,
        # V7.0: Boundary + Hierarchy Loss
        use_boundary: bool = False,
        boundary_max_weight: float = 1.0,
        use_hierarchy: bool = False,
        hierarchy_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.edge_weight = edge_weight
        self.contrastive_weight = contrastive_weight

        if use_ftl:
            self.dice_loss = FocalTverskyLoss(
                alpha=ftl_alpha, beta=ftl_beta, gamma=ftl_gamma,
                class_weights=class_weights,
            )
        else:
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

        # V7.0: Boundary + Hierarchy Loss
        self.use_boundary = use_boundary
        self.boundary_max_weight = boundary_max_weight
        if use_boundary:
            from losses.boundary_loss import BoundaryLoss
            self.boundary_loss = BoundaryLoss(include_background=False)

        self.use_hierarchy = use_hierarchy
        self.hierarchy_weight = hierarchy_weight
        if use_hierarchy:
            from losses.hierarchy_loss import HierarchyLoss
            self.hierarchy_loss = HierarchyLoss()

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
        distance_map: torch.Tensor | None = None,
        epoch: int = 0,
        total_epochs: int = 80,
    ) -> dict[str, torch.Tensor]:
        losses = {}

        # Cast ALL inputs to fp32 — entire loss computation runs in fp32.
        # bf16 model outputs lack backward kernels for some loss ops (interpolate,
        # sqrt, normalize). PyTorch autograd handles fp32→bf16 gradient flow at
        # the .float() boundary automatically.
        pred = pred.float()
        if img_feat is not None:
            img_feat = img_feat.float()
        if text_feat is not None:
            text_feat = text_feat.float()
        if pixel_feat is not None:
            pixel_feat = pixel_feat.float()
        if aux_preds is not None:
            aux_preds = [a.float() for a in aux_preds]

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

        # V7.0: Boundary Loss with linear alpha annealing
        if self.use_boundary and distance_map is not None:
            alpha = min(1.0, epoch / max(1, total_epochs)) * self.boundary_max_weight
            pred_soft = torch.softmax(pred.float(), dim=1)
            bl = self.boundary_loss(pred_soft, distance_map.float())
            losses['boundary'] = bl * alpha
            total = total + losses['boundary']

        # V7.0: Hierarchy Loss (fixed weight)
        if self.use_hierarchy:
            hl = self.hierarchy_loss(pred.float())
            losses['hierarchy'] = hl * self.hierarchy_weight
            total = total + losses['hierarchy']

        # Clamp finite values only — NaN/Inf pass through for training loop detection
        losses['total'] = total.clamp(0.0, 20.0) if total.isfinite() else total

        # Deep supervision: aux heads at intermediate decoder stages
        if aux_preds:
            ds_loss = self._zero(pred)
            target_size = target.shape[1:]  # (D, H, W)
            for w, aux in zip(self.DS_WEIGHTS, aux_preds):
                aux_up = F.interpolate(aux, size=target_size, mode='trilinear', align_corners=False)
                aux_dice = self.dice_loss(aux_up, target)
                aux_ce = F.cross_entropy(aux_up, target, weight=ce_w)
                ds_loss = ds_loss + w * (aux_dice + aux_ce)
            losses['deep_supervision'] = ds_loss
            total_with_ds = losses['total'] + ds_loss
            losses['total'] = total_with_ds.clamp(0.0, 20.0) if total_with_ds.isfinite() else total_with_ds
        else:
            losses['deep_supervision'] = self._zero(pred)

        return losses
