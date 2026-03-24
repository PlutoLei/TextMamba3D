# data/transforms.py
import os

import nibabel as nib
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class TumorAwareCrop3D:
    """Tumor-centered cropping: 50% centered on tumor, 50% random.
    
    nnU-Net core design: ensures tumor is visible in most patches,
    critical for small tumors in large 3D volumes.
    """

    def __init__(self, size: Tuple[int, int, int], tumor_center_prob: float = 0.5, jitter: int = 8):
        self.size = size
        self.tumor_center_prob = tumor_center_prob
        self.jitter = jitter

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, D, H, W = image.shape
        td, th, tw = self.size

        tumor_coords = torch.nonzero(mask > 0)  # [N, 3]
        use_tumor = len(tumor_coords) > 0 and np.random.random() < self.tumor_center_prob

        if use_tumor:
            center = tumor_coords.float().mean(dim=0).long()  # [3]
            jitter_d = np.random.randint(-self.jitter, self.jitter + 1)
            jitter_h = np.random.randint(-self.jitter, self.jitter + 1)
            jitter_w = np.random.randint(-self.jitter, self.jitter + 1)
            d = int(center[0].item()) + jitter_d - td // 2
            h = int(center[1].item()) + jitter_h - th // 2
            w = int(center[2].item()) + jitter_w - tw // 2
            d = max(0, min(d, D - td))
            h = max(0, min(h, H - th))
            w = max(0, min(w, W - tw))
        else:
            d = np.random.randint(0, max(1, D - td + 1))
            h = np.random.randint(0, max(1, H - th + 1))
            w = np.random.randint(0, max(1, W - tw + 1))

        image = image[:, d:d+td, h:h+th, w:w+tw]
        mask = mask[d:d+td, h:h+th, w:w+tw]
        return image, mask


class RandomCrop3D:
    """Random 3D crop."""

    def __init__(self, size: Tuple[int, int, int]):
        self.size = size

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, D, H, W = image.shape
        td, th, tw = self.size

        d = np.random.randint(0, max(1, D - td + 1))
        h = np.random.randint(0, max(1, H - th + 1))
        w = np.random.randint(0, max(1, W - tw + 1))

        image = image[:, d:d+td, h:h+th, w:w+tw]
        mask = mask[d:d+td, h:h+th, w:w+tw]

        return image, mask


class CenterCrop3D:
    """Center 3D crop for deterministic validation."""

    def __init__(self, size: Tuple[int, int, int]):
        self.size = size

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, D, H, W = image.shape
        td, th, tw = self.size

        d = max(0, (D - td) // 2)
        h = max(0, (H - th) // 2)
        w = max(0, (W - tw) // 2)

        image = image[:, d:d+td, h:h+th, w:w+tw]
        mask = mask[d:d+td, h:h+th, w:w+tw]

        return image, mask


class RandomFlip3D:
    """Random 3D flip."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for axis in [1, 2, 3]:  # D, H, W
            if np.random.random() < self.prob:
                image = torch.flip(image, [axis])
                mask = torch.flip(mask, [axis - 1])

        return image, mask


class RandAffine3D:
    """Random affine: rotation (small angles) + isotropic scaling.

    Uses grid_sample for differentiable interpolation.
    Rotation limited to small angles typical in medical imaging.
    """

    def __init__(
        self,
        rot_range: float = 0.1,   # radians (~5.7 degrees)
        scale_range: float = 0.1,  # +/- 10% scaling
        prob: float = 0.5,
    ):
        self.rot_range = rot_range
        self.scale_range = scale_range
        self.prob = prob

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if np.random.random() > self.prob:
            return image, mask

        # Random rotation angles for each axis
        angles = np.random.uniform(-self.rot_range, self.rot_range, size=3)
        scale = 1.0 + np.random.uniform(-self.scale_range, self.scale_range)

        # Build rotation matrix (Rz @ Ry @ Rx)
        cx, cy, cz = np.cos(angles)
        sx, sy, sz = np.sin(angles)

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        R = (Rz @ Ry @ Rx) * scale
        affine_3x4 = np.zeros((3, 4), dtype=np.float32)
        affine_3x4[:3, :3] = R

        theta = torch.from_numpy(affine_3x4).unsqueeze(0)  # [1, 3, 4]

        # Apply to image [C, D, H, W] -> [1, C, D, H, W]
        img_5d = image.unsqueeze(0)
        grid = F.affine_grid(theta, img_5d.shape, align_corners=False)
        image_out = F.grid_sample(
            img_5d, grid, mode='bilinear', padding_mode='zeros', align_corners=False,
        ).squeeze(0)

        # Apply to mask [D, H, W] -> [1, 1, D, H, W]
        mask_5d = mask.unsqueeze(0).unsqueeze(0).float()
        mask_out = F.grid_sample(
            mask_5d, grid, mode='nearest', padding_mode='zeros', align_corners=False,
        ).squeeze(0).squeeze(0).long()

        return image_out, mask_out


class RandGaussianNoise:
    """Add random Gaussian noise to image."""

    def __init__(self, std: float = 0.1, prob: float = 0.3):
        self.std = std
        self.prob = prob

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if np.random.random() < self.prob:
            noise = torch.randn_like(image) * self.std
            image = image + noise
        return image, mask


class RandIntensityShift:
    """Random per-channel brightness shift and contrast scaling."""

    def __init__(
        self,
        shift_range: float = 0.1,
        scale_range: float = 0.1,
        prob: float = 0.3,
    ):
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.prob = prob

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if np.random.random() < self.prob:
            C = image.shape[0]
            shift = torch.from_numpy(
                np.random.uniform(-self.shift_range, self.shift_range, size=C).astype(np.float32)
            ).view(C, 1, 1, 1)
            scale = 1.0 + torch.from_numpy(
                np.random.uniform(-self.scale_range, self.scale_range, size=C).astype(np.float32)
            ).view(C, 1, 1, 1)
            image = image * scale + shift
        return image, mask


class RandElasticDeformation3D:
    """Random elastic deformation for 3D medical images.

    Generates a low-resolution random displacement field, smooths it with
    Gaussian filter, upsamples to full resolution, and applies via grid_sample.
    """

    def __init__(self, sigma: float = 4.0, magnitude: float = 4.0, prob: float = 0.2):
        self.sigma = sigma
        self.magnitude = magnitude
        self.prob = prob
        try:
            from scipy.ndimage import gaussian_filter
            self._gaussian_filter = gaussian_filter
        except ImportError:
            raise ImportError("scipy is required for elastic deformation: pip install scipy")

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if np.random.random() > self.prob:
            return image, mask

        _, D, H, W = image.shape

        # Low-res random displacement field
        grid_size = (max(D // 8, 3), max(H // 8, 3), max(W // 8, 3))
        disp = np.random.randn(3, *grid_size).astype(np.float32)

        # Smooth with Gaussian
        for i in range(3):
            disp[i] = self._gaussian_filter(disp[i], sigma=self.sigma / 8)

        # Upsample to full resolution
        disp_tensor = torch.from_numpy(disp).unsqueeze(0)
        disp_full = F.interpolate(disp_tensor, size=(D, H, W), mode='trilinear', align_corners=False).squeeze(0)

        # Normalize and scale
        disp_full = disp_full * (self.magnitude / max(D, H, W) * 2)

        # Build sampling grid — grid_sample expects (x=W, y=H, z=D) order
        base_grid = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, D),
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij',
        ), dim=-1).unsqueeze(0)  # [1, D, H, W, 3] but order is (D, H, W)

        # Flip to (W, H, D) for grid_sample convention
        base_grid = base_grid[..., [2, 1, 0]]

        # Displacement field is in (D, H, W) order, reorder to (W, H, D)
        disp_reordered = disp_full[[2, 1, 0]]  # [3, D, H, W] with (W, H, D) component order
        grid = base_grid + disp_reordered.permute(1, 2, 3, 0).unsqueeze(0)

        # Apply to image (bilinear) and mask (nearest)
        img_5d = image.unsqueeze(0)
        image_out = F.grid_sample(img_5d, grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0)

        mask_5d = mask.unsqueeze(0).unsqueeze(0).float()
        mask_out = F.grid_sample(mask_5d, grid, mode='nearest', padding_mode='zeros', align_corners=False).squeeze(0).squeeze(0).long()

        return image_out, mask_out


class RandModalityDropout:
    """Randomly zero out entire modality channels during training.

    Prevents overreliance on any single MRI modality (T1, T1ce, T2, FLAIR).
    """

    def __init__(self, prob: float = 0.15, max_drop: int = 1):
        self.prob = prob
        self.max_drop = max_drop

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if np.random.random() > self.prob:
            return image, mask

        C = image.shape[0]
        n_drop = np.random.randint(1, min(self.max_drop + 1, C))
        drop_idx = np.random.choice(C, size=n_drop, replace=False)
        image = image.clone()
        for idx in drop_idx:
            image[idx] = 0.0
        return image, mask


class TumorCopyPaste3D:
    """Copy-paste tumor regions from a donor bank to augment training samples.

    Inspired by BraTS 2023 winner "Faking it!" strategy. Copies tumor regions
    (especially ET) from one scan and pastes them into healthy or low-ET regions
    of another, with intensity blending at boundaries.

    The donor bank is populated lazily from the dataset during training.
    This augmentation specifically targets ET class imbalance by increasing
    the frequency and diversity of ET appearances.

    Args:
        prob: probability of applying copy-paste per sample.
        bank_size: max number of donor tumors to store.
        blend_sigma: Gaussian sigma for boundary blending (voxels).
        min_et_voxels: minimum ET voxels in donor to store.
        paste_jitter: random position jitter when pasting (voxels).
    """

    def __init__(
        self,
        prob: float = 0.3,
        bank_size: int = 50,
        blend_sigma: float = 2.0,
        min_et_voxels: int = 100,
        paste_jitter: int = 16,
    ):
        self.prob = prob
        self.bank_size = bank_size
        self.blend_sigma = blend_sigma
        self.min_et_voxels = min_et_voxels
        self.paste_jitter = paste_jitter
        self.donor_bank = []  # List of (image_crop, mask_crop, bbox) tuples
        if blend_sigma > 0:
            from scipy.ndimage import gaussian_filter
            self._gaussian_filter = gaussian_filter

    def _extract_tumor_crop(self, image: torch.Tensor, mask: torch.Tensor):
        """Extract tumor bounding box crop from a sample."""
        tumor_mask = mask > 0
        if tumor_mask.sum() == 0:
            return None

        coords = torch.nonzero(tumor_mask)  # [N, 3]
        mins = coords.min(dim=0).values
        maxs = coords.max(dim=0).values + 1

        # Check for sufficient ET
        et_count = (mask == 3).sum().item()
        if et_count < self.min_et_voxels:
            return None

        # Add small margin
        margin = 4
        d0, h0, w0 = max(0, mins[0] - margin), max(0, mins[1] - margin), max(0, mins[2] - margin)
        d1 = min(mask.shape[0], maxs[0] + margin)
        h1 = min(mask.shape[1], maxs[1] + margin)
        w1 = min(mask.shape[2], maxs[2] + margin)

        img_crop = image[:, d0:d1, h0:h1, w0:w1].clone()
        mask_crop = mask[d0:d1, h0:h1, w0:w1].clone()

        return (img_crop, mask_crop, (d1 - d0, h1 - h0, w1 - w0))

    def _add_to_bank(self, image: torch.Tensor, mask: torch.Tensor):
        """Try to add current sample's tumor to the donor bank."""
        crop = self._extract_tumor_crop(image, mask)
        if crop is not None:
            if len(self.donor_bank) >= self.bank_size:
                # Replace random entry
                idx = np.random.randint(0, self.bank_size)
                self.donor_bank[idx] = crop
            else:
                self.donor_bank.append(crop)

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Always try to populate bank
        self._add_to_bank(image, mask)

        if np.random.random() > self.prob or len(self.donor_bank) < 3:
            return image, mask

        # Select random donor
        donor_img, donor_mask, donor_size = self.donor_bank[
            np.random.randint(0, len(self.donor_bank))
        ]

        _, D, H, W = image.shape
        dd, dh, dw = donor_size

        # Skip if donor is too large for the patch
        if dd > D or dh > H or dw > W:
            return image, mask

        # Random paste position with jitter
        d0 = np.random.randint(0, max(1, D - dd + 1))
        h0 = np.random.randint(0, max(1, H - dh + 1))
        w0 = np.random.randint(0, max(1, W - dw + 1))

        # Create blending mask (smooth boundary via distance transform)
        tumor_region = donor_mask > 0
        blend_mask = tumor_region.float()

        # Apply Gaussian smoothing for soft boundary
        if self.blend_sigma > 0:
            blend_np = self._gaussian_filter(blend_mask.numpy(), sigma=self.blend_sigma)
            blend_mask = torch.from_numpy(blend_np)

        # Blend: target = (1 - alpha) * target + alpha * donor
        image = image.clone()
        mask = mask.clone()

        target_region = image[:, d0:d0+dd, h0:h0+dh, w0:w0+dw]
        alpha = blend_mask.unsqueeze(0)  # [1, dd, dh, dw]
        image[:, d0:d0+dd, h0:h0+dh, w0:w0+dw] = (
            (1 - alpha) * target_region + alpha * donor_img
        )

        # Paste mask (hard assignment where tumor exists)
        paste_zone = mask[d0:d0+dd, h0:h0+dh, w0:w0+dw]
        has_donor_tumor = donor_mask > 0
        paste_zone[has_donor_tumor] = donor_mask[has_donor_tumor]
        mask[d0:d0+dd, h0:h0+dh, w0:w0+dw] = paste_zone

        return image, mask


class ETOversampler:
    """Wrapper that oversamples training indices with high ET content.

    Not a transform — used to modify the dataset's __getitem__ sampling.
    Provides a weighted index list where ET-rich cases appear more often.
    """

    @staticmethod
    def compute_et_weights(dataset, et_label: int = 3, boost_factor: float = 3.0):
        """Compute sampling weights based on ET voxel count per case.

        Returns list of weights (one per case) for WeightedRandomSampler.
        """
        weights = []
        for case_dir in dataset.cases:
            case_name = os.path.basename(case_dir)
            seg_path = os.path.join(case_dir, f'{case_name}_seg.nii')
            if not os.path.exists(seg_path):
                seg_path += '.gz'
            try:
                seg = nib.load(seg_path).get_fdata()
                seg[seg == 4] = 3  # BraTS label mapping
                et_ratio = (seg == et_label).sum() / max(1, (seg > 0).sum())
                # Higher weight for ET-rich cases
                w = 1.0 + boost_factor * et_ratio
            except Exception:
                w = 1.0
            weights.append(w)
        return weights


class Compose:
    """Compose transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


def get_train_transforms(
    patch_size: Tuple[int, int, int],
    use_elastic: bool = False,
    use_modality_dropout: bool = False,
    use_copy_paste: bool = False,
    copy_paste_prob: float = 0.3,
):
    transforms = [
        TumorAwareCrop3D(patch_size),
        RandomFlip3D(prob=0.5),
        RandAffine3D(rot_range=0.1, scale_range=0.1, prob=0.3),
    ]
    if use_elastic:
        transforms.append(RandElasticDeformation3D(sigma=4.0, magnitude=4.0, prob=0.2))
    transforms.extend([
        RandGaussianNoise(std=0.1, prob=0.2),
        RandIntensityShift(shift_range=0.1, scale_range=0.1, prob=0.2),
    ])
    if use_modality_dropout:
        transforms.append(RandModalityDropout(prob=0.15, max_drop=1))
    if use_copy_paste:
        transforms.append(TumorCopyPaste3D(prob=copy_paste_prob))
    return Compose(transforms)


def get_val_transforms(patch_size: Tuple[int, int, int]):
    return Compose([
        CenterCrop3D(patch_size),
    ])
