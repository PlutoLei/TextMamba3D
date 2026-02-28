# data/transforms.py
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


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
):
    transforms = [
        RandomCrop3D(patch_size),
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
    return Compose(transforms)


def get_val_transforms(patch_size: Tuple[int, int, int]):
    return Compose([
        CenterCrop3D(patch_size),
    ])
