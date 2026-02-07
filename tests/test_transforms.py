"""Comprehensive tests for data.transforms module.

Tests cover all transform classes and factory functions with emphasis on:
- Output shapes
- Value ranges
- Probabilistic behavior (prob=0 => no change, prob=1 => always change)
- Mask dtype preservation (torch.long)
- Compose chaining
- Factory function output types
"""

import pytest
import torch
import numpy as np
from data.transforms import (
    RandomCrop3D,
    CenterCrop3D,
    RandomFlip3D,
    RandAffine3D,
    RandGaussianNoise,
    RandIntensityShift,
    RandElasticDeformation3D,
    RandModalityDropout,
    Compose,
    get_train_transforms,
    get_val_transforms,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_pair():
    """Return a small (C=2, D=16, H=16, W=16) image/mask pair."""
    image = torch.randn(2, 16, 16, 16)
    mask = torch.randint(0, 4, (16, 16, 16), dtype=torch.long)
    return image, mask


@pytest.fixture
def tiny_pair():
    """Return a tiny (C=1, D=8, H=8, W=8) image/mask pair."""
    image = torch.randn(1, 8, 8, 8)
    mask = torch.randint(0, 2, (8, 8, 8), dtype=torch.long)
    return image, mask


@pytest.fixture
def multichannel_pair():
    """Return a 4-channel (C=4, D=8, H=8, W=8) image/mask pair."""
    image = torch.randn(4, 8, 8, 8)
    mask = torch.randint(0, 3, (8, 8, 8), dtype=torch.long)
    return image, mask


# ---------------------------------------------------------------------------
# TestRandomCrop3D
# ---------------------------------------------------------------------------

class TestRandomCrop3D:
    def test_output_shape(self, small_pair):
        crop = RandomCrop3D(size=(8, 8, 8))
        img_out, mask_out = crop(*small_pair)
        assert img_out.shape == (2, 8, 8, 8)
        assert mask_out.shape == (8, 8, 8)

    def test_mask_dtype_preserved(self, small_pair):
        crop = RandomCrop3D(size=(8, 8, 8))
        _, mask_out = crop(*small_pair)
        assert mask_out.dtype == torch.long

    def test_crop_same_size_as_input(self, small_pair):
        """Cropping to same size should return the full volume."""
        crop = RandomCrop3D(size=(16, 16, 16))
        img_out, mask_out = crop(*small_pair)
        assert img_out.shape == (2, 16, 16, 16)
        assert mask_out.shape == (16, 16, 16)

    def test_crop_larger_than_input(self, tiny_pair):
        """When crop size exceeds input, output is smaller due to slicing."""
        crop = RandomCrop3D(size=(16, 16, 16))
        img_out, mask_out = crop(*tiny_pair)
        # The start index is randint(0, max(1, 8-16+1)) = randint(0, 1) = 0
        # Slicing 0:16 on dim of size 8 yields 8
        assert img_out.shape[1] <= 8
        assert mask_out.shape[0] <= 8

    def test_crop_asymmetric(self, small_pair):
        crop = RandomCrop3D(size=(4, 8, 12))
        img_out, mask_out = crop(*small_pair)
        assert img_out.shape == (2, 4, 8, 12)
        assert mask_out.shape == (4, 8, 12)

    def test_values_are_subset_of_original(self, small_pair):
        """Cropped values must be a subset of the original tensor values."""
        image, mask = small_pair
        crop = RandomCrop3D(size=(8, 8, 8))
        img_out, mask_out = crop(image, mask)
        # Mask labels should be a subset
        assert set(mask_out.unique().tolist()).issubset(set(mask.unique().tolist()))

    def test_reproducibility_with_seed(self, small_pair):
        crop = RandomCrop3D(size=(8, 8, 8))
        np.random.seed(42)
        out1 = crop(*small_pair)
        np.random.seed(42)
        out2 = crop(*small_pair)
        assert torch.equal(out1[0], out2[0])
        assert torch.equal(out1[1], out2[1])


# ---------------------------------------------------------------------------
# TestCenterCrop3D
# ---------------------------------------------------------------------------

class TestCenterCrop3D:
    def test_output_shape(self, small_pair):
        crop = CenterCrop3D(size=(8, 8, 8))
        img_out, mask_out = crop(*small_pair)
        assert img_out.shape == (2, 8, 8, 8)
        assert mask_out.shape == (8, 8, 8)

    def test_mask_dtype_preserved(self, small_pair):
        crop = CenterCrop3D(size=(8, 8, 8))
        _, mask_out = crop(*small_pair)
        assert mask_out.dtype == torch.long

    def test_deterministic(self, small_pair):
        """CenterCrop3D should always produce the same result."""
        crop = CenterCrop3D(size=(8, 8, 8))
        out1 = crop(*small_pair)
        out2 = crop(*small_pair)
        assert torch.equal(out1[0], out2[0])
        assert torch.equal(out1[1], out2[1])

    def test_center_location(self, small_pair):
        """Verify that the center crop selects from the middle."""
        image, mask = small_pair
        crop = CenterCrop3D(size=(8, 8, 8))
        img_out, _ = crop(image, mask)
        # Center of 16 with crop 8 -> start = (16-8)//2 = 4
        expected = image[:, 4:12, 4:12, 4:12]
        assert torch.equal(img_out, expected)

    def test_crop_same_size_as_input(self, small_pair):
        crop = CenterCrop3D(size=(16, 16, 16))
        img_out, mask_out = crop(*small_pair)
        assert torch.equal(img_out, small_pair[0])
        assert torch.equal(mask_out, small_pair[1])

    def test_crop_larger_than_input(self, tiny_pair):
        """When crop > input, start = max(0, (8-16)//2) = 0, slicing returns full dim."""
        crop = CenterCrop3D(size=(16, 16, 16))
        img_out, mask_out = crop(*tiny_pair)
        assert img_out.shape[1] == 8
        assert mask_out.shape[0] == 8

    def test_asymmetric_crop(self, small_pair):
        crop = CenterCrop3D(size=(4, 8, 12))
        img_out, mask_out = crop(*small_pair)
        assert img_out.shape == (2, 4, 8, 12)
        assert mask_out.shape == (4, 8, 12)


# ---------------------------------------------------------------------------
# TestRandomFlip3D
# ---------------------------------------------------------------------------

class TestRandomFlip3D:
    def test_output_shape(self, small_pair):
        flip = RandomFlip3D(prob=0.5)
        img_out, mask_out = flip(*small_pair)
        assert img_out.shape == small_pair[0].shape
        assert mask_out.shape == small_pair[1].shape

    def test_mask_dtype_preserved(self, small_pair):
        flip = RandomFlip3D(prob=1.0)
        _, mask_out = flip(*small_pair)
        assert mask_out.dtype == torch.long

    def test_prob_zero_no_change(self, small_pair):
        """With prob=0, no flip should occur."""
        flip = RandomFlip3D(prob=0.0)
        img_out, mask_out = flip(*small_pair)
        assert torch.equal(img_out, small_pair[0])
        assert torch.equal(mask_out, small_pair[1])

    def test_prob_one_always_flips(self, small_pair):
        """With prob=1, all three axes should be flipped."""
        image, mask = small_pair
        flip = RandomFlip3D(prob=1.0)
        img_out, mask_out = flip(image.clone(), mask.clone())
        # All three spatial axes flipped: axes 1,2,3 for image, 0,1,2 for mask
        expected_img = torch.flip(torch.flip(torch.flip(image, [1]), [2]), [3])
        expected_mask = torch.flip(torch.flip(torch.flip(mask, [0]), [1]), [2])
        assert torch.equal(img_out, expected_img)
        assert torch.equal(mask_out, expected_mask)

    def test_double_flip_identity(self, small_pair):
        """Flipping twice with prob=1 should return original."""
        image, mask = small_pair
        flip = RandomFlip3D(prob=1.0)
        img_mid, mask_mid = flip(image.clone(), mask.clone())
        img_out, mask_out = flip(img_mid, mask_mid)
        assert torch.equal(img_out, image)
        assert torch.equal(mask_out, mask)

    def test_values_preserved(self, small_pair):
        """Flipping should not change the multiset of values."""
        image, mask = small_pair
        flip = RandomFlip3D(prob=1.0)
        img_out, mask_out = flip(image.clone(), mask.clone())
        # Sorted flat tensors should be identical
        assert torch.equal(img_out.flatten().sort().values, image.flatten().sort().values)
        assert torch.equal(mask_out.flatten().sort().values, mask.flatten().sort().values)


# ---------------------------------------------------------------------------
# TestRandAffine3D
# ---------------------------------------------------------------------------

class TestRandAffine3D:
    def test_output_shape(self, small_pair):
        affine = RandAffine3D(prob=1.0)
        img_out, mask_out = affine(*small_pair)
        assert img_out.shape == small_pair[0].shape
        assert mask_out.shape == small_pair[1].shape

    def test_mask_dtype_preserved(self, small_pair):
        affine = RandAffine3D(prob=1.0)
        _, mask_out = affine(*small_pair)
        assert mask_out.dtype == torch.long

    def test_prob_zero_no_change(self, small_pair):
        """With prob=0 (i.e., random > 0 never triggers), output equals input."""
        # prob=0 means np.random.random() > 0 is almost always true, so it returns early.
        # Actually: if np.random.random() > self.prob => if rand > 0 => always True => no change
        affine = RandAffine3D(prob=0.0)
        img_out, mask_out = affine(*small_pair)
        assert torch.equal(img_out, small_pair[0])
        assert torch.equal(mask_out, small_pair[1])

    def test_prob_one_applies_transform(self, small_pair):
        """With prob=1, the transform should be applied (output differs)."""
        image, mask = small_pair
        affine = RandAffine3D(rot_range=0.5, scale_range=0.3, prob=1.0)
        np.random.seed(123)
        img_out, mask_out = affine(image.clone(), mask.clone())
        # Very unlikely to be identical after a substantial affine
        assert not torch.equal(img_out, image)

    def test_identity_affine(self, small_pair):
        """With zero rotation and zero scale range, prob=1 should be near-identity."""
        image, mask = small_pair
        affine = RandAffine3D(rot_range=0.0, scale_range=0.0, prob=1.0)
        img_out, _ = affine(image.clone(), mask.clone())
        # Should be very close to original (grid_sample interpolation may cause tiny diffs)
        assert torch.allclose(img_out, image, atol=0.05)

    def test_output_is_finite(self, small_pair):
        affine = RandAffine3D(prob=1.0)
        img_out, mask_out = affine(*small_pair)
        assert torch.isfinite(img_out).all()
        assert torch.isfinite(mask_out.float()).all()


# ---------------------------------------------------------------------------
# TestRandGaussianNoise
# ---------------------------------------------------------------------------

class TestRandGaussianNoise:
    def test_output_shape(self, small_pair):
        noise = RandGaussianNoise(std=0.1, prob=1.0)
        img_out, mask_out = noise(*small_pair)
        assert img_out.shape == small_pair[0].shape
        assert mask_out.shape == small_pair[1].shape

    def test_mask_unchanged(self, small_pair):
        """Noise should only affect the image, not the mask."""
        noise = RandGaussianNoise(std=0.5, prob=1.0)
        _, mask_out = noise(*small_pair)
        assert torch.equal(mask_out, small_pair[1])

    def test_mask_dtype_preserved(self, small_pair):
        noise = RandGaussianNoise(std=0.1, prob=1.0)
        _, mask_out = noise(*small_pair)
        assert mask_out.dtype == torch.long

    def test_prob_zero_no_change(self, small_pair):
        """prob=0 means noise is never applied."""
        noise = RandGaussianNoise(std=1.0, prob=0.0)
        img_out, mask_out = noise(*small_pair)
        assert torch.equal(img_out, small_pair[0])
        assert torch.equal(mask_out, small_pair[1])

    def test_prob_one_always_applies(self, small_pair):
        """prob=1 should always add noise, changing the image."""
        image, mask = small_pair
        noise = RandGaussianNoise(std=0.5, prob=1.0)
        torch.manual_seed(0)
        img_out, _ = noise(image.clone(), mask.clone())
        assert not torch.equal(img_out, image)

    def test_noise_magnitude(self, small_pair):
        """Added noise should have approximately the specified std."""
        image, mask = small_pair
        noise = RandGaussianNoise(std=0.1, prob=1.0)
        torch.manual_seed(42)
        img_out, _ = noise(image.clone(), mask.clone())
        diff = img_out - image
        # Check std of diff is approximately 0.1 (allow generous tolerance for small tensors)
        assert 0.05 < diff.std().item() < 0.2

    def test_output_is_finite(self, small_pair):
        noise = RandGaussianNoise(std=0.1, prob=1.0)
        img_out, _ = noise(*small_pair)
        assert torch.isfinite(img_out).all()


# ---------------------------------------------------------------------------
# TestRandIntensityShift
# ---------------------------------------------------------------------------

class TestRandIntensityShift:
    def test_output_shape(self, small_pair):
        shift = RandIntensityShift(prob=1.0)
        img_out, mask_out = shift(*small_pair)
        assert img_out.shape == small_pair[0].shape
        assert mask_out.shape == small_pair[1].shape

    def test_mask_unchanged(self, small_pair):
        shift = RandIntensityShift(prob=1.0)
        _, mask_out = shift(*small_pair)
        assert torch.equal(mask_out, small_pair[1])

    def test_mask_dtype_preserved(self, small_pair):
        shift = RandIntensityShift(prob=1.0)
        _, mask_out = shift(*small_pair)
        assert mask_out.dtype == torch.long

    def test_prob_zero_no_change(self, small_pair):
        shift = RandIntensityShift(shift_range=0.5, scale_range=0.5, prob=0.0)
        img_out, mask_out = shift(*small_pair)
        assert torch.equal(img_out, small_pair[0])
        assert torch.equal(mask_out, small_pair[1])

    def test_prob_one_applies_per_channel(self, multichannel_pair):
        """With prob=1, all channels should be modified."""
        image, mask = multichannel_pair
        shift = RandIntensityShift(shift_range=0.5, scale_range=0.5, prob=1.0)
        np.random.seed(99)
        img_out, _ = shift(image.clone(), mask.clone())
        # At least some channels should differ
        changed = sum(
            not torch.equal(img_out[c], image[c]) for c in range(image.shape[0])
        )
        assert changed > 0

    def test_output_is_finite(self, small_pair):
        shift = RandIntensityShift(prob=1.0)
        img_out, _ = shift(*small_pair)
        assert torch.isfinite(img_out).all()


# ---------------------------------------------------------------------------
# TestRandElasticDeformation3D
# ---------------------------------------------------------------------------

class TestRandElasticDeformation3D:
    def test_output_shape(self, small_pair):
        elastic = RandElasticDeformation3D(prob=1.0)
        img_out, mask_out = elastic(*small_pair)
        assert img_out.shape == small_pair[0].shape
        assert mask_out.shape == small_pair[1].shape

    def test_mask_dtype_preserved(self, small_pair):
        elastic = RandElasticDeformation3D(prob=1.0)
        _, mask_out = elastic(*small_pair)
        assert mask_out.dtype == torch.long

    def test_prob_zero_no_change(self, small_pair):
        """prob=0 means np.random.random() > 0 is always True => early return."""
        elastic = RandElasticDeformation3D(prob=0.0)
        img_out, mask_out = elastic(*small_pair)
        assert torch.equal(img_out, small_pair[0])
        assert torch.equal(mask_out, small_pair[1])

    def test_prob_one_applies_deformation(self, small_pair):
        image, mask = small_pair
        elastic = RandElasticDeformation3D(sigma=4.0, magnitude=4.0, prob=1.0)
        np.random.seed(7)
        img_out, _ = elastic(image.clone(), mask.clone())
        # Elastic deformation should change the image
        assert not torch.equal(img_out, image)

    def test_output_is_finite(self, small_pair):
        elastic = RandElasticDeformation3D(prob=1.0)
        img_out, mask_out = elastic(*small_pair)
        assert torch.isfinite(img_out).all()
        assert torch.isfinite(mask_out.float()).all()

    def test_small_magnitude_less_distortion_than_large(self, small_pair):
        """Small magnitude should cause less distortion than large magnitude."""
        image, mask = small_pair
        np.random.seed(0)
        elastic_small = RandElasticDeformation3D(sigma=1.0, magnitude=0.01, prob=1.0)
        img_small, _ = elastic_small(image.clone(), mask.clone())
        np.random.seed(0)
        elastic_large = RandElasticDeformation3D(sigma=1.0, magnitude=10.0, prob=1.0)
        img_large, _ = elastic_large(image.clone(), mask.clone())
        mae_small = (img_small - image).abs().mean().item()
        mae_large = (img_large - image).abs().mean().item()
        assert mae_small < mae_large


# ---------------------------------------------------------------------------
# TestRandModalityDropout
# ---------------------------------------------------------------------------

class TestRandModalityDropout:
    def test_output_shape(self, multichannel_pair):
        dropout = RandModalityDropout(prob=1.0, max_drop=2)
        img_out, mask_out = dropout(*multichannel_pair)
        assert img_out.shape == multichannel_pair[0].shape
        assert mask_out.shape == multichannel_pair[1].shape

    def test_mask_unchanged(self, multichannel_pair):
        dropout = RandModalityDropout(prob=1.0, max_drop=2)
        _, mask_out = dropout(*multichannel_pair)
        assert torch.equal(mask_out, multichannel_pair[1])

    def test_mask_dtype_preserved(self, multichannel_pair):
        dropout = RandModalityDropout(prob=1.0, max_drop=1)
        _, mask_out = dropout(*multichannel_pair)
        assert mask_out.dtype == torch.long

    def test_prob_zero_no_change(self, multichannel_pair):
        """prob=0 means np.random.random() > 0 is always True => early return."""
        dropout = RandModalityDropout(prob=0.0, max_drop=4)
        img_out, mask_out = dropout(*multichannel_pair)
        assert torch.equal(img_out, multichannel_pair[0])
        assert torch.equal(mask_out, multichannel_pair[1])

    def test_prob_one_drops_channels(self, multichannel_pair):
        """With prob=1, at least one channel should be zeroed out."""
        image, mask = multichannel_pair
        dropout = RandModalityDropout(prob=1.0, max_drop=1)
        np.random.seed(0)
        img_out, _ = dropout(image.clone(), mask.clone())
        zeroed = sum(
            (img_out[c] == 0).all().item() for c in range(image.shape[0])
        )
        assert zeroed >= 1

    def test_does_not_modify_original(self, multichannel_pair):
        """The transform should clone internally and not modify the input."""
        image, mask = multichannel_pair
        original = image.clone()
        dropout = RandModalityDropout(prob=1.0, max_drop=2)
        _ = dropout(image, mask)
        assert torch.equal(image, original), "Input tensor should not be modified in place"

    def test_max_drop_respected(self, multichannel_pair):
        """Number of dropped channels should not exceed max_drop."""
        image, mask = multichannel_pair
        max_drop = 2
        dropout = RandModalityDropout(prob=1.0, max_drop=max_drop)
        for _ in range(20):
            img_out, _ = dropout(image.clone(), mask.clone())
            zeroed = sum(
                (img_out[c] == 0).all().item() for c in range(image.shape[0])
            )
            # With randn data, naturally-zero channels are near-impossible.
            # Source: randint(1, min(max_drop+1, C)) → at most max_drop channels dropped.
            assert zeroed <= max_drop, f"Dropped {zeroed} channels, max_drop={max_drop}"

    def test_single_channel_raises_on_max_drop_equal_channels(self):
        """With C=1 and max_drop=1, min(max_drop+1, C)=1 so randint(1,1) raises.

        This documents a known edge case in the source implementation:
        when max_drop >= C, np.random.randint(1, min(max_drop+1, C)) gets low >= high.
        """
        image = torch.randn(1, 8, 8, 8)
        mask = torch.randint(0, 2, (8, 8, 8), dtype=torch.long)
        dropout = RandModalityDropout(prob=1.0, max_drop=1)
        with pytest.raises(ValueError, match="low >= high"):
            dropout(image, mask)

    def test_two_channels_max_drop_one(self):
        """With C=2 and max_drop=1, exactly one channel should be zeroed."""
        image = torch.randn(2, 8, 8, 8)
        mask = torch.randint(0, 2, (8, 8, 8), dtype=torch.long)
        dropout = RandModalityDropout(prob=1.0, max_drop=1)
        img_out, _ = dropout(image, mask)
        zeroed = sum((img_out[c] == 0).all().item() for c in range(2))
        assert zeroed == 1


# ---------------------------------------------------------------------------
# TestCompose
# ---------------------------------------------------------------------------

class TestCompose:
    def test_empty_compose(self, small_pair):
        compose = Compose([])
        img_out, mask_out = compose(*small_pair)
        assert torch.equal(img_out, small_pair[0])
        assert torch.equal(mask_out, small_pair[1])

    def test_single_transform(self, small_pair):
        compose = Compose([CenterCrop3D((8, 8, 8))])
        img_out, mask_out = compose(*small_pair)
        assert img_out.shape == (2, 8, 8, 8)
        assert mask_out.shape == (8, 8, 8)

    def test_chaining_multiple_transforms(self, small_pair):
        compose = Compose([
            CenterCrop3D((12, 12, 12)),
            RandomFlip3D(prob=0.0),  # no-op
        ])
        img_out, mask_out = compose(*small_pair)
        assert img_out.shape == (2, 12, 12, 12)
        assert mask_out.shape == (12, 12, 12)

    def test_crop_then_noise(self, small_pair):
        compose = Compose([
            CenterCrop3D((8, 8, 8)),
            RandGaussianNoise(std=0.1, prob=1.0),
        ])
        img_out, mask_out = compose(*small_pair)
        assert img_out.shape == (2, 8, 8, 8)
        assert mask_out.dtype == torch.long

    def test_mask_dtype_after_chain(self, small_pair):
        compose = Compose([
            RandomCrop3D((8, 8, 8)),
            RandomFlip3D(prob=1.0),
            RandGaussianNoise(std=0.1, prob=1.0),
            RandIntensityShift(prob=1.0),
        ])
        _, mask_out = compose(*small_pair)
        assert mask_out.dtype == torch.long

    def test_output_shapes_through_pipeline(self, small_pair):
        compose = Compose([
            RandomCrop3D((10, 10, 10)),
            RandomFlip3D(prob=0.5),
            RandGaussianNoise(std=0.05, prob=0.5),
        ])
        img_out, mask_out = compose(*small_pair)
        assert img_out.shape == (2, 10, 10, 10)
        assert mask_out.shape == (10, 10, 10)


# ---------------------------------------------------------------------------
# TestGetTrainTransforms
# ---------------------------------------------------------------------------

class TestGetTrainTransforms:
    def test_returns_compose(self):
        transforms = get_train_transforms(patch_size=(8, 8, 8))
        assert isinstance(transforms, Compose)

    def test_callable(self, small_pair):
        transforms = get_train_transforms(patch_size=(8, 8, 8))
        img_out, mask_out = transforms(*small_pair)
        assert img_out.shape == (2, 8, 8, 8)
        assert mask_out.shape == (8, 8, 8)

    def test_mask_dtype_preserved(self, small_pair):
        transforms = get_train_transforms(patch_size=(8, 8, 8))
        _, mask_out = transforms(*small_pair)
        assert mask_out.dtype == torch.long

    def test_without_elastic_and_dropout(self):
        transforms = get_train_transforms(
            patch_size=(8, 8, 8), use_elastic=False, use_modality_dropout=False
        )
        # Should have: RandomCrop3D, RandomFlip3D, RandAffine3D, RandGaussianNoise, RandIntensityShift
        assert len(transforms.transforms) == 5
        assert isinstance(transforms.transforms[0], RandomCrop3D)
        assert isinstance(transforms.transforms[1], RandomFlip3D)
        assert isinstance(transforms.transforms[2], RandAffine3D)
        assert isinstance(transforms.transforms[3], RandGaussianNoise)
        assert isinstance(transforms.transforms[4], RandIntensityShift)

    def test_with_elastic(self):
        transforms = get_train_transforms(
            patch_size=(8, 8, 8), use_elastic=True, use_modality_dropout=False
        )
        # +1 for RandElasticDeformation3D
        assert len(transforms.transforms) == 6
        elastic_found = any(
            isinstance(t, RandElasticDeformation3D) for t in transforms.transforms
        )
        assert elastic_found

    def test_with_modality_dropout(self):
        transforms = get_train_transforms(
            patch_size=(8, 8, 8), use_elastic=False, use_modality_dropout=True
        )
        # +1 for RandModalityDropout
        assert len(transforms.transforms) == 6
        dropout_found = any(
            isinstance(t, RandModalityDropout) for t in transforms.transforms
        )
        assert dropout_found

    def test_with_all_options(self):
        transforms = get_train_transforms(
            patch_size=(8, 8, 8), use_elastic=True, use_modality_dropout=True
        )
        # 5 base + elastic + modality_dropout = 7
        assert len(transforms.transforms) == 7

    def test_output_finite(self, small_pair):
        transforms = get_train_transforms(patch_size=(8, 8, 8))
        img_out, mask_out = transforms(*small_pair)
        assert torch.isfinite(img_out).all()
        assert torch.isfinite(mask_out.float()).all()


# ---------------------------------------------------------------------------
# TestGetValTransforms
# ---------------------------------------------------------------------------

class TestGetValTransforms:
    def test_returns_compose(self):
        transforms = get_val_transforms(patch_size=(8, 8, 8))
        assert isinstance(transforms, Compose)

    def test_contains_only_center_crop(self):
        transforms = get_val_transforms(patch_size=(8, 8, 8))
        assert len(transforms.transforms) == 1
        assert isinstance(transforms.transforms[0], CenterCrop3D)

    def test_callable(self, small_pair):
        transforms = get_val_transforms(patch_size=(8, 8, 8))
        img_out, mask_out = transforms(*small_pair)
        assert img_out.shape == (2, 8, 8, 8)
        assert mask_out.shape == (8, 8, 8)

    def test_deterministic(self, small_pair):
        transforms = get_val_transforms(patch_size=(8, 8, 8))
        out1 = transforms(*small_pair)
        out2 = transforms(*small_pair)
        assert torch.equal(out1[0], out2[0])
        assert torch.equal(out1[1], out2[1])

    def test_mask_dtype_preserved(self, small_pair):
        transforms = get_val_transforms(patch_size=(8, 8, 8))
        _, mask_out = transforms(*small_pair)
        assert mask_out.dtype == torch.long

    def test_asymmetric_patch(self, small_pair):
        transforms = get_val_transforms(patch_size=(4, 8, 12))
        img_out, mask_out = transforms(*small_pair)
        assert img_out.shape == (2, 4, 8, 12)
        assert mask_out.shape == (4, 8, 12)


# ---------------------------------------------------------------------------
# Integration / Edge-case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_channel_through_pipeline(self):
        """Single-channel image through train transforms."""
        image = torch.randn(1, 16, 16, 16)
        mask = torch.randint(0, 2, (16, 16, 16), dtype=torch.long)
        transforms = get_train_transforms(patch_size=(8, 8, 8))
        img_out, mask_out = transforms(image, mask)
        assert img_out.shape == (1, 8, 8, 8)
        assert mask_out.shape == (8, 8, 8)
        assert mask_out.dtype == torch.long

    def test_many_channels_through_pipeline(self):
        """Multi-channel (4) image through full train pipeline."""
        image = torch.randn(4, 16, 16, 16)
        mask = torch.randint(0, 5, (16, 16, 16), dtype=torch.long)
        transforms = get_train_transforms(
            patch_size=(8, 8, 8), use_elastic=True, use_modality_dropout=True
        )
        img_out, mask_out = transforms(image, mask)
        assert img_out.shape == (4, 8, 8, 8)
        assert mask_out.shape == (8, 8, 8)
        assert mask_out.dtype == torch.long

    def test_all_zeros_input(self):
        """Transform should handle all-zero inputs without error."""
        image = torch.zeros(2, 8, 8, 8)
        mask = torch.zeros(8, 8, 8, dtype=torch.long)
        transforms = get_val_transforms(patch_size=(4, 4, 4))
        img_out, mask_out = transforms(image, mask)
        assert (img_out == 0).all()
        assert (mask_out == 0).all()

    def test_binary_mask_preservation(self):
        """After val transforms, binary mask should remain binary."""
        image = torch.randn(1, 16, 16, 16)
        mask = torch.randint(0, 2, (16, 16, 16), dtype=torch.long)
        transforms = get_val_transforms(patch_size=(8, 8, 8))
        _, mask_out = transforms(image, mask)
        unique_vals = mask_out.unique()
        assert all(v in [0, 1] for v in unique_vals.tolist())

    def test_crop_1x1x1(self):
        """Extreme case: crop to 1x1x1."""
        image = torch.randn(2, 8, 8, 8)
        mask = torch.randint(0, 3, (8, 8, 8), dtype=torch.long)
        crop = CenterCrop3D(size=(1, 1, 1))
        img_out, mask_out = crop(image, mask)
        assert img_out.shape == (2, 1, 1, 1)
        assert mask_out.shape == (1, 1, 1)
