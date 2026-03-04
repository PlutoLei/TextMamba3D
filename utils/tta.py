# utils/tta.py
"""Test-Time Augmentation (TTA) utilities for 3D medical image segmentation."""

import torch
import torch.nn.functional as F


# Flip axis combinations for [B, C, D, H, W] tensors
# 8 augmentations: all combinations of flipping D(2), H(3), W(4)
FLIP_AXES_8 = [[], [2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]

# 4 augmentations: original + single-axis flips only
FLIP_AXES_4 = [[], [2], [3], [4]]


def tta_predict(
    model,
    image: torch.Tensor,
    text_ids: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
    use_text: bool = True,
    num_flips: int = 8,
) -> torch.Tensor:
    """Test-time augmentation: average predictions over flipped inputs.

    Applies softmax per augmentation then averages probabilities (standard
    approach in medical image segmentation, matching nnUNet convention).

    Attempts batched forward pass for speed; falls back to sequential on OOM.

    Args:
        model: The segmentation model (expects [B, C, D, H, W] input).
        image: Input tensor of shape [B, C, D, H, W].
        text_ids: Optional text token IDs for text-guided models.
        attention_mask: Optional attention mask for text tokens.
        use_text: Whether to use text guidance.
        num_flips: Number of flip augmentations (4 or 8).

    Returns:
        Averaged softmax probabilities of shape [B, num_classes, D, H, W].
    """
    if num_flips == 8:
        flip_axes = FLIP_AXES_8
    elif num_flips == 4:
        flip_axes = FLIP_AXES_4
    else:
        raise ValueError(f"num_flips must be 4 or 8, got {num_flips}")

    try:
        return _tta_batched(model, image, text_ids, attention_mask, use_text, flip_axes)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return _tta_sequential(model, image, text_ids, attention_mask, use_text, flip_axes)


def _tta_batched(model, image, text_ids, attention_mask, use_text, flip_axes):
    """Batched TTA: stack all augmentations into one forward pass."""
    N = len(flip_axes)
    B = image.shape[0]

    # Create all augmented images
    aug_images = []
    for axes in flip_axes:
        img_aug = image
        for ax in axes:
            img_aug = torch.flip(img_aug, [ax])
        aug_images.append(img_aug)

    # Stack: [N*B, C, D, H, W]
    batched = torch.cat(aug_images, dim=0)

    # Replicate text tensors to match batch size
    batched_text_ids = None
    batched_attn_mask = None
    if text_ids is not None:
        batched_text_ids = text_ids.repeat(N, *([1] * (text_ids.dim() - 1)))
    if attention_mask is not None:
        batched_attn_mask = attention_mask.repeat(N, *([1] * (attention_mask.dim() - 1)))

    # Single forward pass
    all_logits = model(
        batched, batched_text_ids,
        attention_mask=batched_attn_mask, use_text=use_text,
    )

    # Split back, unflip, and average probabilities
    logit_chunks = all_logits.split(B, dim=0)
    accum = torch.zeros_like(logit_chunks[0])
    for logits, axes in zip(logit_chunks, flip_axes):
        for ax in axes:
            logits = torch.flip(logits, [ax])
        accum += F.softmax(logits, dim=1)

    return accum / N


def _tta_sequential(model, image, text_ids, attention_mask, use_text, flip_axes):
    """Sequential TTA fallback for low-memory environments."""
    accum = None
    for axes in flip_axes:
        img_aug = image
        for ax in axes:
            img_aug = torch.flip(img_aug, [ax])

        logits = model(img_aug, text_ids, attention_mask=attention_mask, use_text=use_text)

        for ax in axes:
            logits = torch.flip(logits, [ax])

        probs = F.softmax(logits, dim=1)
        accum = probs if accum is None else accum + probs

    return accum / len(flip_axes)
