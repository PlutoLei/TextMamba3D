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
    """Test-time augmentation: average softmax predictions over flipped inputs.

    Flips the input image along spatial axes, runs forward pass for each
    augmented version, flips predictions back, and averages the softmax
    probabilities for a more robust prediction.

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

    accum = None
    for axes in flip_axes:
        # Flip input
        img_aug = image
        for ax in axes:
            img_aug = torch.flip(img_aug, [ax])

        # Forward pass
        logits = model(img_aug, text_ids, attention_mask=attention_mask, use_text=use_text)

        # Flip prediction back
        for ax in axes:
            logits = torch.flip(logits, [ax])

        probs = F.softmax(logits, dim=1)
        accum = probs if accum is None else accum + probs

    return accum / len(flip_axes)
