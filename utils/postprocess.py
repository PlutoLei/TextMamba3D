"""BraTS post-processing utilities.

Pure numpy/scipy — no torch dependency. Can be tested on Windows without CUDA.
"""

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    generate_binary_structure,
    label as ndimage_label,
)

# 6-connectivity structure for 3D dilation
_STRUCT_3D = generate_binary_structure(3, 1)


def postprocess_et(pred_argmax: np.ndarray, et_class: int = 3, min_size: int = 50) -> np.ndarray:
    """Remove small ET connected components (false positives).

    Small ET clusters are reclassified as NCR (class 1) — stays in TC but not ET.
    """
    pred = pred_argmax.copy()
    et_mask = pred == et_class
    if et_mask.sum() == 0:
        return pred
    labeled, n = ndimage_label(et_mask)
    for i in range(1, n + 1):
        comp = labeled == i
        if comp.sum() < min_size:
            pred[comp] = 1
    return pred


def fill_holes_et(pred_argmax: np.ndarray, et_class: int = 3) -> np.ndarray:
    """Fill internal holes in ET regions."""
    pred = pred_argmax.copy()
    et_mask = pred == et_class
    if et_mask.sum() == 0:
        return pred
    filled = binary_fill_holes(et_mask)
    pred[filled & ~et_mask] = et_class
    return pred


def enforce_region_hierarchy(pred_argmax: np.ndarray) -> np.ndarray:
    """Remove tiny isolated tumor islands (< 10 voxels)."""
    pred = pred_argmax.copy()
    wt_mask = pred > 0
    if wt_mask.sum() == 0:
        return pred
    labeled_wt, n_wt = ndimage_label(wt_mask)
    for i in range(1, n_wt + 1):
        comp = labeled_wt == i
        if comp.sum() < 10:
            pred[comp] = 0
    return pred


def dilate_et_within_tc(
    pred_argmax: np.ndarray, et_class: int = 3, necrotic_class: int = 1, iterations: int = 1,
) -> np.ndarray:
    """Dilate ET into NCR region (class 1→3 stays within TC/WT definitions)."""
    pred = pred_argmax.copy()
    et_mask = pred == et_class
    if et_mask.sum() == 0:
        return pred
    tc_mask = (pred == et_class) | (pred == necrotic_class)
    dilated = binary_dilation(et_mask, structure=_STRUCT_3D, iterations=iterations)
    pred[dilated & tc_mask & ~et_mask] = et_class
    return pred


def postprocess_brats_advanced(
    pred_argmax: np.ndarray,
    softmax_probs: np.ndarray = None,
    wt_min_size: int = 500,
    tc_min_size: int = 200,
    et_min_size: int = 200,
    wt_conf_thresh: float = 0.0,
    et_conf_thresh: float = 0.0,
    fill_wt_holes: bool = True,
    fill_et_holes: bool = True,
    et_wt_ratio_thresh: float = 0.0,
) -> np.ndarray:
    """BraTS championship-level post-processing pipeline.

    Top-down hierarchical processing:
      1. WT connected-component filtering
      2. WT hole filling
      3. TC connected-component filtering
      4. ET connected-component filtering (small ET → NCR)
      5. ET confidence filtering (low-prob ET → NCR)
      6. ET hole filling
      7. Hierarchy enforcement (isolated ET in edema → reclassify)

    BraTS labels: 0=bg, 1=NCR, 2=ED, 3=ET
    Regions: WT={1,2,3}, TC={1,3}, ET={3}

    Args:
        pred_argmax: [D, H, W] integer label map (0-3).
        softmax_probs: [4, D, H, W] class probabilities (optional).
        wt_min_size: min voxels to keep a WT component.
        tc_min_size: min voxels to keep a TC component.
        et_min_size: min voxels to keep an ET component.
        wt_conf_thresh: mean WT probability threshold (0=disabled).
        et_conf_thresh: mean ET probability threshold (0=disabled).
        fill_wt_holes: fill holes in WT mask.
        fill_et_holes: fill holes in ET mask.
        et_wt_ratio_thresh: if ET/WT volume ratio < this, relabel all ET as NCR (0=disabled).
            BraTS winners use 0.03-0.04. Catches false positive ET in non-enhancing tumors.
    """
    pred = pred_argmax.copy()

    # Step 1: WT connected-component filtering
    wt_mask = pred > 0
    if wt_mask.sum() == 0:
        return pred

    labeled_wt, n_wt = ndimage_label(wt_mask)
    for i in range(1, n_wt + 1):
        comp = labeled_wt == i
        if comp.sum() < wt_min_size:
            if softmax_probs is not None and wt_conf_thresh > 0:
                wt_prob = 1.0 - softmax_probs[0][comp].mean()
                if wt_prob < wt_conf_thresh:
                    pred[comp] = 0
            else:
                pred[comp] = 0

    # Step 2: WT hole filling
    if fill_wt_holes:
        wt_mask = pred > 0
        if wt_mask.sum() > 0:
            filled = binary_fill_holes(wt_mask)
            pred[filled & ~wt_mask] = 2  # new WT voxels → edema

    # Step 3: TC connected-component filtering
    tc_mask = (pred == 1) | (pred == 3)
    if tc_mask.sum() > 0:
        labeled_tc, n_tc = ndimage_label(tc_mask)
        for i in range(1, n_tc + 1):
            comp = labeled_tc == i
            if comp.sum() < tc_min_size:
                pred[comp] = 2  # small TC → edema (stays in WT)

    # Step 4: ET connected-component filtering
    et_mask = pred == 3
    if et_mask.sum() > 0:
        labeled_et, n_et = ndimage_label(et_mask)
        for i in range(1, n_et + 1):
            comp = labeled_et == i
            if comp.sum() < et_min_size:
                pred[comp] = 1  # small ET → NCR

    # Step 4.5: ET/WT ratio check (BraTS winner trick)
    if et_wt_ratio_thresh > 0:
        et_vol = (pred == 3).sum()
        wt_vol = (pred > 0).sum()
        if wt_vol > 0 and et_vol / wt_vol < et_wt_ratio_thresh:
            pred[pred == 3] = 1  # All ET → NCR

    # Step 5: ET confidence filtering
    if softmax_probs is not None and et_conf_thresh > 0:
        et_mask = pred == 3
        if et_mask.sum() > 0:
            labeled_et, n_et = ndimage_label(et_mask)
            for i in range(1, n_et + 1):
                comp = labeled_et == i
                if softmax_probs[3][comp].mean() < et_conf_thresh:
                    pred[comp] = 1

    # Step 6: ET hole filling
    if fill_et_holes:
        et_mask = pred == 3
        if et_mask.sum() > 0:
            filled = binary_fill_holes(et_mask)
            new_et = filled & ~et_mask
            tc_context = (pred == 1) | (pred == 3)
            pred[new_et & tc_context] = 3

    # Step 7: Hierarchy enforcement
    et_mask = pred == 3
    if et_mask.sum() > 0:
        # Isolated ET with no NCR neighbor → suspicious
        labeled_et, n_et = ndimage_label(et_mask)
        ncr_mask = pred == 1
        for i in range(1, n_et + 1):
            comp = labeled_et == i
            dilated = binary_dilation(comp, structure=_STRUCT_3D, iterations=1)
            has_ncr = (dilated & ~comp & ncr_mask).any()
            if not has_ncr and comp.sum() < et_min_size * 2:
                pred[comp] = 2  # isolated ET in edema → reclassify

    return pred
