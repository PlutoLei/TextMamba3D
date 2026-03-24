# data/et_quantitative_text.py
"""Quantitative ET text generation from segmentation masks.

Computes ET statistics (volume ratio, cluster count, spatial location,
size category) and generates structured English text. Pure numpy -- no
torch dependency. Used in two contexts:
  1. Offline: precompute for all training cases
  2. Online: in __getitem__ after transforms (reflects augmented data)

BraTS labels: 0=bg, 1=NCR, 2=ED, 3=ET
"""

import numpy as np
from scipy.ndimage import label as ndimage_label


def compute_et_stats(mask: np.ndarray) -> dict:
    """Compute ET statistics from a BraTS label map.

    Args:
        mask: [D, H, W] integer label map (0-3).

    Returns:
        dict with keys: et_voxels, wt_voxels, tc_voxels, et_ratio,
        et_tc_ratio, n_et_clusters, size_category, side, position.
    """
    et_mask = mask == 3
    wt_mask = mask > 0
    tc_mask = (mask == 1) | (mask == 3)

    et_voxels = int(et_mask.sum())
    wt_voxels = int(wt_mask.sum())
    tc_voxels = int(tc_mask.sum())

    et_ratio = et_voxels / max(1, wt_voxels)
    et_tc_ratio = et_voxels / max(1, tc_voxels)

    if et_voxels > 0:
        _, n_clusters = ndimage_label(et_mask)
    else:
        n_clusters = 0

    if et_voxels == 0:
        size_cat = "absent"
    elif et_voxels < 50:
        size_cat = "minimal"
    elif et_voxels < 500:
        size_cat = "small"
    elif et_voxels < 2000:
        size_cat = "moderate"
    else:
        size_cat = "large"

    side = ""
    position = ""
    if et_voxels > 0:
        from scipy.ndimage import center_of_mass
        centroid = np.array(center_of_mass(et_mask))
        norm = centroid / np.array(mask.shape)
        # BraTS [D, H, W]: axis 0=depth(SI), axis 1=height(AP), axis 2=width(LR)
        side = "left" if norm[2] > 0.5 else "right"
        ap = "anterior" if norm[1] > 0.5 else "posterior"
        si = "superior" if norm[0] > 0.5 else "inferior"
        position = f"{ap} {si}"

    return {
        "et_voxels": et_voxels,
        "wt_voxels": wt_voxels,
        "tc_voxels": tc_voxels,
        "et_ratio": float(et_ratio),
        "et_tc_ratio": float(et_tc_ratio),
        "n_et_clusters": n_clusters,
        "size_category": size_cat,
        "side": side,
        "position": position,
    }


def generate_quantitative_et_text(mask: np.ndarray) -> str:
    """Generate quantitative ET description from a BraTS label map.

    Args:
        mask: [D, H, W] integer label map (0-3).

    Returns:
        English text describing ET quantitatively.
    """
    stats = compute_et_stats(mask)

    if stats["et_voxels"] == 0:
        return "Enhancing tumor component is absent in this case."

    parts = []

    ratio_pct = stats["et_ratio"] * 100
    tc_pct = stats["et_tc_ratio"] * 100
    parts.append(
        f"The enhancing component constitutes {ratio_pct:.1f}% of the "
        f"whole tumor volume ({stats['size_category']} enhancement)"
    )

    parts.append(f"representing {tc_pct:.1f}% of the tumor core")

    n = stats["n_et_clusters"]
    if n == 1:
        parts.append("forming a single contiguous cluster")
    else:
        parts.append(f"distributed across {n} separate clusters")

    if stats["side"] and stats["position"]:
        parts.append(
            f"centered in the {stats['side']} {stats['position']} region"
        )

    return ", ".join(parts) + "."
