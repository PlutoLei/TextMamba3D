# data/et_text_enrichment.py
"""
ET-Enriched Text Generation from T1ce Images (NOT GT masks).

Extracts enhancement features from T1ce MRI using image intensity analysis,
mimicking how radiologists identify enhancing tumor on contrast-enhanced T1.
Methodologically clean: no information leakage at test time.

Usage:
    python et_text_enrichment.py <data_dir> [output_dir]
"""

import os
import numpy as np
import nibabel as nib
from scipy import ndimage
from typing import Dict, Optional


def extract_enhancement_features(t1ce_path: str) -> Dict:
    """Extract enhancement-related features from T1ce MRI volume.

    Enhancing tumor (ET) appears as high-signal regions on T1ce due to
    contrast agent leakage through disrupted blood-brain barrier.
    We detect these regions using intensity thresholding (mean + 2*std),
    then characterize their morphology.
    """
    vol = nib.load(t1ce_path).get_fdata().astype(np.float32)

    # Brain mask (non-zero voxels)
    brain_mask = vol > 0
    brain_voxels = vol[brain_mask]

    if len(brain_voxels) == 0:
        return {"has_enhancement": False}

    mean = brain_voxels.mean()
    std = brain_voxels.std()

    # High-signal threshold: voxels > mean + 2*std
    # Captures contrast-enhancing regions on T1ce
    threshold = mean + 2.0 * std
    high_signal = (vol > threshold) & brain_mask

    n_high = int(high_signal.sum())
    n_brain = int(brain_mask.sum())

    if n_high < 10:
        return {"has_enhancement": False, "high_signal_ratio": 0.0}

    high_signal_ratio = n_high / n_brain

    # --- Pattern detection (rim vs solid) ---
    # Rim-enhancing: high-signal shell around lower-signal interior (necrosis)
    # Solid-enhancing: compact high-signal without interior void
    labeled, n_components = ndimage.label(high_signal)
    interior_ratio = 0.0
    pattern = "diffuse"

    if n_components > 0:
        component_sizes = ndimage.sum(
            high_signal, labeled, range(1, n_components + 1)
        )
        largest_idx = int(np.argmax(component_sizes)) + 1
        largest_component = labeled == largest_idx

        # Fill holes → interior = filled minus shell
        filled = ndimage.binary_fill_holes(largest_component)
        interior = filled & ~largest_component
        interior_ratio = float(interior.sum() / (filled.sum() + 1e-8))

        if interior_ratio > 0.15:
            pattern = "rim"
        elif interior_ratio > 0.05:
            pattern = "partial-rim"
        else:
            pattern = "solid"

    # --- Boundary regularity ---
    # Compare actual surface/volume ratio to an equivalent sphere
    struct = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(high_signal, struct)
    surface_voxels = int((high_signal.astype(int) - eroded.astype(int)).sum())
    volume_voxels = n_high

    r_equiv = (3 * volume_voxels / (4 * np.pi)) ** (1 / 3)
    sphere_sv = 3 / (r_equiv + 1e-8)
    actual_sv = surface_voxels / (volume_voxels + 1e-8)
    regularity_score = actual_sv / (sphere_sv + 1e-8)

    if regularity_score > 2.0:
        boundary = "irregular"
    elif regularity_score > 1.3:
        boundary = "moderately irregular"
    else:
        boundary = "relatively smooth"

    # --- Spatial location ---
    coords = np.argwhere(high_signal)
    centroid = coords.mean(axis=0)
    norm_centroid = centroid / np.array(vol.shape)

    side = "left" if norm_centroid[0] > 0.5 else "right"
    ap = "anterior" if norm_centroid[1] > 0.5 else "posterior"
    si = "superior" if norm_centroid[2] > 0.5 else "inferior"

    return {
        "has_enhancement": True,
        "high_signal_ratio": float(high_signal_ratio),
        "pattern": pattern,
        "boundary": boundary,
        "regularity_score": float(regularity_score),
        "interior_ratio": float(interior_ratio),
        "side": side,
        "position": f"{ap} {si}",
        "n_components": int(min(n_components, 20)),
    }


def generate_et_description(features: Dict) -> str:
    """Generate ET-specific English text description from image features."""
    if not features.get("has_enhancement", False):
        return "No significant contrast enhancement detected on T1ce."

    parts = []

    # Enhancement pattern
    pattern = features["pattern"]
    pattern_text = {
        "rim": "Ring-enhancing lesion identified on T1ce",
        "partial-rim": "Partially rim-enhancing lesion on T1ce",
        "solid": "Solid enhancing lesion on T1ce",
        "diffuse": "Diffuse enhancement on T1ce",
    }
    parts.append(pattern_text.get(pattern, "Enhancement on T1ce"))

    # Location
    side = features.get("side", "")
    position = features.get("position", "")
    if side and position:
        parts.append(f"in the {side} {position} region")

    # Boundary
    boundary = features.get("boundary", "")
    if boundary:
        parts.append(f"with {boundary} margins")

    # Enhancement extent
    ratio = features.get("high_signal_ratio", 0)
    if ratio > 0.05:
        parts.append("showing extensive enhancement")
    elif ratio > 0.02:
        parts.append("showing moderate enhancement")
    else:
        parts.append("showing focal enhancement")

    # Multi-focal
    n_comp = features.get("n_components", 1)
    if n_comp > 3:
        parts.append("with multifocal enhancing components")

    return ", ".join(parts) + "."


def process_all_cases(
    data_dir: str, output_dir: Optional[str] = None
) -> Dict[str, str]:
    """Process all BraTS cases and generate ET-enriched descriptions.

    Args:
        data_dir: Path containing BraTS20_Training_* directories.
        output_dir: If given, save per-case .txt files here.

    Returns:
        Dict mapping case_name -> ET description string.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = {}
    cases = sorted(
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

    for i, case_name in enumerate(cases):
        case_dir = os.path.join(data_dir, case_name)

        # Find T1ce file
        t1ce_path = os.path.join(case_dir, f"{case_name}_t1ce.nii")
        if not os.path.exists(t1ce_path):
            t1ce_path += ".gz"
        if not os.path.exists(t1ce_path):
            continue

        features = extract_enhancement_features(t1ce_path)
        description = generate_et_description(features)
        results[case_name] = description

        # Save alongside original data
        out_path = os.path.join(case_dir, f"{case_name}_et_enriched.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(description)

        if output_dir:
            out_copy = os.path.join(output_dir, f"{case_name}_et_enriched.txt")
            with open(out_copy, "w", encoding="utf-8") as f:
                f.write(description)

        if (i + 1) % 50 == 0 or i == len(cases) - 1:
            print(f"  Processed {i + 1}/{len(cases)}")

    print(f"Generated ET descriptions for {len(results)} cases")
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python et_text_enrichment.py <data_dir> [output_dir]")
        sys.exit(1)

    data_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    results = process_all_cases(data_dir, output_dir)

    # Print examples
    for name, desc in list(results.items())[:5]:
        print(f"  {name}: {desc}")
