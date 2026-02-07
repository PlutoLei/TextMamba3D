# preprocess_data.py
"""Preprocess BraTS2021 NIfTI files to .pt format for faster loading."""

import os
import argparse
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_case(case_dir: str, output_dir: str) -> str:
    """Process a single case."""
    case_name = os.path.basename(case_dir)
    output_path = os.path.join(output_dir, f"{case_name}.pt")

    if os.path.exists(output_path):
        return f"{case_name}: skipped (exists)"

    modalities = ['t1', 't1ce', 't2', 'flair']

    # Load modalities
    images = []
    for mod in modalities:
        path = os.path.join(case_dir, f"{case_name}_{mod}.nii.gz")
        img = nib.load(path).get_fdata()
        images.append(img)

    image = np.stack(images, axis=0).astype(np.float32)  # [4, D, H, W]

    # Load segmentation
    seg_path = os.path.join(case_dir, f"{case_name}_seg.nii.gz")
    mask = nib.load(seg_path).get_fdata().astype(np.int64)
    mask[mask == 4] = 3  # BraTS label mapping

    # Normalize each modality
    for i in range(image.shape[0]):
        img_i = image[i]
        nonzero = img_i[img_i > 0]
        if len(nonzero) > 0:
            mean, std = nonzero.mean(), nonzero.std()
            image[i] = (img_i - mean) / (std + 1e-8)

    # Save as .pt
    data = {
        'image': torch.from_numpy(image),
        'mask': torch.from_numpy(mask),
        'case_name': case_name,
    }
    torch.save(data, output_path)

    return f"{case_name}: done"


def main():
    parser = argparse.ArgumentParser(description="Preprocess BraTS2021 to .pt")
    parser.add_argument("--data-dir", type=str, default="./data/BraTS2021",
                        help="Path to BraTS2021 directory")
    parser.add_argument("--output-dir", type=str, default="./data/BraTS2021_pt",
                        help="Output directory for .pt files")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    args = parser.parse_args()

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(args.data_dir, split)
        output_split_dir = os.path.join(args.output_dir, split)

        if not os.path.exists(split_dir):
            print(f"Skipping {split}: not found")
            continue

        os.makedirs(output_split_dir, exist_ok=True)

        # Find all cases
        cases = []
        for case_name in sorted(os.listdir(split_dir)):
            case_dir = os.path.join(split_dir, case_name)
            if os.path.isdir(case_dir):
                cases.append(case_dir)

        print(f"\nProcessing {split}: {len(cases)} cases")

        # Process in parallel
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_case, case, output_split_dir): case
                for case in cases
            }

            for future in tqdm(as_completed(futures), total=len(cases), desc=split):
                try:
                    result = future.result()
                except Exception as e:
                    case = futures[future]
                    print(f"\nError processing {case}: {e}")

    print("\nPreprocessing complete!")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
