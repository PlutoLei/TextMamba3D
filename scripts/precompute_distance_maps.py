#!/usr/bin/env python3
"""Precompute signed distance maps for all BraTS cases.

Usage: python scripts/precompute_distance_maps.py --data-dir <path>
"""
import argparse
import os
import numpy as np
import nibabel as nib
from losses.boundary_loss import compute_distance_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--num-classes', type=int, default=4)
    args = parser.parse_args()

    cases = sorted(d for d in os.listdir(args.data_dir)
                   if os.path.isdir(os.path.join(args.data_dir, d))
                   and d.startswith('BraTS'))
    print(f'Found {len(cases)} cases')

    for i, case in enumerate(cases):
        case_dir = os.path.join(args.data_dir, case)
        seg_path = os.path.join(case_dir, f'{case}_seg.nii')
        if not os.path.exists(seg_path):
            seg_path += '.gz'
        if not os.path.exists(seg_path):
            print(f'  SKIP {case}: no seg file')
            continue

        out_path = os.path.join(case_dir, f'{case}_distance_map.npy')
        if os.path.exists(out_path):
            continue

        mask = nib.load(seg_path).get_fdata().astype(np.int64)
        mask[mask == 4] = 3

        dist = compute_distance_map(mask, num_classes=args.num_classes)
        np.save(out_path, dist)

        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(cases)} done')

    print(f'Done: {len(cases)} cases')


if __name__ == '__main__':
    main()
