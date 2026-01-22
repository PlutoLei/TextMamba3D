#!/usr/bin/env python3
"""
BraTS 2021 数据集准备脚本

使用方法:
    1. 从官方渠道下载数据集:
       - Synapse: https://www.synapse.org/#!Synapse:syn25829067
       - Kaggle: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1

    2. 运行此脚本组织数据:
       python scripts/prepare_brats.py --input /path/to/downloaded/data --output data/BraTS2021

作者: TextMamba3D Team
日期: 2026-01-22
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare BraTS 2021 dataset')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to downloaded BraTS data')
    parser.add_argument('--output', type=str, default='data/BraTS2021',
                        help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--copy', action='store_true',
                        help='Copy files instead of symlink')
    return parser.parse_args()


def find_cases(input_dir: str) -> list:
    """Find all case directories in input."""
    cases = []
    input_path = Path(input_dir)

    # Try different directory structures
    for pattern in ['BraTS*', 'RSNA*', '*']:
        for case_dir in input_path.glob(pattern):
            if case_dir.is_dir():
                # Check if it contains required files
                files = list(case_dir.glob('*.nii.gz'))
                if len(files) >= 4:  # At least 4 modalities
                    cases.append(case_dir)

    # Remove duplicates
    cases = list(set(cases))
    return sorted(cases, key=lambda x: x.name)


def verify_case(case_dir: Path) -> bool:
    """Verify a case has all required files."""
    case_name = case_dir.name
    required_files = [
        f'{case_name}_t1.nii.gz',
        f'{case_name}_t1ce.nii.gz',
        f'{case_name}_t2.nii.gz',
        f'{case_name}_flair.nii.gz',
        f'{case_name}_seg.nii.gz',
    ]

    for f in required_files:
        if not (case_dir / f).exists():
            # Try alternative naming
            alt_name = f.replace('_', '-')
            if not (case_dir / alt_name).exists():
                return False
    return True


def link_or_copy(src: Path, dst: Path, copy: bool = False):
    """Create symlink or copy file."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    if copy:
        shutil.copy2(src, dst)
    else:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.absolute())


def prepare_dataset(args):
    """Main function to prepare dataset."""
    random.seed(args.seed)

    # Find all cases
    print(f"Scanning {args.input} for cases...")
    cases = find_cases(args.input)

    if not cases:
        print("Error: No valid cases found!")
        print("Expected directory structure:")
        print("  input_dir/")
        print("    BraTS2021_00000/")
        print("      BraTS2021_00000_t1.nii.gz")
        print("      BraTS2021_00000_t1ce.nii.gz")
        print("      BraTS2021_00000_t2.nii.gz")
        print("      BraTS2021_00000_flair.nii.gz")
        print("      BraTS2021_00000_seg.nii.gz")
        return

    # Verify cases
    valid_cases = [c for c in cases if verify_case(c)]
    print(f"Found {len(valid_cases)} valid cases out of {len(cases)} total")

    if not valid_cases:
        print("Error: No valid cases with all required files!")
        return

    # Shuffle and split
    random.shuffle(valid_cases)
    n = len(valid_cases)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_cases = valid_cases[:n_train]
    val_cases = valid_cases[n_train:n_train + n_val]
    test_cases = valid_cases[n_train + n_val:]

    print(f"\nSplit: train={len(train_cases)}, val={len(val_cases)}, test={len(test_cases)}")

    # Create output directories
    output_path = Path(args.output)
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    # Process each split
    splits = [
        ('train', train_cases),
        ('val', val_cases),
        ('test', test_cases),
    ]

    for split_name, split_cases in splits:
        print(f"\nProcessing {split_name}...")
        for case_dir in tqdm(split_cases):
            case_name = case_dir.name
            dst_dir = output_path / split_name / case_name
            dst_dir.mkdir(parents=True, exist_ok=True)

            # Link/copy all nii.gz files
            for f in case_dir.glob('*.nii.gz'):
                link_or_copy(f, dst_dir / f.name, args.copy)

    # Summary
    print("\n" + "="*50)
    print("Dataset preparation complete!")
    print(f"Output: {output_path.absolute()}")
    print(f"  train: {len(train_cases)} cases")
    print(f"  val: {len(val_cases)} cases")
    print(f"  test: {len(test_cases)} cases")
    print("="*50)

    # Create info file
    info_file = output_path / 'dataset_info.txt'
    with open(info_file, 'w') as f:
        f.write(f"BraTS 2021 Dataset\n")
        f.write(f"==================\n\n")
        f.write(f"Source: {args.input}\n")
        f.write(f"Seed: {args.seed}\n\n")
        f.write(f"Split:\n")
        f.write(f"  train: {len(train_cases)} cases ({args.train_ratio*100:.0f}%)\n")
        f.write(f"  val: {len(val_cases)} cases ({args.val_ratio*100:.0f}%)\n")
        f.write(f"  test: {len(test_cases)} cases ({(1-args.train_ratio-args.val_ratio)*100:.0f}%)\n\n")
        f.write(f"Cases:\n")
        for split_name, split_cases in splits:
            f.write(f"\n[{split_name}]\n")
            for case in split_cases:
                f.write(f"  {case.name}\n")


def download_instructions():
    """Print download instructions."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              BraTS 2021 数据集下载说明                           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  方式1: Synapse官方下载 (推荐)                                   ║
║  ────────────────────────────────────────────────────────────   ║
║  1. 访问 https://www.synapse.org/#!Synapse:syn25829067          ║
║  2. 注册Synapse账号并登录                                        ║
║  3. 加入BraTS 2021挑战赛                                        ║
║  4. 下载 BraTS2021_Training_Data.tar                            ║
║                                                                  ║
║  方式2: Kaggle镜像下载                                           ║
║  ────────────────────────────────────────────────────────────   ║
║  1. 访问 https://www.kaggle.com/datasets/dschettler8845/        ║
║     brats-2021-task1                                            ║
║  2. 登录Kaggle账号                                               ║
║  3. 点击Download下载数据集                                       ║
║                                                                  ║
║  下载后运行:                                                     ║
║  python scripts/prepare_brats.py --input <下载路径> \\           ║
║                                  --output data/BraTS2021         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        download_instructions()
    else:
        prepare_dataset(args)
