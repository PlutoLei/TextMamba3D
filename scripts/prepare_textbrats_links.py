#!/usr/bin/env python3
"""
Prepare a unified TextBraTS training directory by matching BraTS2020 MRI cases
with local TextBraTS text files by case name.

By default this script creates symlinks for all matched files, so the large MRI
files stay on the Windows drive and are not copied into the WSL workspace.

Example:
    python scripts/prepare_textbrats_links.py \
        --mri-root /mnt/d/Brats2020/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData
"""

from __future__ import annotations

import argparse
from pathlib import Path


MRI_SUFFIXES = ("_t1", "_t1ce", "_t2", "_flair", "_seg")
TEXT_SUFFIXES = ("_flair_text.txt", "_flair_text.npy")
MRI_EXTENSIONS = (".nii.gz", ".nii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Link BraTS2020 MRI files and TextBraTS text files into one dataset root."
    )
    parser.add_argument(
        "--mri-root",
        type=Path,
        default=Path("/mnt/d/Brats2020/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"),
        help="Root directory containing BraTS20_Training_* MRI case folders.",
    )
    parser.add_argument(
        "--text-root",
        type=Path,
        default=Path("data/TextBraTS/TextBraTSData"),
        help="Root directory containing TextBraTS text case folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"),
        help="Output directory that will contain matched case folders.",
    )
    parser.add_argument(
        "--copy-text",
        action="store_true",
        help="Copy text files instead of symlinking them.",
    )
    parser.add_argument(
        "--copy-all",
        action="store_true",
        help="Copy all files instead of creating symlinks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only prepare the first N matched cases for quick smoke tests.",
    )
    return parser.parse_args()


def resolve_mri_file(case_dir: Path, case_name: str, suffix: str) -> Path | None:
    for ext in MRI_EXTENSIONS:
        path = case_dir / f"{case_name}{suffix}{ext}"
        if path.exists():
            return path
    return None


def ensure_link_or_copy(src: Path, dst: Path, copy_file: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if copy_file:
        dst.write_bytes(src.read_bytes())
    else:
        dst.symlink_to(src.resolve())


def collect_case_pairs(mri_root: Path, text_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for mri_case_dir in sorted(mri_root.glob("BraTS20_Training_*")):
        if not mri_case_dir.is_dir():
            continue
        case_name = mri_case_dir.name
        text_case_dir = text_root / case_name
        if text_case_dir.is_dir():
            pairs.append((mri_case_dir, text_case_dir))
    return pairs


def main() -> int:
    args = parse_args()

    if not args.mri_root.exists():
        print(f"Error: MRI root not found: {args.mri_root}")
        return 1

    if not args.text_root.exists():
        print(f"Error: text root not found: {args.text_root}")
        return 1

    pairs = collect_case_pairs(args.mri_root, args.text_root)
    if args.limit is not None:
        pairs = pairs[:args.limit]

    print(f"MRI root:    {args.mri_root}")
    print(f"Text root:   {args.text_root}")
    print(f"Output root: {args.output_root}")
    print(f"Matched cases: {len(pairs)}")

    if not pairs:
        print("No matched cases found.")
        return 1

    missing_required = 0
    linked_cases = 0

    for mri_case_dir, text_case_dir in pairs:
        case_name = mri_case_dir.name
        out_case_dir = args.output_root / case_name

        required_files: list[Path] = []
        for suffix in MRI_SUFFIXES:
            mri_file = resolve_mri_file(mri_case_dir, case_name, suffix)
            if mri_file is None:
                print(f"[SKIP] Missing MRI file for {case_name}: {suffix}")
                missing_required += 1
                required_files = []
                break
            required_files.append(mri_file)

        if not required_files:
            continue

        text_txt = text_case_dir / f"{case_name}_flair_text.txt"
        if not text_txt.exists():
            print(f"[SKIP] Missing text file for {case_name}: {text_txt.name}")
            missing_required += 1
            continue

        out_case_dir.mkdir(parents=True, exist_ok=True)

        for src in required_files:
            ensure_link_or_copy(src, out_case_dir / src.name, copy_file=args.copy_all)

        for suffix in TEXT_SUFFIXES:
            src = text_case_dir / f"{case_name}{suffix}"
            if not src.exists():
                continue
            copy_text = args.copy_all or args.copy_text
            ensure_link_or_copy(src, out_case_dir / src.name, copy_file=copy_text)

        linked_cases += 1

    print()
    print(f"Prepared cases: {linked_cases}")
    print(f"Skipped cases:  {missing_required}")
    print(f"Done. Dataset root ready at: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
