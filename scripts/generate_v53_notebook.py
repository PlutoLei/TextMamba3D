"""Generate the TextMamba3D_A100_V5.3.ipynb Colab notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "TextMamba3D_A100_V5.3.ipynb"


def cell_0_title() -> nbformat.NotebookNode:
    return new_markdown_cell(
        dedent(
            """
            # TextMamba3D - A100 V5.3 Text-Aware Augmentation Pipeline

            **Goal:** Fine-tune V5.0 with text-aware Copy-Paste augmentation, ET oversampling, quantitative ET text, and advanced BraTS post-processing.

            | Stage | Cells | Description | GPU Time |
            |-------|-------|-------------|----------|
            | Setup | 1-3 | Mount Drive, clone repo, extract data, restore ET text | ~2 min |
            | Smoke Tests | 4-7 | Verify Copy-Paste, quantitative ET text, and advanced PP | ~3-5 min |
            | Training | 8-9 | V5.3 fine-tuning from V5.0 | ~11 hours |
            | Evaluation | 10-12 | Advanced PP sweep on val + 8-config test eval | ~1-2 hours |
            | Results | 13 | Compare against V5.0 baseline | instant |
            | Resume | 14-15 | Resume after disconnect | as needed |
            """
        ).strip()
    )


def cell_1_mount_and_install() -> nbformat.NotebookNode:
    return new_code_cell(
        dedent(
            """
            # Cell 1: Mount Drive + Install deps for V5.3
            from google.colab import drive
            drive.mount('/content/drive')

            !nvidia-smi 2>/dev/null || echo "No GPU"

            # V5.3 still uses the V5.0-compatible Mamba-2 path.
            !pip install -q mamba-ssm causal-conv1d einops \\
                transformers nibabel tensorboard pyyaml tqdm scipy nbformat
            """
        ).strip()
    )


def cell_2_clone_and_extract() -> nbformat.NotebookNode:
    return new_code_cell(
        dedent(
            """
            # Cell 2: Clone repo + extract data
            import os, zipfile, shutil, subprocess, time

            REPO_DIR = '/content/TextMamba3D'
            DRIVE_BASE = '/content/drive/MyDrive/TextMamba3D'

            git_dir = os.path.join(REPO_DIR, '.git')
            if os.path.isdir(REPO_DIR) and not os.path.isdir(git_dir):
                print(f'Removing old non-git code at {REPO_DIR}...')
                shutil.rmtree(REPO_DIR)

            if os.path.isdir(git_dir):
                os.chdir(REPO_DIR)
                subprocess.run(['git', 'pull'], check=True)
                print(f'Updated existing repo at {REPO_DIR}')
            else:
                for attempt in range(1, 4):
                    print(f'Cloning (attempt {attempt}/3)...')
                    ret = subprocess.run(
                        ['git', 'clone', '--depth', '1',
                         'https://github.com/PlutoLei/TextMamba3D.git', REPO_DIR],
                        capture_output=True, text=True
                    )
                    if ret.returncode == 0 and os.path.exists(
                        os.path.join(REPO_DIR, 'models/textmamba3d.py')
                    ):
                        break
                    print(f'  Failed (code {ret.returncode}): {ret.stderr.strip()}')
                    if os.path.isdir(REPO_DIR):
                        shutil.rmtree(REPO_DIR)
                    if attempt < 3:
                        time.sleep(5 * attempt)
                else:
                    raise RuntimeError(
                        f'Clone failed after 3 attempts. Last error: {ret.stderr.strip()}'
                    )
                os.chdir(REPO_DIR)
                print(f'Cloned to {REPO_DIR}')

            print(f'Working directory: {os.getcwd()}')

            DATA_ZIP = os.path.join(DRIVE_BASE, "TextBraTS_data.zip")
            DATA_DIR = os.path.join(
                REPO_DIR,
                "data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
            )

            if not os.path.exists(DATA_DIR):
                os.makedirs(os.path.dirname(DATA_DIR), exist_ok=True)
                if os.path.exists(DATA_ZIP):
                    print(f"Extracting {DATA_ZIP}...")
                    with zipfile.ZipFile(DATA_ZIP, 'r') as zf:
                        zf.extractall(os.path.dirname(DATA_DIR))
                    if os.path.exists(DATA_DIR):
                        print(f"Data extracted. Cases: {len(os.listdir(DATA_DIR))}")
                    else:
                        print(f"ERROR: Expected path not found: {DATA_DIR}")
                else:
                    print(f"ERROR: {DATA_ZIP} not found on Drive")
            else:
                print(f"Data already exists. Cases: {len(os.listdir(DATA_DIR))}")

            if os.path.exists(DATA_DIR):
                cases = [d for d in os.listdir(DATA_DIR)
                         if os.path.isdir(os.path.join(DATA_DIR, d))]
                print(f"Total BraTS cases: {len(cases)}")
            """
        ).strip()
    )


def cell_3_restore_et_text() -> nbformat.NotebookNode:
    return new_code_cell(
        dedent(
            """
            # Cell 3: ET-enriched text restore for V5.3
            import os, sys, zipfile
            os.chdir(REPO_DIR)

            DATA_DIR = "./data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
            ET_CACHE_ZIP = os.path.join(DRIVE_BASE, "et_enriched.zip")

            cases = sorted(
                d for d in os.listdir(DATA_DIR)
                if os.path.isdir(os.path.join(DATA_DIR, d))
            ) if os.path.isdir(DATA_DIR) else []
            if not cases:
                raise RuntimeError(f"No BraTS cases found in {DATA_DIR}")

            sample_enriched = os.path.join(
                DATA_DIR, cases[0], f"{cases[0]}_et_enriched.txt"
            )

            if os.path.exists(sample_enriched):
                count = sum(
                    1 for d in cases
                    if os.path.exists(os.path.join(DATA_DIR, d, f"{d}_et_enriched.txt"))
                )
                print(f"ET-enriched text already present for {count} cases, skipping")
            elif os.path.exists(ET_CACHE_ZIP):
                print(f"Restoring ET-enriched text from {ET_CACHE_ZIP}...")
                with zipfile.ZipFile(ET_CACHE_ZIP, 'r') as zf:
                    zf.extractall(DATA_DIR)
                count = sum(
                    1 for d in cases
                    if os.path.exists(os.path.join(DATA_DIR, d, f"{d}_et_enriched.txt"))
                )
                print(f"Restored ET-enriched text for {count} cases")
            else:
                print("Generating ET-enriched text descriptions from T1ce images...")
                sys.path.insert(0, '.')
                from data.et_text_enrichment import process_all_cases
                results = process_all_cases(DATA_DIR)
                print(f"Generated for {len(results)} cases")
                with zipfile.ZipFile(ET_CACHE_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for case_dir in cases:
                        et_file = os.path.join(
                            DATA_DIR, case_dir, f"{case_dir}_et_enriched.txt"
                        )
                        if os.path.exists(et_file):
                            zf.write(
                                et_file,
                                os.path.join(case_dir, f"{case_dir}_et_enriched.txt")
                            )
                print(f"Cached ET text to {ET_CACHE_ZIP}")

            et_count = sum(
                1 for d in cases
                if os.path.exists(os.path.join(DATA_DIR, d, f'{d}_et_enriched.txt'))
            )
            if et_count == 0:
                raise RuntimeError(
                    'No ET-enriched text files found. Run text generation first.'
                )
            print(f'ET-enriched text verified: {et_count}/{len(cases)} cases')
            """
        ).strip()
    )


def cell_4_smoke_tests_header() -> nbformat.NotebookNode:
    return new_markdown_cell(
        dedent(
            """
            ## Smoke Tests: Verify V5.3 Features

            Run these before training so Copy-Paste augmentation, quantitative ET text,
            and advanced post-processing are validated in isolation.
            """
        ).strip()
    )


def cell_5_copy_paste_smoke_test() -> nbformat.NotebookNode:
    return new_code_cell(
        dedent(
            """
            # Cell 5: Test 1 - TumorCopyPaste3D
            import os
            import sys
            import numpy as np
            import torch

            os.chdir(REPO_DIR)
            sys.path.insert(0, REPO_DIR)

            from data.transforms import TumorCopyPaste3D


            def make_sphere_mask(shape, center, radius, label=3):
                zz, yy, xx = torch.meshgrid(
                    torch.arange(shape[0]),
                    torch.arange(shape[1]),
                    torch.arange(shape[2]),
                    indexing='ij',
                )
                dist2 = (
                    (zz - center[0]) ** 2
                    + (yy - center[1]) ** 2
                    + (xx - center[2]) ** 2
                )
                mask = torch.zeros(shape, dtype=torch.long)
                mask[dist2 <= radius ** 2] = label
                return mask


            def make_donor(center, et_radius, wt_radius):
                image = torch.zeros((4, 128, 128, 128), dtype=torch.float32)
                mask = torch.zeros((128, 128, 128), dtype=torch.long)
                wt = make_sphere_mask(mask.shape, center=center, radius=wt_radius, label=2)
                et = make_sphere_mask(mask.shape, center=center, radius=et_radius, label=3)
                mask[wt > 0] = 2
                mask[et > 0] = 3
                for channel in range(4):
                    image[channel] = (mask > 0).float() * (0.2 + 0.1 * channel)
                    image[channel] += (mask == 3).float() * (0.4 + 0.05 * channel)
                return image, mask


            np.random.seed(42)
            torch.manual_seed(42)

            cp = TumorCopyPaste3D(
                prob=1.0,
                bank_size=10,
                blend_sigma=0.0,
                min_et_voxels=200,
                paste_jitter=0,
            )

            donors = [
                make_donor(center=(36, 42, 48), et_radius=4, wt_radius=8),
                make_donor(center=(72, 68, 60), et_radius=5, wt_radius=9),
                make_donor(center=(92, 84, 86), et_radius=4, wt_radius=7),
            ]
            for donor_image, donor_mask in donors:
                crop = cp._extract_tumor_crop(donor_image, donor_mask)
                assert crop is not None, "Expected ET-rich donor crop"
                cp.donor_bank.append(crop)

            assert len(cp.donor_bank) >= 3, "TumorCopyPaste3D donor bank should be prefilled"

            target_image = torch.zeros((4, 128, 128, 128), dtype=torch.float32)
            target_mask = torch.zeros((128, 128, 128), dtype=torch.long)
            et_before = int((target_mask == 3).sum().item())

            aug_image, aug_mask = cp(target_image, target_mask)
            et_after = int((aug_mask == 3).sum().item())

            assert aug_image.shape == target_image.shape
            assert aug_mask.shape == target_mask.shape
            assert et_after != et_before, "Copy-paste should change ET voxel count"

            print(f"Donor bank size: {len(cp.donor_bank)}")
            print(f"ET voxels: before={et_before}, after={et_after}")
            print("TumorCopyPaste3D smoke test passed")
            """
        ).strip()
    )


def cell_6_quantitative_text_smoke_test() -> nbformat.NotebookNode:
    return new_code_cell(
        dedent(
            """
            # Cell 6: Test 2 - Quantitative ET Text
            import os
            import sys
            import nibabel as nib
            import numpy as np

            os.chdir(REPO_DIR)
            sys.path.insert(0, REPO_DIR)

            from data.et_quantitative_text import (
                compute_et_stats,
                generate_quantitative_et_text,
            )

            DATA_DIR = os.path.join(
                REPO_DIR,
                "data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
            )


            def load_case_mask(case_name):
                seg_path = os.path.join(DATA_DIR, case_name, f"{case_name}_seg.nii")
                if not os.path.exists(seg_path):
                    seg_path += ".gz"
                if not os.path.exists(seg_path):
                    raise FileNotFoundError(f"Missing seg file for {case_name}")
                seg = nib.load(seg_path).get_fdata().astype(np.int16)
                seg[seg == 4] = 3
                return seg


            def find_case(predicate, preferred=None, exclude=None):
                exclude = set(exclude or [])
                if preferred and preferred not in exclude:
                    try:
                        mask = load_case_mask(preferred)
                        if predicate(mask):
                            return preferred, mask
                    except FileNotFoundError:
                        pass
                for case_name in sorted(os.listdir(DATA_DIR)):
                    case_dir = os.path.join(DATA_DIR, case_name)
                    if not os.path.isdir(case_dir) or case_name in exclude:
                        continue
                    try:
                        mask = load_case_mask(case_name)
                    except FileNotFoundError:
                        continue
                    if predicate(mask):
                        return case_name, mask
                return None, None


            pos_case, pos_mask = find_case(lambda m: np.any(m == 3), preferred="BraTS20_Training_001")
            assert pos_case is not None, "No ET-positive case found"
            pos_stats = compute_et_stats(pos_mask)
            pos_text = generate_quantitative_et_text(pos_mask)
            assert "%" in pos_text, "ET-positive text should include percentages"

            neg_case, neg_mask = find_case(lambda m: not np.any(m == 3), exclude={pos_case})
            assert neg_case is not None, "No ET-absent case found"
            neg_stats = compute_et_stats(neg_mask)
            neg_text = generate_quantitative_et_text(neg_mask)
            assert "absent" in neg_text.lower(), "ET-absent text should say absent"

            extra_case, extra_mask = find_case(
                lambda m: np.any(m == 3),
                exclude={pos_case, neg_case},
            )
            assert extra_case is not None, "Need a second ET-positive case for comparison"
            extra_stats = compute_et_stats(extra_mask)
            extra_text = generate_quantitative_et_text(extra_mask)

            print("Sample outputs:")
            for case_name, stats, text in [
                (pos_case, pos_stats, pos_text),
                (extra_case, extra_stats, extra_text),
                (neg_case, neg_stats, neg_text),
            ]:
                print("-" * 80)
                print(case_name)
                print(
                    f"  ET voxels={stats['et_voxels']}, WT voxels={stats['wt_voxels']}, "
                    f"ET ratio={stats['et_ratio'] * 100:.2f}%, clusters={stats['n_et_clusters']}"
                )
                print(f"  Text: {text}")

            print("Quantitative ET text smoke test passed")
            """
        ).strip()
    )


def cell_7_advanced_pp_and_dry_run() -> nbformat.NotebookNode:
    return new_code_cell(
        dedent(
            """
            # Cell 7: Test 3 - Advanced PP + Config Dry-Run
            import os
            import sys
            import yaml
            import tempfile
            import subprocess
            import numpy as np

            os.chdir(REPO_DIR)
            sys.path.insert(0, REPO_DIR)

            from utils.postprocess import postprocess_brats_advanced

            pred = np.zeros((64, 64, 64), dtype=np.int64)
            pred[16:48, 16:48, 16:48] = 2
            pred[24:40, 24:40, 24:40] = 1

            pred[28:31, 28:31, 28:31] = 3
            pred[36:40, 18:22, 18:22] = 3

            processed = postprocess_brats_advanced(
                pred,
                wt_min_size=10,
                tc_min_size=20,
                et_min_size=50,
                et_conf_thresh=0.0,
                et_wt_ratio_thresh=0.0,
            )

            assert np.all(processed[28:31, 28:31, 28:31] == 1), "Small ET island should be reclassified to NCR"
            assert np.all(processed[36:40, 18:22, 18:22] == 2), "Isolated ET island should be reclassified to edema"
            print("Advanced PP synthetic test passed")

            config_path = os.path.join(REPO_DIR, "configs", "a100_v5.3.yaml")
            with open(config_path, "r", encoding="utf-8") as f:
                smoke_cfg = yaml.safe_load(f)
            smoke_cfg["augmentation"]["use_et_oversample"] = False

            with tempfile.NamedTemporaryFile(
                "w",
                suffix=".yaml",
                prefix="a100_v5.3_smoke_",
                dir=os.path.join(REPO_DIR, "configs"),
                delete=False,
                encoding="utf-8",
            ) as tmp:
                yaml.safe_dump(smoke_cfg, tmp, sort_keys=False)
                smoke_config_path = tmp.name

            cmd = [
                "python", "-u", "train.py",
                "--config", smoke_config_path,
                "--max-samples", "4",
                "--max-epochs", "1",
            ]
            ret = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True)

            tail = ret.stdout[-1500:] if ret.stdout else ""
            if tail:
                print(tail)
            if ret.returncode != 0:
                err_tail = ret.stderr[-1500:] if ret.stderr else ""
                raise RuntimeError(
                    "V5.3 dry run failed\\n"
                    f"STDOUT tail:\\n{tail}\\n\\nSTDERR tail:\\n{err_tail}"
                )

            print("Dry run passed")
            """
        ).strip()
    )


def cell_8_training_header() -> nbformat.NotebookNode:
    return new_markdown_cell(
        dedent(
            """
            ## V5.3 Training: Text-Aware Copy-Paste + ET Oversample

            Fine-tune from `best_v5.0.pth` with the V5.3 config:
            Copy-Paste augmentation, ET oversampling, and quantitative ET text.
            Early stopping focuses on ET Dice.
            """
        ).strip()
    )


def cell_9_training() -> nbformat.NotebookNode:
    return new_code_cell(
        dedent(
            """
            # Cell 9: V5.3 Training
            import glob
            import os
            import shutil
            import subprocess

            os.chdir(REPO_DIR)
            DRIVE_CKPT = os.path.join(DRIVE_BASE, "checkpoints")
            V50_CKPT = os.path.join(DRIVE_CKPT, "best_v5.0.pth")
            assert os.path.exists(V50_CKPT), f"V5.0 checkpoint not found: {V50_CKPT}"

            for f in glob.glob(os.path.join(REPO_DIR, "checkpoints", "*.pth")):
                os.remove(f)
            print("Cleaned local checkpoints for V5.3 fresh start")


            def sync_checkpoints_to_drive(tag):
                local_ckpt = os.path.join(REPO_DIR, "checkpoints")
                if not os.path.exists(local_ckpt):
                    return
                os.makedirs(DRIVE_CKPT, exist_ok=True)
                for f in glob.glob(os.path.join(local_ckpt, "*.pth")):
                    dst = os.path.join(DRIVE_CKPT, os.path.basename(f))
                    shutil.copy2(f, dst)
                local_best = os.path.join(local_ckpt, "best.pth")
                if os.path.exists(local_best):
                    tagged = os.path.join(DRIVE_CKPT, f"best_{tag}.pth")
                    shutil.copy2(local_best, tagged)
                    print(f"Best checkpoint saved as {tagged}")
                print(f"Synced checkpoints to {DRIVE_CKPT}")


            env = dict(os.environ, DRIVE_CKPT_DIR=DRIVE_CKPT)
            ret = subprocess.run(
                [
                    "python", "-u", "train.py",
                    "--config", "configs/a100_v5.3.yaml",
                    "--resume", V50_CKPT,
                    "--reset-lr",
                    "--reset-optimizer",
                    "--es-metric", "et",
                ],
                cwd=REPO_DIR,
                env=env,
            )
            if ret.returncode != 0:
                raise RuntimeError(f"V5.3 training failed with exit code {ret.returncode}")

            sync_checkpoints_to_drive("v5.3")

            local_logs = os.path.join(REPO_DIR, "logs")
            if os.path.isdir(local_logs):
                drive_logs = os.path.join(DRIVE_BASE, "logs_v5.3")
                if os.path.isdir(drive_logs):
                    shutil.rmtree(drive_logs)
                shutil.copytree(local_logs, drive_logs)
                print(f"Archived V5.3 logs to {drive_logs}")

            print("V5.3 training complete!")
            """
        ).strip()
    )


def cell_10_eval_header() -> nbformat.NotebookNode:
    return new_markdown_cell(
        dedent(
            """
            ## V5.3 Evaluation

            Use advanced BraTS post-processing for validation sweep and final test evaluation.
            The sweep optimizes `et_min_size` and `et_wt_ratio` while keeping the rest of the
            advanced pipeline fixed.
            """
        ).strip()
    )


def cell_11_advanced_pp_sweep() -> nbformat.NotebookNode:
    return new_code_cell(
        dedent(
            """
            # Cell 11: V5.3 Advanced PP Sweep
            import os
            import re
            import subprocess

            os.chdir(REPO_DIR)

            DRIVE_CKPT = os.path.join(DRIVE_BASE, "checkpoints")
            ckpt = os.path.join(DRIVE_CKPT, "best_v5.3.pth")
            if not os.path.exists(ckpt):
                ckpt = os.path.join(REPO_DIR, "checkpoints", "best.pth")
            assert os.path.exists(ckpt), f"No V5.3 checkpoint found at {ckpt}"

            CONFIG = "configs/a100_v5.3.yaml"
            ET_MIN_SIZES = [50, 100, 200]
            ET_WT_RATIOS = [0.0, 0.03, 0.04]
            WT_MIN_SIZE = 500
            TC_MIN_SIZE = 200
            ET_CONF_THRESH = 0.0


            def run_eval(split, use_text=True, tta=True, advanced_pp=True, **pp_kwargs):
                flags = ["--use-text"] if use_text else ["--no-text"]
                if tta:
                    flags.append("--tta")
                if advanced_pp:
                    flags.extend(
                        [
                            "--advanced-pp",
                            "--wt-min-size", str(pp_kwargs["wt_min_size"]),
                            "--tc-min-size", str(pp_kwargs["tc_min_size"]),
                            "--et-min-size", str(pp_kwargs["et_min_size"]),
                            "--et-conf-thresh", str(pp_kwargs["et_conf_thresh"]),
                            "--et-wt-ratio", str(pp_kwargs["et_wt_ratio"]),
                        ]
                    )
                cmd = [
                    "python", "-u", "evaluate_full.py",
                    "--config", CONFIG,
                    "--checkpoint", ckpt,
                    "--split", split,
                    "--overlap", "0.5",
                ] + flags
                ret = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True)
                if ret.returncode != 0:
                    print(f"FAILED: {ret.stderr[-400:]}")
                    return None
                metrics = {}
                for line in ret.stdout.splitlines():
                    for key in ["dice_ET", "dice_TC", "dice_WT", "dice_mean"]:
                        match = re.search(key + r": ([\\d.]+) \\+/- ([\\d.]+)", line)
                        if match:
                            metrics[key] = float(match.group(1))
                return metrics


            print("=" * 78)
            print("ADVANCED PP SWEEP: val split, text + TTA + advanced PP")
            print("=" * 78)
            sweep_results = []
            for et_min_size in ET_MIN_SIZES:
                for et_wt_ratio in ET_WT_RATIOS:
                    print(
                        f"\\n--- et_min_size={et_min_size}, et_wt_ratio={et_wt_ratio:.2f} ---"
                    )
                    metrics = run_eval(
                        "val",
                        et_min_size=et_min_size,
                        wt_min_size=WT_MIN_SIZE,
                        tc_min_size=TC_MIN_SIZE,
                        et_conf_thresh=ET_CONF_THRESH,
                        et_wt_ratio=et_wt_ratio,
                    )
                    if metrics:
                        result = {
                            "et_min_size": et_min_size,
                            "et_wt_ratio": et_wt_ratio,
                            **metrics,
                        }
                        sweep_results.append(result)
                        print(
                            f"  ET={metrics['dice_ET']:.4f}  "
                            f"TC={metrics['dice_TC']:.4f}  "
                            f"WT={metrics['dice_WT']:.4f}  "
                            f"Mean={metrics['dice_mean']:.4f}"
                        )

            assert sweep_results, "Advanced PP sweep produced no valid results"
            sweep_results.sort(key=lambda r: (r["dice_ET"], r["dice_mean"]), reverse=True)

            print("\\n" + "=" * 78)
            print("RANKING (sorted by ET Dice, then mean Dice)")
            print("=" * 78)
            print(f"{'Config':<32} {'ET':>8} {'TC':>8} {'WT':>8} {'Mean':>8}")
            print("-" * 72)
            for row in sweep_results:
                tag = f"et={row['et_min_size']}, ratio={row['et_wt_ratio']:.2f}"
                print(
                    f"{tag:<32} "
                    f"{row['dice_ET']:>7.4f} "
                    f"{row['dice_TC']:>7.4f} "
                    f"{row['dice_WT']:>7.4f} "
                    f"{row['dice_mean']:>7.4f}"
                )

            best_pp = {
                "wt_min_size": WT_MIN_SIZE,
                "tc_min_size": TC_MIN_SIZE,
                "et_min_size": sweep_results[0]["et_min_size"],
                "et_conf_thresh": ET_CONF_THRESH,
                "et_wt_ratio": sweep_results[0]["et_wt_ratio"],
            }
            print(f"\\nBest advanced PP params: {best_pp}")
            """
        ).strip()
    )


def cell_12_test_eval() -> nbformat.NotebookNode:
    return new_code_cell(
        dedent(
            """
            # Cell 12: V5.3 8-Config Test Eval
            import json
            import os
            import re
            import subprocess

            os.chdir(REPO_DIR)

            DRIVE_CKPT = os.path.join(DRIVE_BASE, "checkpoints")
            ckpt = os.path.join(DRIVE_CKPT, "best_v5.3.pth")
            if not os.path.exists(ckpt):
                ckpt = os.path.join(REPO_DIR, "checkpoints", "best.pth")
            assert os.path.exists(ckpt), f"No V5.3 checkpoint found at {ckpt}"

            CONFIG = "configs/a100_v5.3.yaml"
            if "best_pp" not in globals():
                best_pp = {
                    "wt_min_size": 500,
                    "tc_min_size": 200,
                    "et_min_size": 100,
                    "et_conf_thresh": 0.0,
                    "et_wt_ratio": 0.03,
                }
                print(f"best_pp not found from Cell 11, using fallback: {best_pp}")

            eval_configs = [
                ("text",          True,  False, False),
                ("text+PP",       True,  False, True),
                ("text+TTA",      True,  True,  False),
                ("text+TTA+PP",   True,  True,  True),
                ("notext",        False, False, False),
                ("notext+PP",     False, False, True),
                ("notext+TTA",    False, True,  False),
                ("notext+TTA+PP", False, True,  True),
            ]


            def run_eval(split, use_text, tta, use_pp):
                cmd = [
                    "python", "-u", "evaluate_full.py",
                    "--config", CONFIG,
                    "--checkpoint", ckpt,
                    "--split", split,
                    "--overlap", "0.5",
                ]
                cmd.append("--use-text" if use_text else "--no-text")
                if tta:
                    cmd.append("--tta")
                if use_pp:
                    cmd.extend(
                        [
                            "--advanced-pp",
                            "--wt-min-size", str(best_pp["wt_min_size"]),
                            "--tc-min-size", str(best_pp["tc_min_size"]),
                            "--et-min-size", str(best_pp["et_min_size"]),
                            "--et-conf-thresh", str(best_pp["et_conf_thresh"]),
                            "--et-wt-ratio", str(best_pp["et_wt_ratio"]),
                        ]
                    )
                ret = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True)
                if ret.returncode != 0:
                    print(f"WARNING: eval failed: {ret.stderr[-400:]}")
                    return None
                metrics = {}
                for line in ret.stdout.splitlines():
                    for key in ["dice_ET", "dice_TC", "dice_WT", "dice_mean"]:
                        match = re.search(key + r": ([\\d.]+) \\+/- ([\\d.]+)", line)
                        if match:
                            metrics[key] = float(match.group(1))
                return metrics


            print("=" * 78)
            print(f"V5.3 TEST EVAL WITH BEST ADVANCED PP: {best_pp}")
            print("=" * 78)

            test_results = {}
            for name, use_text, tta, use_pp in eval_configs:
                print(f"--- {name} ---")
                metrics = run_eval("test", use_text=use_text, tta=tta, use_pp=use_pp)
                if metrics:
                    test_results[name] = metrics
                    print(
                        f"  ET={metrics['dice_ET']:.4f}  "
                        f"TC={metrics['dice_TC']:.4f}  "
                        f"WT={metrics['dice_WT']:.4f}  "
                        f"Mean={metrics['dice_mean']:.4f}"
                    )

            DRIVE_EVAL = os.path.join(DRIVE_BASE, "eval_results")
            os.makedirs(DRIVE_EVAL, exist_ok=True)
            payload = {
                "best_pp": best_pp,
                "test_results": test_results,
            }
            out_path = os.path.join(DRIVE_EVAL, "v5.3_eval.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved V5.3 eval results to {out_path}")
            """
        ).strip()
    )


def cell_13_results() -> nbformat.NotebookNode:
    return new_code_cell(
        dedent(
            """
            # Cell 13: Results comparison table + V5.0 baseline delta
            assert "test_results" in globals(), "Run Cell 12 first to populate test_results"

            v50_baseline = {
                "text+TTA+PP": {"ET": 0.7910, "TC": 0.8560, "WT": 0.8967},
                "text+TTA":    {"ET": 0.7897, "TC": 0.8560, "WT": 0.8967},
                "text":        {"ET": 0.7760, "TC": 0.8402, "WT": 0.8866},
            }

            print("=" * 78)
            print("RESULTS: V5.3 vs V5.0 baseline")
            print("=" * 78)
            print(f"{'Config':<16} {'Version':<8} {'ET':>8} {'TC':>8} {'WT':>8} {'Mean':>8}")
            print("-" * 72)

            for cfg_name in [
                "text+TTA+PP",
                "text+TTA",
                "text",
                "notext+TTA+PP",
                "notext+TTA",
                "notext",
            ]:
                if cfg_name in test_results:
                    current = test_results[cfg_name]
                    current_mean = (
                        current["dice_ET"] + current["dice_TC"] + current["dice_WT"]
                    ) / 3
                    print(
                        f"{cfg_name:<16} {'V5.3':<8} "
                        f"{current['dice_ET']:>7.2%} {current['dice_TC']:>7.2%} "
                        f"{current['dice_WT']:>7.2%} {current_mean:>7.2%}"
                    )
                if cfg_name in v50_baseline:
                    base = v50_baseline[cfg_name]
                    base_mean = (base["ET"] + base["TC"] + base["WT"]) / 3
                    print(
                        f"{cfg_name:<16} {'V5.0':<8} "
                        f"{base['ET']:>7.2%} {base['TC']:>7.2%} "
                        f"{base['WT']:>7.2%} {base_mean:>7.2%}"
                    )
                print()

            if "text+TTA+PP" in test_results:
                current = test_results["text+TTA+PP"]
                base = v50_baseline["text+TTA+PP"]
                print("Delta vs V5.0 baseline (text+TTA+PP):")
                print(f"  ET: {base['ET']:.4f} -> {current['dice_ET']:.4f} ({current['dice_ET'] - base['ET']:+.4f})")
                print(f"  TC: {base['TC']:.4f} -> {current['dice_TC']:.4f} ({current['dice_TC'] - base['TC']:+.4f})")
                print(f"  WT: {base['WT']:.4f} -> {current['dice_WT']:.4f} ({current['dice_WT'] - base['WT']:+.4f})")

            best_name, best_metrics = max(
                test_results.items(),
                key=lambda item: (item[1]["dice_ET"], item[1]["dice_mean"]),
            )
            print()
            print(
                f"Best V5.3 config by ET Dice: {best_name} "
                f"(ET={best_metrics['dice_ET']:.4f}, "
                f"TC={best_metrics['dice_TC']:.4f}, "
                f"WT={best_metrics['dice_WT']:.4f}, "
                f"Mean={best_metrics['dice_mean']:.4f})"
            )
            print(f"Advanced PP params used: {best_pp}")
            """
        ).strip()
    )


def cell_14_resume_header() -> nbformat.NotebookNode:
    return new_markdown_cell(
        dedent(
            """
            ## Resume Training

            Run Cells 1-3 first to reinstall dependencies and restore data, then run this cell.
            """
        ).strip()
    )


def cell_15_resume_training() -> nbformat.NotebookNode:
    return new_code_cell(
        dedent(
            """
            # Cell 15: Resume training after disconnect
            import glob
            import os
            import shutil
            import subprocess

            REPO_DIR = '/content/TextMamba3D'
            DRIVE_BASE = '/content/drive/MyDrive/TextMamba3D'
            DRIVE_CKPT = os.path.join(DRIVE_BASE, "checkpoints")
            os.chdir(REPO_DIR)

            RESUME_CONFIG = "configs/a100_v5.3.yaml"
            RESUME_TAG = "v5.3"
            resume_ckpt = os.path.join(DRIVE_CKPT, "last.pth")

            if os.path.exists(resume_ckpt):
                print(f"Resuming {RESUME_TAG} from {resume_ckpt}")
                env = dict(os.environ, DRIVE_CKPT_DIR=DRIVE_CKPT)
                ret = subprocess.run(
                    [
                        "python", "-u", "train.py",
                        "--config", RESUME_CONFIG,
                        "--resume", resume_ckpt,
                        "--es-metric", "et",
                    ],
                    cwd=REPO_DIR,
                    env=env,
                )
                if ret.returncode != 0:
                    raise RuntimeError(f"Resume failed with exit code {ret.returncode}")

                local_ckpt = os.path.join(REPO_DIR, "checkpoints")
                if os.path.exists(local_ckpt):
                    for f in glob.glob(os.path.join(local_ckpt, "*.pth")):
                        dst = os.path.join(DRIVE_CKPT, os.path.basename(f))
                        shutil.copy2(f, dst)
                    local_best = os.path.join(local_ckpt, "best.pth")
                    if os.path.exists(local_best):
                        tagged = os.path.join(DRIVE_CKPT, f"best_{RESUME_TAG}.pth")
                        shutil.copy2(local_best, tagged)
                        print(f"Tagged: {tagged}")
                    print(f"Synced to {DRIVE_CKPT}")
            else:
                print(f"No checkpoint to resume from at {resume_ckpt}")
            """
        ).strip()
    )


def build_notebook() -> nbformat.NotebookNode:
    return new_notebook(
        cells=[
            cell_0_title(),
            cell_1_mount_and_install(),
            cell_2_clone_and_extract(),
            cell_3_restore_et_text(),
            cell_4_smoke_tests_header(),
            cell_5_copy_paste_smoke_test(),
            cell_6_quantitative_text_smoke_test(),
            cell_7_advanced_pp_and_dry_run(),
            cell_8_training_header(),
            cell_9_training(),
            cell_10_eval_header(),
            cell_11_advanced_pp_sweep(),
            cell_12_test_eval(),
            cell_13_results(),
            cell_14_resume_header(),
            cell_15_resume_training(),
        ],
        metadata={
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "gpuClass": "standard",
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
        },
    )


def main() -> None:
    notebook = build_notebook()
    if len(notebook.cells) != 16:
        raise ValueError(f"Expected 16 cells, got {len(notebook.cells)}")
    OUTPUT_PATH.write_text(nbformat.writes(notebook), encoding="utf-8")
    print(f"Wrote notebook to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
