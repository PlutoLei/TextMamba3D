# evaluate.py
"""Evaluation script for TextMamba3D model with TTA and AMP support."""

import argparse
import os

import numpy as np
import torch
import yaml
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import BraTSDataset, get_val_transforms
from data.brats_textbrats_dataset import TextBraTSDataset
from models import TextMamba3D
from models.text_encoder import TextMambaEncoder
from utils.metrics import (
    dice_score_brats_regions,
    hausdorff_distance_95_brats_regions,
    BRATS_REGIONS,
)
from utils.sliding_window import gaussian_weight_3d  # noqa: F401 (available for future use)
from utils.tta import tta_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate TextMamba3D')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--save_pred', action='store_true')
    parser.add_argument('--no-text', action='store_true',
                        help='Evaluate without text guidance')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--tta', action='store_true',
                        help='Enable test-time augmentation (flip along 3 axes)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable AMP inference')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model, loader, device, save_pred=False, save_dir='predictions',
    use_text=True, use_tta=False, use_amp=True,
) -> dict:
    """Evaluate model on dataset using BraTS standard regions (ET, TC, WT)."""
    model.eval()

    all_dice = {r: [] for r in BRATS_REGIONS}
    all_hd95 = {r: [] for r in BRATS_REGIONS}

    if save_pred:
        os.makedirs(save_dir, exist_ok=True)

    mode_parts = []
    if use_text:
        mode_parts.append('text-guided')
    else:
        mode_parts.append('no-text (fair mode)')
    if use_tta:
        mode_parts.append('TTA')
    if use_amp:
        mode_parts.append('AMP')
    print(f'\nEvaluation mode: {", ".join(mode_parts)}\n')

    for batch in tqdm(loader, desc='Evaluating'):
        image = batch['image'].to(device)
        mask = batch['mask'].to(device)
        case_name = batch['case_name'][0]

        text_ids = batch['text_ids'].to(device) if use_text else None
        attn_mask = batch.get('attention_mask')
        if use_text and attn_mask is not None:
            attn_mask = attn_mask.to(device)
        else:
            attn_mask = None

        with autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp and device.type == 'cuda'):
            if use_tta:
                probs = tta_predict(model, image, text_ids, attn_mask, use_text)
                pred_argmax = probs.argmax(dim=1)
                # Use logits-like tensor for dice_score_brats_regions
                pred_for_dice = probs
            else:
                pred = model(image, text_ids, attention_mask=attn_mask, use_text=use_text)
                pred_argmax = pred.argmax(dim=1)
                pred_for_dice = pred

        # BraTS region Dice scores
        dice = dice_score_brats_regions(pred_for_dice, mask)
        for r in BRATS_REGIONS:
            all_dice[r].append(dice[f'dice_{r}'])

        # BraTS region HD95
        pred_np = pred_argmax[0].cpu().numpy()
        mask_np = mask[0].cpu().numpy()
        hd95 = hausdorff_distance_95_brats_regions(pred_np, mask_np)
        for r in BRATS_REGIONS:
            val = hd95[f'hd95_{r}']
            if not np.isnan(val):
                all_hd95[r].append(val)

        if save_pred:
            np.save(os.path.join(save_dir, f'{case_name}_pred.npy'), pred_np)

    # Summary
    print('\n=== Evaluation Results (BraTS Regions) ===')

    print('\nDice Scores:')
    dice_means = {}
    for r in BRATS_REGIONS:
        scores = all_dice[r]
        mean_val = np.mean(scores) if scores else 0.0
        std_val = np.std(scores) if scores else 0.0
        dice_means[r] = mean_val
        print(f'  {r}: {mean_val:.4f} +/- {std_val:.4f}')
    mean_dice = np.mean(list(dice_means.values()))
    print(f'  Mean: {mean_dice:.4f}')

    print('\nHD95 (mm):')
    hd95_means = {}
    for r in BRATS_REGIONS:
        scores = all_hd95[r]
        if scores:
            mean_val = np.mean(scores)
            std_val = np.std(scores)
            hd95_means[r] = mean_val
            print(f'  {r}: {mean_val:.2f} +/- {std_val:.2f}')
        else:
            hd95_means[r] = np.nan
            print(f'  {r}: N/A')

    return {
        'dice_mean': mean_dice,
        'dice_ET': dice_means.get('ET', 0),
        'dice_TC': dice_means.get('TC', 0),
        'dice_WT': dice_means.get('WT', 0),
        'hd95_ET': hd95_means.get('ET', np.nan),
        'hd95_TC': hd95_means.get('TC', np.nan),
        'hd95_WT': hd95_means.get('WT', np.nan),
        'all_dice': all_dice,
        'all_hd95': all_hd95,
    }


def main() -> None:
    args = parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_text = not args.no_text
    use_amp = not args.no_amp and device.type == 'cuda'

    # Tokenizer
    use_pretrained_text = config['model'].get('use_pretrained_text', True)
    tokenizer = None
    if use_pretrained_text:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TextMambaEncoder.PUBMEDBERT_NAME)

    transform = get_val_transforms(tuple(config['data']['patch_size']))
    max_text_len = config['model'].get('text_max_len', 128)

    dataset_type = config['data'].get('dataset_type', 'brats2021')
    if dataset_type == 'textbrats':
        dataset = TextBraTSDataset(
            data_dir=config['data']['data_dir'],
            split=args.split,
            transform=transform,
            tokenizer=tokenizer,
            max_text_len=max_text_len,
            train_ratio=config['data'].get('train_ratio', 0.8),
        )
    else:
        dataset = BraTSDataset(
            data_dir=config['data']['data_dir'],
            split=args.split,
            transform=transform,
            tokenizer=tokenizer,
            max_text_len=max_text_len,
        )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f'Dataset: {len(dataset)} samples from {args.split} split')

    model = TextMamba3D(
        img_size=tuple(config['model']['img_size']),
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        text_embed_dim=config['model']['text_embed_dim'],
        text_max_len=max_text_len,
        use_pretrained_text=use_pretrained_text,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    print(f'Loaded checkpoint from {args.checkpoint}')

    evaluate(
        model, loader, device,
        save_pred=args.save_pred,
        use_text=use_text,
        use_tta=args.tta,
        use_amp=use_amp,
    )


if __name__ == '__main__':
    main()
