# evaluate.py
import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import TextMamba3D
from data import BraTSDataset, get_val_transforms
from utils.metrics import dice_score, hausdorff_distance_95


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TextMamba3D')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--save_pred', action='store_true')
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, save_pred=False, save_dir='predictions'):
    model.eval()

    all_dice = {f'class_{i}': [] for i in range(1, 4)}
    all_hd95 = {f'class_{i}': [] for i in range(1, 4)}

    if save_pred:
        os.makedirs(save_dir, exist_ok=True)

    for batch in tqdm(loader, desc='Evaluating'):
        image = batch['image'].to(device)
        mask = batch['mask'].to(device)
        text_ids = batch['text_ids'].to(device)
        case_name = batch['case_name'][0]

        pred = model(image, text_ids)
        pred_argmax = pred.argmax(dim=1)

        # Dice scores
        dice = dice_score(pred, mask, num_classes=4)
        for c in range(1, 4):
            all_dice[f'class_{c}'].append(dice[f'dice_class_{c}'])

        # HD95
        pred_np = pred_argmax[0].cpu().numpy()
        mask_np = mask[0].cpu().numpy()

        for c in range(1, 4):
            pred_c = (pred_np == c).astype(np.uint8)
            mask_c = (mask_np == c).astype(np.uint8)
            hd95 = hausdorff_distance_95(pred_c, mask_c)
            if not np.isnan(hd95):
                all_hd95[f'class_{c}'].append(hd95)

        # Save prediction
        if save_pred:
            np.save(os.path.join(save_dir, f'{case_name}_pred.npy'), pred_np)

    # Summary
    print('\n=== Evaluation Results ===')
    print('\nDice Scores:')
    for c in range(1, 4):
        scores = all_dice[f'class_{c}']
        print(f'  Class {c}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}')

    mean_dice = np.mean([np.mean(v) for v in all_dice.values()])
    print(f'  Mean: {mean_dice:.4f}')

    print('\nHD95:')
    for c in range(1, 4):
        scores = all_hd95[f'class_{c}']
        if scores:
            print(f'  Class {c}: {np.mean(scores):.2f} +/- {np.std(scores):.2f}')

    return {'dice': mean_dice}


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    transform = get_val_transforms(tuple(config['data']['patch_size']))
    dataset = BraTSDataset(
        data_dir=config['data']['data_dir'],
        split='test',
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Model
    model = TextMamba3D(
        img_size=tuple(config['model']['img_size']),
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        text_embed_dim=config['model']['text_embed_dim'],
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f'Loaded checkpoint from {args.checkpoint}')

    # Evaluate
    results = evaluate(model, loader, device, save_pred=args.save_pred)


if __name__ == '__main__':
    main()
