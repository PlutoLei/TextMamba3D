# train.py
import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import TextMamba3D
from losses import CombinedLoss
from data import BraTSDataset, get_train_transforms, get_val_transforms
from utils.metrics import dice_score


def parse_args():
    parser = argparse.ArgumentParser(description='Train TextMamba3D')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch in pbar:
        image = batch['image'].to(device)
        mask = batch['mask'].to(device)
        text_ids = batch['text_ids'].to(device)

        optimizer.zero_grad()

        # Forward
        pred, img_feat, text_feat = model(image, text_ids, return_features=True)

        # Loss
        losses = criterion(pred, mask, img_feat, text_feat)
        loss = losses['total']

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_dice = []

    for batch in tqdm(loader, desc='Validating'):
        image = batch['image'].to(device)
        mask = batch['mask'].to(device)
        text_ids = batch['text_ids'].to(device)

        pred, img_feat, text_feat = model(image, text_ids, return_features=True)
        losses = criterion(pred, mask, img_feat, text_feat)

        total_loss += losses['total'].item()

        # Dice score
        dice = dice_score(pred, mask, num_classes=4)
        all_dice.append(dice['dice_mean'])

    return total_loss / len(loader), np.mean(all_dice)


def main():
    args = parse_args()
    config = load_config(args.config)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data
    train_transform = get_train_transforms(tuple(config['data']['patch_size']))
    val_transform = get_val_transforms(tuple(config['data']['patch_size']))

    train_dataset = BraTSDataset(
        data_dir=config['data']['data_dir'],
        split='train',
        transform=train_transform,
    )
    val_dataset = BraTSDataset(
        data_dir=config['data']['data_dir'],
        split='val',
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )

    # Model
    model = TextMamba3D(
        img_size=tuple(config['model']['img_size']),
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        text_embed_dim=config['model']['text_embed_dim'],
    ).to(device)

    # Loss
    criterion = CombinedLoss(
        dice_weight=config['loss']['dice_weight'],
        ce_weight=config['loss']['ce_weight'],
        edge_weight=config['loss']['edge_weight'],
        contrastive_weight=config['loss']['contrastive_weight'],
        temperature=config['loss']['temperature'],
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
    )

    # Tensorboard
    writer = SummaryWriter('logs')

    # Resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

    # Training loop
    best_dice = 0
    for epoch in range(start_epoch, config['training']['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Log
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_dice={val_dice:.4f}')

        # Save checkpoint
        os.makedirs('checkpoints', exist_ok=True)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_dice': best_dice,
            }, 'checkpoints/best.pth')

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, 'checkpoints/last.pth')

    writer.close()


if __name__ == '__main__':
    main()
