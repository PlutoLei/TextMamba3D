#!/usr/bin/env python3
"""Simple Masked Image Modeling (MIM) pretraining for TextMamba3D.

Masks 50% of input patches randomly, trains encoder+lightweight decoder
to reconstruct the masked patches. Backbone learns brain anatomy
but NOT tumor segmentation — preserving the "knowledge gap" for text.

Usage:
    python pretrain_ssl.py --data-dir /path/to/BraTS2021 --epochs 200 --batch-size 4

Reference: MambaMIM (MedIA 2025), MAE (CVPR 2022)
"""

import argparse
import os
import math
import shutil
import glob
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder_3d import MambaEncoder3D
from data.brats_dataset import BraTS2021Dataset


class MIMDecoder(nn.Module):
    """Lightweight decoder for masked image reconstruction."""

    def __init__(self, bottleneck_dim: int, out_channels: int = 4, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, out_channels * (patch_size ** 3)),
        )
        self.patch_size = patch_size
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] bottleneck features
        Returns:
            [B, C, D', H', W'] reconstructed patches
        """
        B, L, D = x.shape
        x = self.proj(x)  # [B, L, C*p^3]
        p = self.patch_size
        C = self.out_channels
        # L = (D/p) * (H/p) * (W/p)
        side = round(L ** (1/3))
        x = x.reshape(B, side, side, side, C, p, p, p)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)  # [B, C, side, p, side, p, side, p]
        x = x.reshape(B, C, side*p, side*p, side*p)
        return x


class PatchMasker(nn.Module):
    """Random patch masking for MIM pretraining."""

    def __init__(self, mask_ratio: float = 0.5):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, D, H, W] input volume
        Returns:
            masked_x: [B, C, D, H, W] with masked patches zeroed
            mask: [B, 1, D, H, W] binary mask (1=visible, 0=masked)
        """
        B, C, D, H, W = x.shape
        # Create patch-level mask
        mask = torch.ones(B, 1, D, H, W, device=x.device)
        # Mask at 8x8x8 block level for efficiency
        block = 8
        nD, nH, nW = D // block, H // block, W // block
        block_mask = torch.bernoulli(
            torch.ones(B, 1, nD, nH, nW, device=x.device) * (1 - self.mask_ratio)
        )
        mask = F.interpolate(block_mask, size=(D, H, W), mode='nearest')
        masked_x = x * mask
        return masked_x, mask


def get_lr(epoch, warmup, base_lr, total):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    return base_lr * 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / (total - warmup)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--mask-ratio', type=float, default=0.5)
    parser.add_argument('--embed-dim', type=int, default=48)
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Dataset (images only, no labels needed for SSL)
    dataset = BraTS2021Dataset(
        data_dir=args.data_dir,
        split='train',
        patch_size=(128, 128, 128),
        text_max_len=1,  # dummy, not used
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    print(f'Training samples: {len(dataset)}')

    # Val dataset
    val_dataset = BraTS2021Dataset(
        data_dir=args.data_dir,
        split='val',
        patch_size=(128, 128, 128),
        text_max_len=1,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model: encoder + MIM decoder
    encoder = MambaEncoder3D(
        img_size=(128, 128, 128),
        in_channels=4,
        embed_dim=args.embed_dim,
        depths=[2, 2, 2, 2],
        patch_size=(4, 4, 4),
        d_state=16,
        use_mamba3=True,
    ).to(device)

    bottleneck_dim = args.embed_dim * 8  # 384 for embed_dim=48
    mim_decoder = MIMDecoder(
        bottleneck_dim=bottleneck_dim,
        out_channels=4,
        patch_size=4,
    ).to(device)

    masker = PatchMasker(mask_ratio=args.mask_ratio)

    # Only need to reconstruct from bottleneck features
    params = list(encoder.parameters()) + list(mim_decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    os.makedirs(args.save_dir, exist_ok=True)

    # Resume
    start_epoch = 0
    best_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt['encoder'])
        mim_decoder.load_state_dict(ckpt['mim_decoder'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f'Resumed from epoch {start_epoch}')

    # Drive sync
    drive_ckpt = os.environ.get('DRIVE_CKPT_DIR')

    print(f'SSL Pretraining: {args.epochs} epochs, mask_ratio={args.mask_ratio}')
    print(f'Encoder params: {sum(p.numel() for p in encoder.parameters()):,}')

    for epoch in range(start_epoch, args.epochs):
        encoder.train()
        mim_decoder.train()

        lr = get_lr(epoch, args.warmup_epochs, args.lr, args.epochs)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        total_loss = 0
        pbar = tqdm(loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            image = batch['image'].to(device)

            # Mask input
            masked_img, mask = masker(image)

            with torch.amp.autocast('cuda'):
                features = encoder(masked_img)
                bottleneck = features[-1]  # deepest features
                recon = mim_decoder(bottleneck)

                # Reconstruction loss only on MASKED regions
                # Resize recon to match input if needed
                if recon.shape != image.shape:
                    recon = F.interpolate(recon, size=image.shape[2:], mode='trilinear', align_corners=False)

                inv_mask = 1 - mask  # 1 where masked
                loss = F.l1_loss(recon * inv_mask, image * inv_mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{lr:.2e}')

        avg_loss = total_loss / len(loader)

        # Validation
        encoder.eval()
        mim_decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].to(device)
                masked_img, mask = masker(image)
                with torch.amp.autocast('cuda'):
                    features = encoder(masked_img)
                    recon = mim_decoder(features[-1])
                    if recon.shape != image.shape:
                        recon = F.interpolate(recon, size=image.shape[2:], mode='trilinear', align_corners=False)
                    inv_mask = 1 - mask
                    loss = F.l1_loss(recon * inv_mask, image * inv_mask)
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        print(f'Epoch {epoch}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}, lr={lr:.2e}')

        # Save checkpoint
        ckpt = {
            'encoder': encoder.state_dict(),
            'mim_decoder': mim_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss,
        }
        torch.save(ckpt, os.path.join(args.save_dir, 'last_ssl.pth'))

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(ckpt, os.path.join(args.save_dir, 'best_ssl.pth'))
            print(f'  -> New best: {val_loss:.4f}')

        # Drive sync every 10 epochs
        if drive_ckpt and (epoch + 1) % 10 == 0:
            for name in ['last_ssl.pth', 'best_ssl.pth']:
                src = os.path.join(args.save_dir, name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(drive_ckpt, name))
            print(f'  -> Synced to Drive')

    # Final sync
    if drive_ckpt:
        for name in ['last_ssl.pth', 'best_ssl.pth']:
            src = os.path.join(args.save_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(drive_ckpt, name))
    print(f'SSL pretraining complete. Best val_loss: {best_loss:.4f}')


if __name__ == '__main__':
    main()
