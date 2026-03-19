"""Smoke test: verify full train pipeline (forward + loss + backward + step)."""

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from models import TextMamba3D
from losses import CombinedLoss


def smoke_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Model (smaller volume for 8GB VRAM smoke test)
    vol_size = 64  # 96 may OOM on 8GB; use 64 for smoke test
    model = TextMamba3D(
        img_size=(vol_size, vol_size, vol_size),
        in_channels=4,
        out_channels=4,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        text_embed_dim=256,
        text_max_len=128,
        use_pretrained_text=False,  # skip PubMedBERT download
        use_checkpoint=True,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {num_params:,}")
    print(f"Trainable params: {trainable:,}")

    # Loss
    criterion = CombinedLoss(
        dice_weight=1.0, ce_weight=1.0, edge_weight=1.0,
        contrastive_weight=0.0, temperature=0.07,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = GradScaler() if device.type == 'cuda' else None

    # Synthetic data (batch_size=1)
    img = torch.randn(1, 4, vol_size, vol_size, vol_size, device=device)
    mask = torch.randint(0, 4, (1, vol_size, vol_size, vol_size), device=device).long()
    text_ids = torch.randint(0, 1000, (1, 128), device=device)
    attn_mask = torch.ones(1, 128, device=device)

    # === Test 1: Forward + backward WITH text ===
    print("\n--- Test 1: Forward + backward (with text) ---")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    optimizer.zero_grad()
    with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
        pred, img_feat, text_feat, _pixel_feat = model(
            img, text_ids, attention_mask=attn_mask,
            return_features=True, use_text=True,
        )
        loss_dict = criterion(pred, mask, img_feat, text_feat)
        loss = loss_dict['total']

    output_shape = pred.shape
    loss_val = loss.item()
    print(f"  Output shape: {output_shape}")
    print(f"  Loss: {loss_val:.4f}")
    for k, v in loss_dict.items():
        if k != 'total':
            print(f"    {k}: {v.item():.4f}")

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Free forward graph before optimizer step to reduce peak VRAM
        del pred, img_feat, text_feat, loss_dict, loss
        torch.cuda.empty_cache()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        del pred, img_feat, text_feat, loss_dict, loss
        optimizer.step()

    if device.type == 'cuda':
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak VRAM: {peak_mb:.0f} MB ({peak_mb/1024:.1f} GB)")

    # Check gradients
    grad_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad and p.grad is not None
    )
    has_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    total_req = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  Gradients: {has_grad}/{total_req} params have grad, finite={grad_ok}")

    # === Test 2: Forward WITHOUT text ===
    print("\n--- Test 2: Forward (without text) ---")
    with torch.no_grad(), autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
        pred_no_text = model.forward_without_text(img)
    print(f"  Output shape: {pred_no_text.shape}")
    print(f"  Output finite: {torch.isfinite(pred_no_text).all().item()}")

    # === Test 3: Second step (verify no state corruption) ===
    print("\n--- Test 3: Second forward+backward step ---")
    optimizer.zero_grad()
    with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
        pred2, img_feat2, text_feat2, _pixel_feat2 = model(
            img, text_ids, attention_mask=attn_mask,
            return_features=True, use_text=True,
        )
        loss2 = criterion(pred2, mask, img_feat2, text_feat2)['total']

    loss2_val = loss2.item()
    if scaler is not None:
        scaler.scale(loss2).backward()
        del pred2, img_feat2, text_feat2, loss2
        torch.cuda.empty_cache()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss2.backward()
        del pred2, img_feat2, text_feat2, loss2
        optimizer.step()

    print(f"  Loss step 2: {loss2_val:.4f}")
    print(f"  Loss decreased: {loss2_val < loss_val}")

    print("\n=== SMOKE TEST PASSED ===")


if __name__ == '__main__':
    smoke_test()
