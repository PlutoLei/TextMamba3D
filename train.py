# train.py
"""Training script for TextMamba3D model."""

import argparse
import math
import os
import shutil
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transformers import AutoTokenizer

from data import BraTSDataset, get_train_transforms, get_val_transforms
from data.brats_textbrats_dataset import TextBraTSDataset
from losses import CombinedLoss
from models import TextMamba3D
from models.text_encoder import TextMambaEncoder
from utils.metrics import dice_score_brats_regions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train TextMamba3D')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--no-text-ratio', type=float, default=None,
                        help='Ratio of samples trained without text (overrides config)')
    parser.add_argument('--grad-accum', type=int, default=None,
                        help='Gradient accumulation steps (overrides config)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit training samples (e.g., 200 for quick training)')
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='Override max epochs (e.g., 1 for smoke test)')
    return parser.parse_args()


class EarlyStopping:
    """Early stopping when validation metric stops improving."""

    def __init__(self, patience: int = 20, min_delta: float = 0.001, mode: str = 'max') -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score > self.best_score + self.min_delta
            if self.mode == 'max'
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience

        return self.early_stop


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_lr(epoch: int, warmup_epochs: int, base_lr: float, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr * 0.5 * (
        1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))
    )


from utils.precision import get_amp_context


def train_epoch(
    model, loader, criterion, optimizer, device, epoch,
    scaler=None, use_amp=False, no_text_ratio=0.1, grad_accum=1,
    need_features=True, clip_norm=1.0, bf16_mode=None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    nan_count = 0
    max_grad_norm = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        image = batch['image'].to(device)
        if bf16_mode == 'pure':
            image = image.to(dtype=torch.bfloat16)
        mask = batch['mask'].to(device)
        text_ids = batch['text_ids'].to(device)
        attn_mask = batch.get('attention_mask')
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        use_text = np.random.random() > no_text_ratio

        with get_amp_context(bf16_mode, use_amp):
            if need_features and use_text:
                pred, img_feat, text_feat, pixel_feat = model(
                    image,
                    text_ids,
                    attention_mask=attn_mask,
                    return_features=True,
                    use_text=True,
                )
            else:
                pred = model(
                    image,
                    text_ids if use_text else None,
                    attention_mask=attn_mask if use_text else None,
                    return_features=False,
                    use_text=use_text,
                )
                img_feat, text_feat, pixel_feat = None, None, None
            # Deep supervision: aux outputs stored in decoder during training
            aux_preds = getattr(model.decoder, '_aux_outputs', None) or None
            loss = criterion(
                pred,
                mask,
                img_feat,
                text_feat,
                pixel_feat=pixel_feat,
                mask_orig=mask,
                aux_preds=aux_preds,
            )['total']
            loss = loss / grad_accum  # 梯度累积：损失除以累积步数

        # Detect anomalously high but finite loss (indicates impending NaN)
        # CombinedLoss clamps at 20.0; skip truly anomalous batches above 10.0
        if loss.isfinite() and loss.item() * grad_accum > 10.0:
            nan_count += 1
            if nan_count <= 5:
                print(f'[HIGH LOSS] epoch={epoch}, batch={batch_idx}, text={use_text}, '
                      f'loss={loss.item() * grad_accum:.4f}')
            optimizer.zero_grad()
            pbar.set_postfix({'loss': 'HIGH', 'text': 'Y' if use_text else 'N'})
            continue

        if torch.isnan(loss).any().item() or torch.isinf(loss).any().item():
            nan_count += 1
            # Diagnostic: check if NaN is from model output (first 5 per epoch)
            if nan_count <= 5:
                pred_nan = torch.isnan(pred).sum().item()
                pred_inf = torch.isinf(pred).sum().item()
                pred_max = pred.float().abs().max().item()
                print(f'[NaN/Inf] epoch={epoch}, batch={batch_idx}, text={use_text}, '
                      f'pred_nan={pred_nan}, pred_inf={pred_inf}, pred_absmax={pred_max:.2e}')
            optimizer.zero_grad()
            pbar.set_postfix({'loss': 'NaN/Inf', 'text': 'Y' if use_text else 'N'})
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 每 grad_accum 步更新一次参数
        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                optimizer.step()
            grad_norm_value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            max_grad_norm = max(max_grad_norm, grad_norm_value)
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        pbar.set_postfix({'loss': f'{loss.item() * grad_accum:.4f}', 'text': 'Y' if use_text else 'N'})

    print(f'Epoch {epoch}: NaN/Inf skipped batches={nan_count}')
    return total_loss / len(loader), max_grad_norm


@torch.no_grad()
def validate(
    model, loader, criterion, device, use_amp=True, use_text=True, bf16_mode=None
) -> tuple[float, float]:
    """Validate model with or without text guidance."""
    model.eval()
    total_loss = 0.0
    all_dice = []

    desc = 'Validating' if use_text else 'Validating (no-text)'
    for batch in tqdm(loader, desc=desc):
        image = batch['image'].to(device)
        if bf16_mode == 'pure':
            image = image.to(dtype=torch.bfloat16)
        mask = batch['mask'].to(device)
        text_ids = batch['text_ids'].to(device) if use_text else None
        attn_mask = batch.get('attention_mask')
        if use_text and attn_mask is not None:
            attn_mask = attn_mask.to(device)
        else:
            attn_mask = None

        with get_amp_context(bf16_mode, use_amp):
            pred = model(
                image, text_ids, attention_mask=attn_mask,
                return_features=False, use_text=use_text
            )
            loss = criterion(pred, mask)['total']

        total_loss += loss.item()
        all_dice.append(dice_score_brats_regions(pred, mask)['dice_mean'])

    return total_loss / len(loader), np.mean(all_dice)


@torch.no_grad()
def log_gate_values(model, writer, epoch):
    """Log TextScaleGate sigmoid values to TensorBoard."""
    text_gate = getattr(model, 'text_gate', None)
    if text_gate is None:
        return
    gate_values = {}
    for i, gate in enumerate(text_gate.gates):
        bias = gate.gate_proj.bias.item()
        gate_val = torch.sigmoid(torch.tensor(bias)).item()
        scale_name = f'scale_{i+1}'
        gate_values[scale_name] = gate_val
        writer.add_scalar(f'Gate/{scale_name}_bias', bias, epoch)
        writer.add_scalar(f'Gate/{scale_name}_sigmoid', gate_val, epoch)
    vals = [f'{k}={v:.4f}' for k, v in gate_values.items()]
    print(f'  Gate values: {", ".join(vals)}')
    return gate_values


def main():
    args = parse_args()
    config = load_config(args.config)
    contrastive_warmup = config['training'].get('contrastive_warmup_epochs', 30)
    base_contrastive_weight = config['loss'].get('contrastive_weight', 0.0)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = config['training'].get('use_amp', False)
    if args.no_amp:
        use_amp = False
    use_amp = use_amp and device.type == 'cuda'
    bf16_mode = config['training'].get('bf16_mode', None)
    if bf16_mode == 'pure':
        use_amp = False
    # Resolve CLI overrides vs config defaults
    no_text_ratio = args.no_text_ratio if args.no_text_ratio is not None else config['training'].get('no_text_ratio', 0.1)
    grad_accum = args.grad_accum if args.grad_accum is not None else config['training'].get('gradient_accumulation', 1)

    print(f'Using device: {device}')
    print(f'Mixed precision (AMP): {use_amp}')
    print(f'No-text training ratio: {no_text_ratio}')
    print(f'Gradient accumulation: {grad_accum}')

    # Data
    use_elastic = config.get('augmentation', {}).get('use_elastic', False)
    use_modality_dropout = config.get('augmentation', {}).get('use_modality_dropout', False)
    train_transform = get_train_transforms(
        tuple(config['data']['patch_size']),
        use_elastic=use_elastic,
        use_modality_dropout=use_modality_dropout,
    )
    val_transform = get_val_transforms(tuple(config['data']['patch_size']))

    # Initialize PubMedBERT tokenizer
    use_pretrained_text = config['model'].get('use_pretrained_text', True)
    text_model_path = config['model'].get('text_model_path', None)
    tokenizer = None
    if use_pretrained_text:
        tokenizer_path = text_model_path or TextMambaEncoder.PUBMEDBERT_NAME
        print(f"Loading PubMedBERT tokenizer: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 根据配置选择数据集类型
    dataset_type = config['data'].get('dataset_type', 'brats2021')
    max_text_len = config['model'].get('text_max_len', 128)

    if dataset_type == 'textbrats':
        # TextBraTS 数据集 (专家标注文本, 无信息泄露)
        print("Using TextBraTS dataset (expert-annotated text)")
        train_ratio = config['data'].get('train_ratio', 0.8)
        val_ratio = config['data'].get('val_ratio', 0.0)
        et_enriched = config['data'].get('et_enriched', False)
        enriched_prob = config['data'].get('enriched_prob', 0.5)

        train_dataset = TextBraTSDataset(
            data_dir=config['data']['data_dir'],
            split='train',
            transform=train_transform,
            tokenizer=tokenizer,
            max_text_len=max_text_len,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            et_enriched=et_enriched,
            enriched_prob=enriched_prob,
        )
        val_dataset = TextBraTSDataset(
            data_dir=config['data']['data_dir'],
            split='val',
            transform=val_transform,
            tokenizer=tokenizer,
            max_text_len=max_text_len,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            et_enriched=et_enriched,
            enriched_prob=enriched_prob,
        )
    else:
        # 原始 BraTS2021 数据集 (自动生成文本)
        print("Using BraTS2021 dataset (auto-generated text)")
        train_dataset = BraTSDataset(
            data_dir=config['data']['data_dir'],
            split='train',
            transform=train_transform,
            tokenizer=tokenizer,
            max_text_len=max_text_len,
        )
        val_dataset = BraTSDataset(
            data_dir=config['data']['data_dir'],
            split='val',
            transform=val_transform,
            tokenizer=tokenizer,
            max_text_len=max_text_len,
        )

    # 限制训练样本数量（用于快速测试或显存不足）
    if args.max_samples and args.max_samples < len(train_dataset):
        from torch.utils.data import Subset
        indices = list(range(args.max_samples))
        train_dataset = Subset(train_dataset, indices)
        print(f'Limited training to {args.max_samples} samples')

    if len(train_dataset) == 0:
        raise RuntimeError(
            f"No training samples found in {config['data']['data_dir']}. "
            "Check that data has been extracted to the correct path."
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

    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

    # Model
    use_checkpoint = config['training'].get('gradient_checkpointing', False)
    if use_checkpoint:
        print('Gradient checkpointing: ENABLED (saves ~30-50% GPU memory)')

    deep_supervision = config['training'].get('deep_supervision', False)
    if deep_supervision:
        print('Deep supervision: ENABLED (aux heads at intermediate decoder stages)')

    use_mamba3 = config['model'].get('use_mamba3', False)
    rope_fraction = config['model'].get('rope_fraction', None)
    chunk_size = config['model'].get('chunk_size', None)
    is_mimo = config['model'].get('is_mimo', False)
    model = TextMamba3D(
        img_size=tuple(config['model']['img_size']),
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        d_state=config['model'].get('d_state', 16),
        text_embed_dim=config['model']['text_embed_dim'],
        text_max_len=max_text_len,
        use_pretrained_text=use_pretrained_text,
        unfreeze_text_layers=config['model'].get('unfreeze_text_layers', 0),
        use_checkpoint=use_checkpoint,
        text_model_path=text_model_path,
        deep_supervision=deep_supervision,
        dropout=config['model'].get('dropout', 0.0),
        use_text_gate=config['model'].get('use_text_gate', False),
        use_cross_scale_skip=config['model'].get('use_cross_scale_skip', False),
        text_gate_init_bias=config['model'].get('text_gate_init_bias', 2.0),
        use_mamba3=use_mamba3,
        headdim=config['model'].get('headdim', None),
        rope_fraction=rope_fraction,
        chunk_size=chunk_size,
        is_mimo=is_mimo,
        fusion_type=config['model'].get('fusion_type', 'seqca'),
    ).to(device)

    if bf16_mode == 'pure':
        model.to_bf16_with_fp32_text()
        print('Pure bf16 mode: model bf16, BERT fp32')
    elif use_amp and use_mamba3:
        model.prepare_for_amp()
        print('AMP + Mamba3: SSM modules bf16, rest fp32 (autocast handles casting)')

    if use_mamba3:
        print(f'SSM Backend: Mamba-3 (d_state={config["model"].get("d_state", 16)}, '
              f'headdim={config["model"].get("headdim", "auto")})')
    else:
        print('SSM Backend: Mamba-1 (default)')

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {num_params:,}')

    # Loss (with optional class weighting for BraTS imbalance)
    from losses.dice_loss import BRATS_CLASS_WEIGHTS
    class_weights = config['loss'].get('class_weights', BRATS_CLASS_WEIGHTS)
    print(f'Class weights: {class_weights}')

    criterion = CombinedLoss(
        dice_weight=config['loss']['dice_weight'],
        ce_weight=config['loss']['ce_weight'],
        edge_weight=config['loss']['edge_weight'],
        contrastive_weight=config['loss']['contrastive_weight'],
        temperature=config['loss']['temperature'],
        feat_dim=config['model']['embed_dim'] * (2 ** (len(config['model']['depths']) - 1)),
        text_dim=config['model']['text_embed_dim'],
        class_weights=class_weights,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )

    # Manual LR schedule config (warmup + cosine decay)
    warmup_epochs = config['training'].get('warmup_epochs', 5)
    total_epochs = args.max_epochs if args.max_epochs is not None else config['training']['epochs']
    base_lr = config['training']['lr']
    cfg_clip = config['training'].get('gradient_clip_norm', 1.0)

    # AMP scaler — bfloat16 does NOT need GradScaler (same exponent range as fp32)
    # GradScaler is only needed for float16 to prevent gradient underflow
    scaler = None

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training'].get('patience', 30),
        min_delta=0.001,
        mode='max'
    )

    # Tensorboard
    writer = SummaryWriter('logs')

    # Resume
    start_epoch = 0
    best_dice = 0
    best_dice_no_text = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scaler is not None and checkpoint.get('scaler') is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0)
        best_dice_no_text = checkpoint.get('best_dice_no_text', 0)
        print(f'Resumed from epoch {start_epoch}')

    # Training loop
    for epoch in range(start_epoch, total_epochs):
        if epoch < contrastive_warmup:
            criterion.contrastive_weight = 0.0
        elif epoch < contrastive_warmup * 2:
            criterion.contrastive_weight = (
                base_contrastive_weight * (epoch - contrastive_warmup) / contrastive_warmup
            )
        else:
            criterion.contrastive_weight = base_contrastive_weight
        need_features = criterion.contrastive_weight > 0

        current_lr = get_lr(epoch, warmup_epochs, base_lr, total_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Train
        train_loss, max_grad_norm = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scaler=scaler, use_amp=use_amp, no_text_ratio=no_text_ratio,
            grad_accum=grad_accum, need_features=need_features, clip_norm=cfg_clip,
            bf16_mode=bf16_mode,
        )

        # Validate with text (standard mode)
        val_loss, val_dice = validate(
            model, val_loader, criterion, device, use_amp=use_amp,
            use_text=True, bf16_mode=bf16_mode
        )

        # Validate without text (fair evaluation mode)
        val_loss_no_text, val_dice_no_text = validate(
            model, val_loader, criterion, device, use_amp=use_amp,
            use_text=False, bf16_mode=bf16_mode
        )

        # Log
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/val_no_text', val_loss_no_text, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('Dice/val_no_text', val_dice_no_text, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        writer.add_scalar('GradNorm/max', max_grad_norm, epoch)
        writer.add_scalar('Loss/contrastive_weight', criterion.contrastive_weight, epoch)
        # Log TextScaleGate values (V4.6)
        log_gate_values(model, writer, epoch)

        print(f'Epoch {epoch}: train_loss={train_loss:.4f}')
        print(f'  With text:    val_loss={val_loss:.4f}, val_dice={val_dice:.4f}')
        print(f'  Without text: val_loss={val_loss_no_text:.4f}, val_dice={val_dice_no_text:.4f}')
        print(f'  Contrastive weight: {criterion.contrastive_weight:.4f}')

        os.makedirs('checkpoints', exist_ok=True)

        def save_checkpoint(path: str) -> None:
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_dice': best_dice,
                'best_dice_no_text': best_dice_no_text,
                'scaler': scaler.state_dict() if scaler is not None else None,
            }
            torch.save(state, path)

        if val_dice_no_text > best_dice_no_text:
            best_dice_no_text = val_dice_no_text
            save_checkpoint('checkpoints/best_no_text.pth')
            print(f'  -> New best (no-text): {best_dice_no_text:.4f}')

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint('checkpoints/best.pth')
            print(f'  -> New best (with-text): {best_dice:.4f}')

        save_checkpoint('checkpoints/last.pth')

        # Colab Drive auto-backup every 10 epochs
        drive_ckpt = os.environ.get('DRIVE_CKPT_DIR')
        if drive_ckpt and (epoch + 1) % 10 == 0:
            try:
                os.makedirs(drive_ckpt, exist_ok=True)
                for fname in ['best.pth', 'best_no_text.pth', 'last.pth']:
                    src = f'checkpoints/{fname}'
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(drive_ckpt, fname))
                print(f'  -> Synced checkpoints to Drive (epoch {epoch})')
            except OSError as e:
                print(f'  [WARN] Drive sync failed: {e}')

        # Early stopping (based on best of both modes)
        if early_stopping(max(val_dice, val_dice_no_text)):
            print(f'\nEarly stopping triggered at epoch {epoch}')
            break

    print('\nTraining complete!')
    print(f'Best Dice (with text): {best_dice:.4f}')
    print(f'Best Dice (no text):   {best_dice_no_text:.4f}')
    writer.close()


if __name__ == '__main__':
    main()
