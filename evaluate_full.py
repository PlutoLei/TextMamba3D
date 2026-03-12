# evaluate_full.py
"""Full-volume evaluation with sliding window inference for TextMamba3D."""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.amp import autocast
from tqdm import tqdm
from transformers import AutoTokenizer

from data.brats_textbrats_dataset import TextBraTSDataset
from models import TextMamba3D
from models.text_encoder import TextMambaEncoder
from utils.metrics import dice_score_brats_regions, hausdorff_distance_95_brats_regions
from utils.sliding_window import gaussian_weight_3d


def parse_args():
    parser = argparse.ArgumentParser(description='Full-volume evaluation with sliding window')
    parser.add_argument('--config', type=str, default='configs/textbrats_a100.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--use-text', action='store_true', default=True)
    parser.add_argument('--no-text', dest='use_text', action='store_false')
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--tta', action='store_true', help='Enable test-time augmentation')
    parser.add_argument('--save-preds', type=str, default=None, help='Directory to save predictions')
    return parser.parse_args()


def sliding_window_inference(
    model,
    image: torch.Tensor,
    text_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    patch_size: tuple,
    overlap: float = 0.5,
    use_text: bool = True,
    use_amp: bool = True,
    sw_batch_size: int = 2,
) -> torch.Tensor:
    """Sliding window inference on a full-volume image.

    Args:
        model: TextMamba3D model
        image: [1, C, D, H, W] full volume
        text_ids: [1, L] text tokens
        attention_mask: [1, L] attention mask
        patch_size: (pD, pH, pW) patch dimensions
        overlap: fraction of overlap between adjacent patches
        use_text: whether to use text guidance
        use_amp: use bf16
        sw_batch_size: number of patches to process in parallel

    Returns:
        [1, num_classes, D, H, W] softmax probabilities
    """
    device = image.device
    B, C, D, H, W = image.shape
    pD, pH, pW = patch_size

    # Compute step sizes
    step_d = max(1, int(pD * (1 - overlap)))
    step_h = max(1, int(pH * (1 - overlap)))
    step_w = max(1, int(pW * (1 - overlap)))

    # Gaussian importance map
    importance = gaussian_weight_3d(patch_size).to(device)

    # Pad volume if smaller than patch
    pad_d = max(0, pD - D)
    pad_h = max(0, pH - H)
    pad_w = max(0, pW - W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d))
        D_pad, H_pad, W_pad = D + pad_d, H + pad_h, W + pad_w
    else:
        D_pad, H_pad, W_pad = D, H, W

    # Collect all patch coordinates — generate per-axis positions then Cartesian product
    def _axis_positions(length, patch_len, step):
        positions = list(range(0, length - patch_len + 1, step))
        # Ensure last position covers the edge
        if not positions or positions[-1] + patch_len < length:
            positions.append(length - patch_len)
        return sorted(set(positions))

    d_positions = _axis_positions(D_pad, pD, step_d)
    h_positions = _axis_positions(H_pad, pH, step_h)
    w_positions = _axis_positions(W_pad, pW, step_w)
    coords = [(d, h, w) for d in d_positions for h in h_positions for w in w_positions]

    # Initialize accumulators
    num_classes = 4  # BraTS
    output_sum = torch.zeros(1, num_classes, D_pad, H_pad, W_pad, device=device)
    weight_sum = torch.zeros(1, 1, D_pad, H_pad, W_pad, device=device)

    # Process patches in mini-batches
    for i in range(0, len(coords), sw_batch_size):
        batch_coords = coords[i:i + sw_batch_size]
        patches = []
        for (d, h, w) in batch_coords:
            patches.append(image[:, :, d:d+pD, h:h+pH, w:w+pW])

        patch_batch = torch.cat(patches, dim=0)  # [N, C, pD, pH, pW]
        N = patch_batch.shape[0]

        # Replicate text for batch
        if use_text and text_ids is not None:
            batch_text = text_ids.expand(N, -1)
            batch_mask = attention_mask.expand(N, -1) if attention_mask is not None else None
        else:
            batch_text = None
            batch_mask = None

        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            logits = model(patch_batch, batch_text, attention_mask=batch_mask, use_text=use_text)
            probs = F.softmax(logits, dim=1).float()

        # Accumulate weighted predictions
        for j, (d, h, w) in enumerate(batch_coords):
            output_sum[:, :, d:d+pD, h:h+pH, w:w+pW] += probs[j:j+1] * importance
            weight_sum[:, :, d:d+pD, h:h+pH, w:w+pW] += importance

    # Normalize
    output = output_sum / weight_sum.clamp(min=1e-8)

    # Remove padding
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        output = output[:, :, :D, :H, :W]

    return output


def load_model(config, checkpoint_path, device):
    """Load model from checkpoint."""
    model_cfg = config['model']
    training_cfg = config['training']

    model = TextMamba3D(
        img_size=tuple(model_cfg['img_size']),
        in_channels=model_cfg['in_channels'],
        out_channels=model_cfg['out_channels'],
        embed_dim=model_cfg['embed_dim'],
        depths=model_cfg['depths'],
        text_embed_dim=model_cfg['text_embed_dim'],
        text_max_len=model_cfg.get('text_max_len', 256),
        use_pretrained_text=model_cfg.get('use_pretrained_text', True),
        unfreeze_text_layers=model_cfg.get('unfreeze_text_layers', 0),
        use_checkpoint=training_cfg.get('gradient_checkpointing', False),
        text_model_path=model_cfg.get('text_model_path'),
        deep_supervision=training_cfg.get('deep_supervision', False),
        dropout=model_cfg.get('dropout', 0.0),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()

    epoch = ckpt.get('epoch', '?')
    best_dice = ckpt.get('best_dice', '?')
    print(f"Loaded checkpoint: epoch={epoch}, best_dice={best_dice}")

    return model


def main():
    args = parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(config, args.checkpoint, device)

    # Tokenizer
    model_cfg = config['model']
    text_model_path = model_cfg.get('text_model_path') or TextMambaEncoder.PUBMEDBERT_NAME
    tokenizer = AutoTokenizer.from_pretrained(text_model_path)

    # Dataset (no transform — we want full volumes)
    data_cfg = config['data']
    dataset = TextBraTSDataset(
        data_dir=data_cfg['data_dir'],
        split=args.split,
        transform=None,  # No crop — full volume
        tokenizer=tokenizer,
        max_text_len=model_cfg.get('text_max_len', 256),
        train_ratio=data_cfg.get('train_ratio', 0.596),
        val_ratio=data_cfg.get('val_ratio', 0.149),
    )

    patch_size = tuple(config['data']['patch_size'])
    eval_cfg = config.get('eval', {})
    overlap = args.overlap
    sw_batch_size = eval_cfg.get('sw_batch_size', 2)

    all_dice = []
    all_hd95 = []

    if args.save_preds:
        os.makedirs(args.save_preds, exist_ok=True)

    print(f"\nEvaluating {len(dataset)} cases ({args.split} split)")
    print(f"Sliding window: patch={patch_size}, overlap={overlap}, text={args.use_text}")
    print()

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc='Evaluating'):
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)  # [1, 4, D, H, W]
            mask = sample['mask']  # [D, H, W]
            case_name = sample['case_name']

            text_ids = sample['text_ids'].unsqueeze(0).to(device)
            attn_mask = sample['attention_mask'].unsqueeze(0).to(device)

            # Sliding window inference
            use_amp = config['training'].get('use_amp', True) and device.type == 'cuda'
            probs = sliding_window_inference(
                model, image, text_ids, attn_mask,
                patch_size=patch_size,
                overlap=overlap,
                use_text=args.use_text,
                use_amp=use_amp,
                sw_batch_size=sw_batch_size,
            )

            # Convert to prediction
            pred_argmax = probs.argmax(dim=1).cpu()  # [1, D, H, W]

            # Compute metrics
            dice = dice_score_brats_regions(probs.cpu(), mask.unsqueeze(0))
            all_dice.append(dice)

            pred_np = pred_argmax.squeeze().numpy()
            target_np = mask.numpy()
            hd95 = hausdorff_distance_95_brats_regions(pred_np, target_np)
            all_hd95.append(hd95)

            print(f"  {case_name}: Dice={dice['dice_mean']:.4f} "
                  f"(ET={dice['dice_ET']:.4f}, TC={dice['dice_TC']:.4f}, WT={dice['dice_WT']:.4f}) "
                  f"HD95_ET={hd95.get('hd95_ET', float('nan')):.2f}")

            # Optionally save predictions
            if args.save_preds:
                np.save(os.path.join(args.save_preds, f'{case_name}_pred.npy'), pred_np)

    # Aggregate results
    print("\n" + "=" * 60)
    print(f"Results: {args.split} split, {len(dataset)} cases, text={args.use_text}")
    print("=" * 60)

    metrics = {}
    for key in ['dice_ET', 'dice_TC', 'dice_WT', 'dice_mean']:
        values = [d[key] for d in all_dice]
        metrics[key] = (np.mean(values), np.std(values))
        print(f"  {key}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

    for key in ['hd95_ET', 'hd95_TC', 'hd95_WT']:
        values = [d[key] for d in all_hd95 if not np.isnan(d[key])]
        if values:
            metrics[key] = (np.mean(values), np.std(values))
            print(f"  {key}: {np.mean(values):.2f} +/- {np.std(values):.2f}")

    print("=" * 60)

    return metrics


if __name__ == '__main__':
    main()
