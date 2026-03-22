# evaluate_full.py
"""Full-volume evaluation with sliding window inference for TextMamba3D."""

import argparse
import os
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import label as ndimage_label
from tqdm import tqdm
from transformers import AutoTokenizer

from data.brats_textbrats_dataset import TextBraTSDataset
from models import TextMamba3D
from models.text_encoder import TextMambaEncoder
from utils.metrics import dice_score_brats_regions, hausdorff_distance_95_brats_regions
from utils.sliding_window import gaussian_weight_3d


def parse_args():
    parser = argparse.ArgumentParser(description='Full-volume evaluation with sliding window')
    parser.add_argument('--config', type=str, default='configs/a100.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--use-text', action='store_true', default=True)
    parser.add_argument('--no-text', dest='use_text', action='store_false')
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--tta', action='store_true', help='Enable 8-fold flip TTA')
    parser.add_argument('--postprocess', action='store_true',
                        help='Enable ET connected-component post-processing')
    parser.add_argument('--et-min-size', type=int, default=50,
                        help='Min voxel count for ET connected components (default: 50)')
    parser.add_argument('--save-preds', type=str, default=None, help='Directory to save predictions')
    return parser.parse_args()


from utils.precision import get_amp_context


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
    bf16_mode=None,
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

    if bf16_mode == 'pure':
        image = image.to(dtype=torch.bfloat16)

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

        with get_amp_context(bf16_mode, use_amp):
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


def sliding_window_inference_tta(
    model,
    image: torch.Tensor,
    text_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    patch_size: tuple,
    overlap: float = 0.5,
    use_text: bool = True,
    use_amp: bool = True,
    sw_batch_size: int = 2,
    bf16_mode=None,
) -> torch.Tensor:
    """8-fold TTA: flip along each spatial axis combination, average softmax probs.

    Flip combos: 2^3 = 8 (no flip, flip D, flip H, flip W, flip DH, ..., flip DHW).
    Each flip is applied to the input image, inference is run, then the output
    is flipped back before accumulation.

    Returns:
        [1, num_classes, D, H, W] averaged softmax probabilities.
    """
    # spatial dims for [B, C, D, H, W] are 2, 3, 4
    spatial_dims = (2, 3, 4)
    flip_combos = list(product([False, True], repeat=3))  # 8 combos

    prob_sum = None

    for flip_d, flip_h, flip_w in flip_combos:
        # Build list of dims to flip
        dims_to_flip = []
        if flip_d:
            dims_to_flip.append(spatial_dims[0])
        if flip_h:
            dims_to_flip.append(spatial_dims[1])
        if flip_w:
            dims_to_flip.append(spatial_dims[2])

        # Flip input
        img_aug = image
        for d in dims_to_flip:
            img_aug = torch.flip(img_aug, [d])

        # Infer
        probs = sliding_window_inference(
            model, img_aug, text_ids, attention_mask,
            patch_size=patch_size, overlap=overlap,
            use_text=use_text, use_amp=use_amp,
            sw_batch_size=sw_batch_size,
            bf16_mode=bf16_mode,
        )

        # Flip output back
        for d in dims_to_flip:
            probs = torch.flip(probs, [d])

        if prob_sum is None:
            prob_sum = probs
        else:
            prob_sum = prob_sum + probs

    return prob_sum / len(flip_combos)


def postprocess_et(pred_argmax: np.ndarray, et_class: int = 3, min_size: int = 50) -> np.ndarray:
    """Remove small ET connected components (false positives).

    Args:
        pred_argmax: [D, H, W] integer label map.
        et_class: label value for enhancing tumor.
        min_size: minimum voxel count to keep an ET component.

    Returns:
        Cleaned label map with small ET components reclassified as necrotic (class 1).
    """
    pred = pred_argmax.copy()
    et_mask = (pred == et_class)
    if et_mask.sum() == 0:
        return pred

    labeled, n_components = ndimage_label(et_mask)

    for comp_id in range(1, n_components + 1):
        comp_mask = (labeled == comp_id)
        if comp_mask.sum() < min_size:
            # Reclassify small ET as necrotic core (class 1) — stays in TC but not ET
            pred[comp_mask] = 1

    return pred


def load_model(config, checkpoint_path, device, bf16_mode=None):
    """Load model from checkpoint."""
    model_cfg = config['model']
    training_cfg = config['training']

    model = TextMamba3D(
        img_size=tuple(model_cfg['img_size']),
        in_channels=model_cfg['in_channels'],
        out_channels=model_cfg['out_channels'],
        embed_dim=model_cfg['embed_dim'],
        depths=model_cfg['depths'],
        d_state=model_cfg.get('d_state', 16),
        text_embed_dim=model_cfg['text_embed_dim'],
        text_max_len=model_cfg.get('text_max_len', 256),
        use_pretrained_text=model_cfg.get('use_pretrained_text', True),
        unfreeze_text_layers=model_cfg.get('unfreeze_text_layers', 0),
        use_checkpoint=training_cfg.get('gradient_checkpointing', False),
        text_model_path=model_cfg.get('text_model_path'),
        deep_supervision=training_cfg.get('deep_supervision', False),
        dropout=model_cfg.get('dropout', 0.0),
        use_text_gate=model_cfg.get('use_text_gate', False),
        use_cross_scale_skip=model_cfg.get('use_cross_scale_skip', False),
        text_gate_init_bias=model_cfg.get('text_gate_init_bias', 2.0),
        use_mamba3=model_cfg.get('use_mamba3', False),
        headdim=model_cfg.get('headdim', None),
        rope_fraction=model_cfg.get('rope_fraction', None),
        chunk_size=model_cfg.get('chunk_size', None),
        is_mimo=model_cfg.get('is_mimo', False),
        fusion_type=model_cfg.get('fusion_type', 'seqca'),
        use_edge_enhance=model_cfg.get('use_edge_enhance', False),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])

    if bf16_mode == 'pure':
        model.to_bf16_with_fp32_text()
        print('Pure bf16 mode: model bf16, BERT fp32')

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
    bf16_mode = config['training'].get('bf16_mode', None)
    model = load_model(config, args.checkpoint, device, bf16_mode=bf16_mode)

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
        et_enriched=data_cfg.get('et_enriched', False),
        enriched_prob=data_cfg.get('enriched_prob', 0.5),
    )

    patch_size = tuple(config['data']['patch_size'])
    eval_cfg = config.get('eval', {})
    overlap = args.overlap
    sw_batch_size = eval_cfg.get('sw_batch_size', 2)

    all_dice = []
    all_hd95 = []

    if args.save_preds:
        os.makedirs(args.save_preds, exist_ok=True)

    use_tta = args.tta
    use_postprocess = args.postprocess
    et_min_size = args.et_min_size

    print(f"\nEvaluating {len(dataset)} cases ({args.split} split)")
    print(f"Sliding window: patch={patch_size}, overlap={overlap}, text={args.use_text}")
    if use_tta:
        print("TTA: 8-fold flip ensemble ENABLED")
    if use_postprocess:
        print(f"Post-processing: ET connected-component filter (min_size={et_min_size})")
    print()

    use_amp = config['training'].get('use_amp', True) and device.type == 'cuda'
    if bf16_mode == 'pure':
        use_amp = False
    infer_fn = sliding_window_inference_tta if use_tta else sliding_window_inference

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc='Evaluating'):
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)  # [1, 4, D, H, W]
            mask = sample['mask']  # [D, H, W]
            case_name = sample['case_name']

            text_ids = sample['text_ids'].unsqueeze(0).to(device)
            attn_mask = sample['attention_mask'].unsqueeze(0).to(device)
            probs = infer_fn(
                model, image, text_ids, attn_mask,
                patch_size=patch_size,
                overlap=overlap,
                use_text=args.use_text,
                use_amp=use_amp,
                sw_batch_size=sw_batch_size,
                bf16_mode=bf16_mode,
            )

            # Convert to prediction
            pred_argmax = probs.argmax(dim=1).cpu()  # [1, D, H, W]
            pred_np = pred_argmax.squeeze().numpy()

            # Post-processing: remove small ET components
            if use_postprocess:
                pred_np = postprocess_et(pred_np, et_class=3, min_size=et_min_size)
                # Rebuild tensor for dice_score_brats_regions (needs softmax-like input)
                # Use one-hot from cleaned argmax as "probabilities"
                pred_onehot = torch.zeros(1, 4, *pred_np.shape)
                for c in range(4):
                    pred_onehot[0, c] = torch.from_numpy((pred_np == c).astype(np.float32))
                probs_for_dice = pred_onehot
            else:
                probs_for_dice = probs.cpu()

            # Compute metrics
            dice = dice_score_brats_regions(probs_for_dice, mask.unsqueeze(0))
            all_dice.append(dice)

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
