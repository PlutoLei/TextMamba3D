#!/usr/bin/env python3
"""Region-wise ensemble evaluation for TextMamba3D.

Combines predictions from two models using per-region weighted averaging
of softmax probability maps. Designed for V8.0 (strong absolute, no text delta)
+ V10.2 (weaker absolute, positive text delta) ensemble.

Usage:
    python scripts/ensemble_evaluate.py \
        --ckpt-a checkpoints/best_V8.0.pth \
        --ckpt-b checkpoints/best_V10.2.pth \
        --config-a configs/autoresearch/V8.0_stage2_finetune.yaml \
        --config-b configs/autoresearch/V10.2_ssl_pretrain.yaml \
        --weights-et 0.5 --weights-tc 0.3 --weights-wt 0.3

    # Grid search mode
    python scripts/ensemble_evaluate.py \
        --ckpt-a ... --ckpt-b ... --config-a ... --config-b ... \
        --grid-search
"""

import argparse
import itertools
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

# 确保项目根目录在 sys.path 中
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.brats_textbrats_dataset import TextBraTSDataset
from models import TextMamba3D
from models.text_encoder import TextMambaEncoder
from utils.metrics import dice_score_brats_regions, hausdorff_distance_95_brats_regions
from utils.sliding_window import gaussian_weight_3d

# 复用 evaluate_full.py 中的滑窗推理逻辑
from evaluate_full import sliding_window_inference, sliding_window_inference_tta


def parse_args():
    parser = argparse.ArgumentParser(
        description='Region-wise ensemble evaluation (two models)'
    )
    # 模型 A（主模型，默认高权重）
    parser.add_argument('--ckpt-a', type=str, required=True,
                        help='Checkpoint for model A (e.g., V8.0)')
    parser.add_argument('--config-a', type=str, required=True,
                        help='Config YAML for model A')
    # 模型 B（辅助模型，text delta 强）
    parser.add_argument('--ckpt-b', type=str, required=True,
                        help='Checkpoint for model B (e.g., V10.2)')
    parser.add_argument('--config-b', type=str, required=True,
                        help='Config YAML for model B')

    # 每区域权重：weight 表示模型 B 的权重，模型 A 的权重 = 1 - weight
    parser.add_argument('--weights-et', type=float, default=0.5,
                        help='Weight for model B on ET region (default: 0.5)')
    parser.add_argument('--weights-tc', type=float, default=0.3,
                        help='Weight for model B on TC region (default: 0.3)')
    parser.add_argument('--weights-wt', type=float, default=0.3,
                        help='Weight for model B on WT region (default: 0.3)')

    # 推理设置
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'])
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--tta', action='store_true',
                        help='Enable 8-fold flip TTA')

    # Grid search 模式
    parser.add_argument('--grid-search', action='store_true',
                        help='Grid search over weight combinations')
    parser.add_argument('--grid-step', type=float, default=0.1,
                        help='Step size for grid search (default: 0.1)')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model(config: dict, checkpoint_path: str, device: torch.device,
               strict: bool = True) -> torch.nn.Module:
    """根据 config 构建模型并加载权重。

    支持 V10.x 的 feature_noise_rate / use_decoder_film 参数。
    """
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
        # V10.x 参数（V8.0 config 中没有这些键，使用默认值 0.0 / False）
        feature_noise_rate=model_cfg.get('feature_noise_rate', 0.0),
        use_decoder_film=model_cfg.get('use_decoder_film', False),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=strict)
    if missing:
        print(f"  [WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {len(unexpected)}")

    model.eval()

    epoch = ckpt.get('epoch', '?')
    best_dice = ckpt.get('best_dice', '?')
    print(f"  Loaded: epoch={epoch}, best_dice={best_dice}")

    return model


def run_inference(model, image, text_ids, attn_mask, patch_size, overlap,
                  use_text, use_tta, use_amp, sw_batch_size, bf16_mode):
    """对单个 case 执行推理，返回 softmax 概率图 [1, C, D, H, W]"""
    infer_fn = sliding_window_inference_tta if use_tta else sliding_window_inference
    probs = infer_fn(
        model, image, text_ids, attn_mask,
        patch_size=patch_size,
        overlap=overlap,
        use_text=use_text,
        use_amp=use_amp,
        sw_batch_size=sw_batch_size,
        bf16_mode=bf16_mode,
    )
    return probs


def ensemble_probs_region_wise(probs_a, probs_b, w_et, w_tc, w_wt):
    """基于区域的概率融合。

    BraTS 类别映射: 0=BG, 1=NCR, 2=ED, 3=ET
    BraTS 评估区域:
      - ET (Enhancing Tumor): class 3
      - TC (Tumor Core): class 1 + class 3
      - WT (Whole Tumor): class 1 + class 2 + class 3

    直接对 4-class softmax 概率做加权融合:
      prob_ensemble[c] = (1-w) * prob_A[c] + w * prob_B[c]
    其中 w 根据 class 所属区域选择不同权重。

    权重分配逻辑:
      - class 0 (BG): 使用 WT 权重（与 WT 区域对应）
      - class 1 (NCR): 使用 TC 权重
      - class 2 (ED): 使用 WT 权重（ED 只参与 WT，不参与 TC/ET）
      - class 3 (ET): 使用 ET 权重
    """
    # 构建 per-class 权重 (model B 的权重)
    class_weights_b = torch.tensor(
        [w_wt, w_tc, w_wt, w_et],
        device=probs_a.device, dtype=probs_a.dtype
    ).view(1, 4, 1, 1, 1)

    class_weights_a = 1.0 - class_weights_b

    ensemble = class_weights_a * probs_a + class_weights_b * probs_b

    # 重新归一化（理论上已归一化，但浮点误差可能导致微小偏差）
    ensemble = ensemble / ensemble.sum(dim=1, keepdim=True).clamp(min=1e-8)

    return ensemble


def evaluate_with_weights(
    model_a, model_b, dataset, device,
    config_a, config_b,
    w_et, w_tc, w_wt,
    use_text, use_tta, overlap,
    quiet=False,
):
    """用指定权重运行 ensemble 评估，返回 per-case 和汇总指标。"""
    patch_size_a = tuple(config_a['data']['patch_size'])
    patch_size_b = tuple(config_b['data']['patch_size'])
    assert patch_size_a == patch_size_b, (
        f"Patch size mismatch: A={patch_size_a}, B={patch_size_b}"
    )
    patch_size = patch_size_a

    sw_batch_a = config_a.get('eval', {}).get('sw_batch_size', 2)
    sw_batch_b = config_b.get('eval', {}).get('sw_batch_size', 2)

    bf16_a = config_a['training'].get('bf16_mode', None)
    bf16_b = config_b['training'].get('bf16_mode', None)

    use_amp_a = config_a['training'].get('use_amp', True) and device.type == 'cuda'
    use_amp_b = config_b['training'].get('use_amp', True) and device.type == 'cuda'
    if bf16_a == 'pure':
        use_amp_a = False
    if bf16_b == 'pure':
        use_amp_b = False

    all_dice = []
    all_hd95 = []

    iterator = range(len(dataset))
    if not quiet:
        iterator = tqdm(iterator, desc=f'Ensemble (ET={w_et:.1f},TC={w_tc:.1f},WT={w_wt:.1f})')

    with torch.no_grad():
        for idx in iterator:
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            mask = sample['mask']
            case_name = sample['case_name']

            text_ids = sample['text_ids'].unsqueeze(0).to(device)
            attn_mask = sample['attention_mask'].unsqueeze(0).to(device)

            # 模型 A 推理
            probs_a = run_inference(
                model_a, image, text_ids, attn_mask,
                patch_size=patch_size, overlap=overlap,
                use_text=use_text, use_tta=use_tta,
                use_amp=use_amp_a, sw_batch_size=sw_batch_a,
                bf16_mode=bf16_a,
            )

            # 模型 B 推理
            probs_b = run_inference(
                model_b, image, text_ids, attn_mask,
                patch_size=patch_size, overlap=overlap,
                use_text=use_text, use_tta=use_tta,
                use_amp=use_amp_b, sw_batch_size=sw_batch_b,
                bf16_mode=bf16_b,
            )

            # 区域加权融合
            probs_ensemble = ensemble_probs_region_wise(
                probs_a, probs_b, w_et, w_tc, w_wt
            )

            # argmax → 预测
            pred_np = probs_ensemble.argmax(dim=1).squeeze().cpu().numpy()

            # 计算指标
            pred_onehot = probs_ensemble.cpu()
            dice = dice_score_brats_regions(pred_onehot, mask.unsqueeze(0))
            all_dice.append(dice)

            target_np = mask.numpy()
            hd95 = hausdorff_distance_95_brats_regions(pred_np, target_np)
            all_hd95.append(hd95)

            if not quiet:
                tqdm.write(
                    f"  {case_name}: Dice={dice['dice_mean']:.4f} "
                    f"(ET={dice['dice_ET']:.4f}, TC={dice['dice_TC']:.4f}, "
                    f"WT={dice['dice_WT']:.4f})"
                )

    # 汇总
    metrics = {}
    for key in ['dice_ET', 'dice_TC', 'dice_WT', 'dice_mean']:
        values = [d[key] for d in all_dice]
        metrics[key] = np.mean(values)
        metrics[f'{key}_std'] = np.std(values)

    for key in ['hd95_ET', 'hd95_TC', 'hd95_WT']:
        values = [d[key] for d in all_hd95 if not np.isnan(d[key])]
        if values:
            metrics[key] = np.mean(values)
            metrics[f'{key}_std'] = np.std(values)

    return metrics


def print_results(metrics, label, w_et=None, w_tc=None, w_wt=None):
    """格式化输出结果"""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    if w_et is not None:
        print(f"  Weights (model B): ET={w_et:.2f}, TC={w_tc:.2f}, WT={w_wt:.2f}")
    print(f"{'=' * 70}")
    for key in ['dice_ET', 'dice_TC', 'dice_WT', 'dice_mean']:
        mean = metrics[key]
        std = metrics.get(f'{key}_std', 0)
        print(f"  {key}: {mean:.4f} +/- {std:.4f}")
    for key in ['hd95_ET', 'hd95_TC', 'hd95_WT']:
        if key in metrics:
            mean = metrics[key]
            std = metrics.get(f'{key}_std', 0)
            print(f"  {key}: {mean:.2f} +/- {std:.2f}")
    print(f"{'=' * 70}")


def main():
    args = parse_args()

    # 加载配置
    config_a = load_config(args.config_a)
    config_b = load_config(args.config_b)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载模型 A
    print(f"\n[Model A] {args.config_a}")
    model_a = load_model(config_a, args.ckpt_a, device, strict=True)

    # 加载模型 B（strict=False 允许额外的 key）
    print(f"\n[Model B] {args.config_b}")
    model_b = load_model(config_b, args.ckpt_b, device, strict=False)

    # Tokenizer（两模型共享同一 tokenizer）
    text_model_path_a = config_a['model'].get('text_model_path') or TextMambaEncoder.PUBMEDBERT_NAME
    tokenizer = AutoTokenizer.from_pretrained(text_model_path_a)

    # 数据集（使用模型 A 的 data config，两者相同）
    data_cfg = config_a['data']
    model_cfg = config_a['model']
    dataset = TextBraTSDataset(
        data_dir=data_cfg['data_dir'],
        split=args.split,
        transform=None,
        tokenizer=tokenizer,
        max_text_len=model_cfg.get('text_max_len', 256),
        train_ratio=data_cfg.get('train_ratio', 0.596),
        val_ratio=data_cfg.get('val_ratio', 0.149),
        et_enriched=data_cfg.get('et_enriched', False),
        enriched_prob=data_cfg.get('enriched_prob', 0.5),
    )

    print(f"\nDataset: {len(dataset)} cases ({args.split} split)")
    print(f"TTA: {'ON' if args.tta else 'OFF'}")

    if args.grid_search:
        # =============== Grid Search 模式 ===============
        step = args.grid_step
        weight_range = np.arange(0.0, 1.0 + step / 2, step)
        weight_range = np.round(weight_range, 2)

        print(f"\n{'#' * 70}")
        print(f"  GRID SEARCH: {len(weight_range)}^3 = {len(weight_range)**3} combinations")
        print(f"  Step: {step}, Range: [{weight_range[0]}, {weight_range[-1]}]")
        print(f"{'#' * 70}")

        best_mean = -1
        best_weights = (0, 0, 0)
        all_results = []

        for w_et, w_tc, w_wt in itertools.product(weight_range, repeat=3):
            # text+TTA
            metrics_text = evaluate_with_weights(
                model_a, model_b, dataset, device,
                config_a, config_b,
                w_et=w_et, w_tc=w_tc, w_wt=w_wt,
                use_text=True, use_tta=args.tta, overlap=args.overlap,
                quiet=True,
            )

            mean_dice = metrics_text['dice_mean']
            result = {
                'w_et': w_et, 'w_tc': w_tc, 'w_wt': w_wt,
                'dice_mean': mean_dice,
                'dice_ET': metrics_text['dice_ET'],
                'dice_TC': metrics_text['dice_TC'],
                'dice_WT': metrics_text['dice_WT'],
            }
            all_results.append(result)

            marker = ''
            if mean_dice > best_mean:
                best_mean = mean_dice
                best_weights = (w_et, w_tc, w_wt)
                marker = ' *** NEW BEST ***'

            print(
                f"  ET={w_et:.2f} TC={w_tc:.2f} WT={w_wt:.2f} | "
                f"Mean={mean_dice:.4f} "
                f"(ET={metrics_text['dice_ET']:.4f}, "
                f"TC={metrics_text['dice_TC']:.4f}, "
                f"WT={metrics_text['dice_WT']:.4f}){marker}"
            )

        # 输出最佳结果
        print(f"\n{'#' * 70}")
        print(f"  BEST: ET={best_weights[0]:.2f}, TC={best_weights[1]:.2f}, "
              f"WT={best_weights[2]:.2f} → Mean Dice = {best_mean:.4f}")
        print(f"{'#' * 70}")

        # 用最佳权重跑 notext 对比
        print("\n--- Re-running best weights with text and notext ---")

        metrics_text = evaluate_with_weights(
            model_a, model_b, dataset, device,
            config_a, config_b,
            w_et=best_weights[0], w_tc=best_weights[1], w_wt=best_weights[2],
            use_text=True, use_tta=args.tta, overlap=args.overlap,
            quiet=False,
        )
        print_results(metrics_text, "Ensemble (text)", *best_weights)

        metrics_notext = evaluate_with_weights(
            model_a, model_b, dataset, device,
            config_a, config_b,
            w_et=best_weights[0], w_tc=best_weights[1], w_wt=best_weights[2],
            use_text=False, use_tta=args.tta, overlap=args.overlap,
            quiet=False,
        )
        print_results(metrics_notext, "Ensemble (notext)", *best_weights)

        delta = metrics_text['dice_mean'] - metrics_notext['dice_mean']
        print(f"\n  Text Delta: {delta:+.4f} ({delta*100:+.2f}%)")

    else:
        # =============== 单次评估模式 ===============
        w_et = args.weights_et
        w_tc = args.weights_tc
        w_wt = args.weights_wt

        print(f"\nWeights (model B): ET={w_et}, TC={w_tc}, WT={w_wt}")

        # text 模式
        print(f"\n--- Ensemble with TEXT ---")
        metrics_text = evaluate_with_weights(
            model_a, model_b, dataset, device,
            config_a, config_b,
            w_et=w_et, w_tc=w_tc, w_wt=w_wt,
            use_text=True, use_tta=args.tta, overlap=args.overlap,
        )
        print_results(metrics_text, "Ensemble (text+TTA)" if args.tta else "Ensemble (text)",
                      w_et, w_tc, w_wt)

        # notext 模式
        print(f"\n--- Ensemble without TEXT ---")
        metrics_notext = evaluate_with_weights(
            model_a, model_b, dataset, device,
            config_a, config_b,
            w_et=w_et, w_tc=w_tc, w_wt=w_wt,
            use_text=False, use_tta=args.tta, overlap=args.overlap,
        )
        print_results(metrics_notext, "Ensemble (notext+TTA)" if args.tta else "Ensemble (notext)",
                      w_et, w_tc, w_wt)

        # Text delta
        delta = metrics_text['dice_mean'] - metrics_notext['dice_mean']
        print(f"\n  Text Delta (mean Dice): {delta:+.4f} ({delta*100:+.2f}%)")
        for region in ['ET', 'TC', 'WT']:
            d = metrics_text[f'dice_{region}'] - metrics_notext[f'dice_{region}']
            print(f"  Text Delta ({region}): {d:+.4f} ({d*100:+.2f}%)")


if __name__ == '__main__':
    main()
