"""Parse evaluation output from evaluate_full.py."""
import re


def parse_eval_output(text: str) -> dict:
    """Extract dice/hd95 metrics from evaluate_full.py stdout."""
    metrics = {}
    for key in ['dice_ET', 'dice_TC', 'dice_WT', 'dice_mean',
                'hd95_ET', 'hd95_TC', 'hd95_WT']:
        match = re.search(rf'{key}: ([\d.]+) \+/- ([\d.]+)', text)
        if match:
            metrics[key] = float(match.group(1))
            metrics[f'{key}_std'] = float(match.group(2))
    return metrics


def is_improvement(metrics: dict, baseline_mean_dice: float = 0.8479) -> bool:
    """Check if experiment improved over baseline."""
    return metrics.get('dice_mean', 0) > baseline_mean_dice
