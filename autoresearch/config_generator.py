"""Generate YAML configs for L1 hyperparameter search."""
import copy
import os
import yaml


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base dict."""
    result = copy.deepcopy(base)
    for k, v in overrides.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def generate_l1_config(
    base_config: str,
    overrides: dict,
    experiment_name: str,
) -> dict:
    """Generate a config for L1 hyperparameter search."""
    with open(base_config) as f:
        base = yaml.safe_load(f)
    cfg = _deep_merge(base, overrides)
    cfg.setdefault('experiment', {})
    cfg['experiment']['name'] = experiment_name
    return cfg


def save_config(cfg: dict, path: str) -> str:
    """Save config dict as YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path
