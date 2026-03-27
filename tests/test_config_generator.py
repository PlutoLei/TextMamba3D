import yaml
from autoresearch.config_generator import generate_l1_config, save_config

def test_generate_l1_config():
    cfg = generate_l1_config(
        base_config='configs/archive/textbrats_a100_v5.yaml',
        overrides={'training': {'lr': 0.0002, 'epochs': 140}},
        experiment_name='L1-test-lr0002',
    )
    assert cfg['training']['lr'] == 0.0002
    assert cfg['training']['epochs'] == 140
    assert cfg['experiment']['name'] == 'L1-test-lr0002'

def test_generate_l1_config_preserves_base():
    cfg = generate_l1_config(
        base_config='configs/archive/textbrats_a100_v5.yaml',
        overrides={'training': {'lr': 0.001}},
        experiment_name='test',
    )
    assert cfg['model']['embed_dim'] == 48
    assert cfg['data']['batch_size'] == 2
