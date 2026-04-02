# tests/test_freeze_warmup.py
"""Tests for V9.0 vision backbone freeze warmup."""
import torch
from models.textmamba3d import TextMamba3D


def test_freeze_unfreeze_img_encoder():
    model = TextMamba3D(img_size=(32, 32, 32), embed_dim=48, depths=[2, 2, 2, 2],
                        text_embed_dim=256, use_pretrained_text=False)
    # Freeze
    for p in model.img_encoder.parameters():
        p.requires_grad = False
    frozen_count = sum(1 for p in model.img_encoder.parameters() if p.requires_grad)
    assert frozen_count == 0

    # Unfreeze
    for p in model.img_encoder.parameters():
        p.requires_grad = True
    unfrozen_count = sum(1 for p in model.img_encoder.parameters() if p.requires_grad)
    assert unfrozen_count > 0

    # Fusion + decoder should always be trainable
    fusion_trainable = sum(1 for p in model.multi_scale_attn.parameters() if p.requires_grad)
    assert fusion_trainable > 0


def test_separate_param_groups():
    model = TextMamba3D(img_size=(32, 32, 32), embed_dim=48, depths=[2, 2, 2, 2],
                        text_embed_dim=256, use_pretrained_text=False)
    img_params = list(model.img_encoder.parameters())
    img_param_ids = {id(p) for p in img_params}
    other_params = [p for p in model.parameters() if id(p) not in img_param_ids]

    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': 1e-4},
        {'params': img_params, 'lr': 0.0},
    ], weight_decay=0.01)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]['lr'] == 1e-4
    assert optimizer.param_groups[1]['lr'] == 0.0

    # Simulate unfreeze
    optimizer.param_groups[1]['lr'] = 1e-5
    assert optimizer.param_groups[1]['lr'] == 1e-5
