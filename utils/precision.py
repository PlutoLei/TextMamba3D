# utils/precision.py
"""Precision utilities for pure bf16 training (Mamba-3 Triton kernel workaround)."""

import contextlib

import torch
from torch.amp import autocast


def get_amp_context(bf16_mode, use_amp):
    """Return appropriate AMP context manager.

    When bf16_mode='pure', returns nullcontext (model/inputs already cast to bf16).
    Otherwise returns standard bfloat16 autocast.
    """
    if bf16_mode == 'pure':
        return contextlib.nullcontext()
    return autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp)
