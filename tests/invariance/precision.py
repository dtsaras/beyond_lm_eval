"""Precision transforms that separate storage error from arithmetic error.

fp32                      reference
bf16_weights_fp32_math    weights rounded to bf16, stored and computed in fp32:
                          differs from fp32 by STORAGE error only
run_with_dtype(bf16)      weights and arithmetic in bf16: storage + ARITHMETIC

So |bf16w_fp32 - fp32| is the storage error and |bf16 - bf16w_fp32| the
arithmetic error. Only parameters are cast, as ``blme.core`` does: casting
buffers would also round float32 RoPE tables (inv_freq) in GPT-NeoX/Llama.
"""
from __future__ import annotations

import copy
from typing import Dict

import torch

from .model_transforms import logits


def run_with_dtype(model, dtype: torch.dtype):
    """Copy of ``model`` with every floating parameter cast to ``dtype``.

    Buffers keep their dtype. The copy computes in ``dtype``.
    """
    new = copy.deepcopy(model).eval()
    with torch.no_grad():
        for p in new.parameters():
            if p.is_floating_point():
                p.data = p.data.to(dtype)
    return new


def bf16_weights_fp32_math(model, storage_dtype: torch.dtype = torch.bfloat16):
    """Copy of ``model`` whose parameters are rounded to ``storage_dtype`` and
    stored back as float32: bf16 storage, fp32 arithmetic."""
    new = copy.deepcopy(model).eval()
    with torch.no_grad():
        for p in new.parameters():
            if p.is_floating_point():
                p.data = p.data.to(storage_dtype).to(torch.float32)
    return new


def precision_error_split(model, inputs, low: torch.dtype = torch.bfloat16) -> Dict[str, float]:
    """max |dlogit| for storage, arithmetic and total error of ``low``.

    ``model`` must be float32. Returns {"storage", "arithmetic", "total"}.
    """
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError("precision_error_split needs a float32 reference model")
    ref = logits(model, inputs)
    stor = logits(bf16_weights_fp32_math(model, low), inputs)
    low_m = logits(run_with_dtype(model, low), inputs)

    def mx(a, b):
        return max(float((x - y).abs().max()) for x, y in zip(a, b))

    return {"storage": mx(stor, ref), "arithmetic": mx(low_m, stor), "total": mx(low_m, ref)}
