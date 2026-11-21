"""Eager attention with float32 scores (opt-in: model argument ``attn_score_dtype=float32``).

transformers' eager attention computes ``matmul(q, k^T) * scaling`` in the
model's dtype and only then applies a float32 softmax. In bfloat16, the scores
are rounded before the softmax ever sees them. Some checkpoints have scores of
1e3 to 1e5 (large k_proj biases, for example), where bfloat16's spacing is 16
to 1024, so the token-dependent part of the score is rounded away. SDPA does
not round the scores this way. BLME switches a whole run to eager attention
whenever an attention-weight or double-backward task is in it (``core.py``).

``attn_score_dtype=float32`` makes that eager attention compute ``q k^T`` in
float32, and changes nothing else. The model's own ``eager_attention_forward``
(from its modeling module, so softcaps, GPT-2's scaling and every other
per-model detail are kept) runs inside a ``TorchFunctionMode``. The mode
upcasts the operands of its first matmul, the score product, which must be on
the query tensor passed in. The scaling, the mask and the float32 softmax then
work on float32 scores. The probabilities are cast back to the model's dtype
and multiplied with v exactly as transformers does, and the weights are
returned, so ``output_attentions`` and double backward still work. A float32
model gives bit-identical results. An attention module whose modeling module
has no ``eager_attention_forward``, or whose first matmul is not on the query,
raises instead of running plain eager.

The function is registered with transformers' ``AttentionInterface`` (and
the eager mask with ``AttentionMaskInterface``) as ``blme_eager_fp32_scores``.
The option is off by default. Without it, eager attention is transformers'
own, untouched.
"""
from __future__ import annotations

import sys
from typing import Any, Optional

import torch
from torch.overrides import TorchFunctionMode

FP32_SCORES = "blme_eager_fp32_scores"
SCORE_DTYPE_ARG = "attn_score_dtype"
_SCORE_DTYPES = {"model": "model", "auto": "model", "float32": "float32", "fp32": "float32"}
_MATMULS = (torch.matmul, torch.Tensor.matmul, torch.Tensor.__matmul__)
_registered = False


def parse_score_dtype(value: Optional[str]) -> str:
    """``"model"`` (default: scores in the model's dtype) or ``"float32"``."""
    if value is None:
        return "model"
    key = str(value).strip().lower()
    if key not in _SCORE_DTYPES:
        raise ValueError(f"{SCORE_DTYPE_ARG} must be 'model' or 'float32', got {value!r}")
    return _SCORE_DTYPES[key]


class _Float32Scores(TorchFunctionMode):
    """Upcasts the operands of the first matmul, which must be ``query @ ...``."""

    def __init__(self, query: torch.Tensor):
        super().__init__()
        self.query = query
        self.hits = 0
        self.problem: Optional[str] = None

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self.hits == 0 and func in _MATMULS:
            self.hits += 1
            a, b = args[0], args[1]
            if a is not self.query:
                self.problem = "its first matmul is not on the query tensor"
                return func(*args, **kwargs)
            return func(a.float(), b.float(), *args[2:], **kwargs)
        return func(*args, **kwargs)


def eager_fp32_scores_forward(module, query, key, value, attention_mask, **kwargs):
    """The model's own eager attention with ``q k^T`` computed in float32."""
    owner = sys.modules.get(type(module).__module__)
    eager = getattr(owner, "eager_attention_forward", None)
    if eager is None:
        raise NotImplementedError(
            f"{FP32_SCORES}: {type(module).__name__} ({type(module).__module__}) has no "
            "eager_attention_forward to run with float32 scores; use attn_score_dtype=model")
    mode = _Float32Scores(query)
    with mode:
        out = eager(module, query, key, value, attention_mask, **kwargs)
    if mode.hits != 1 or mode.problem:
        raise RuntimeError(
            f"{FP32_SCORES}: cannot find the score product in {type(module).__name__}'s eager "
            f"attention ({mode.problem or 'no matmul'}); use attn_score_dtype=model")
    return out


def register() -> None:
    """Register the function and the eager mask under ``blme_eager_fp32_scores`` (idempotent)."""
    global _registered
    if _registered:
        return
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface
    AttentionInterface.register(FP32_SCORES, eager_fp32_scores_forward)
    AttentionMaskInterface.register(FP32_SCORES, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
    _registered = True


def eager_implementation(model: Any) -> str:
    """The eager implementation a run should switch ``model`` to: the float32-score
    variant when the model was loaded with ``attn_score_dtype=float32``, else ``"eager"``."""
    from .dtype_contract import PRECISION_ATTR
    record = getattr(model, PRECISION_ATTR, None) or {}
    if record.get(SCORE_DTYPE_ARG) == "float32":
        register()
        return FP32_SCORES
    return "eager"
