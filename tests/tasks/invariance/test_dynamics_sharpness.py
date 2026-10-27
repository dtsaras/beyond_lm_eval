"""dynamics_sharpness: precision and reparameterisation invariance on tiny CPU models."""
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamics_fixtures import (dataset, inv, tiny_gpt2, tiny_neox, token_batches,  # noqa: E402
                               tokenizer)

from blme.tasks.dynamics.sharpness import LossSharpnessTask  # noqa: E402

CFG = {"num_samples": 3, "n_hutchinson": 4, "n_power_iter": 6, "seed": 0,
       "param_scope": "last_layer"}
KEYS = ["baseline_loss", "hutchinson_trace_estimate", "top_eigenvalue_estimate",
        "sam_sharpness"]


def _run(model, tok, cfg=CFG):
    with inv.limit_threads(1):
        out = LossSharpnessTask(config=dict(cfg)).evaluate(model, tok, dataset())
    assert "error" not in out, out
    return out


def test_bf16_buffers_are_promoted_and_restored():
    """A model cast whole to bf16 (its RoPE inv_freq buffer included) must give
    exactly the result of the same bf16 values held in fp32, report the dtypes
    it computed in, and come back in bf16. Pre-fix only parameters were promoted
    (no number moves on GPT-NeoX, whose rotary forward casts inv_freq to fp32)
    and nothing reported the compute dtype."""
    tok = tokenizer()
    bf16 = tiny_neox().to(torch.bfloat16)
    ref = copy.deepcopy(bf16).float()
    got, want = _run(bf16, tok), _run(ref, tok)
    inv.assert_invariant(want, got, KEYS, rtol=0.0)
    assert got["_meta_compute_dtype"] == "float32"
    assert got["_meta_n_buffers_promoted"] >= 1
    assert {p.dtype for p in bf16.parameters()} == {torch.bfloat16}
    assert {b.dtype for b in bf16.buffers() if b.is_floating_point()} == {torch.bfloat16}
