"""dynamics_gradient_flow: precision and depth/scale invariance on tiny CPU models."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamics_fixtures import (dataset, inv, tiny_gpt2, tiny_neox, token_batches,  # noqa: E402
                               tokenizer)

from blme.tasks.dynamics.gradient_flow import GradientFlowTask  # noqa: E402

CFG = {"num_samples": 4}


def _run(model, tok, cfg=CFG):
    with inv.limit_threads(1):
        out = GradientFlowTask(config=dict(cfg)).evaluate(model, tok, dataset())
    assert "error" not in out, out
    return out


@pytest.mark.parametrize("build", [tiny_gpt2, tiny_neox], ids=["gpt2", "neox"])
def test_bf16_model_computes_in_fp32(build):
    """A bf16 model must give exactly the bf16-weights/fp32-math profile.

    Pre-fix the forward and backward passes ran in bf16 (only the logits were
    promoted), and the per-layer gradient norms moved by percent-level amounts.
    """
    tok = tokenizer()
    model = build()
    bf16 = inv.run_with_dtype(model, torch.bfloat16)
    ref = inv.bf16_weights_fp32_math(model)
    got, want = _run(bf16, tok), _run(ref, tok)
    inv.assert_invariant(want, got, ["gradient_norm_per_layer"], rtol=0.0)
    assert got["_meta_compute_dtype"] == "float32"
    assert got["_meta_promoted_to_fp32"] is True
    assert {p.dtype for p in bf16.parameters()} == {torch.bfloat16}
    assert not any(p.requires_grad is False for p in bf16.parameters())
