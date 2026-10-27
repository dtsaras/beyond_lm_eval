"""dynamics_interpolation: precision and depth invariance on tiny CPU models."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamics_fixtures import (dataset, inv, tiny_gpt2, tiny_neox, token_batches,  # noqa: E402
                               tokenizer)

from blme.tasks.dynamics.trajectories import LatentInterpolationTask  # noqa: E402

CFG = {"num_samples": 8, "num_pairs": 6, "steps": 3, "seed": 0}


def _run(model, tok, cfg=CFG):
    with inv.limit_threads(1):
        out = LatentInterpolationTask(config=dict(cfg)).evaluate(model, tok, dataset())
    assert "error" not in out, out
    return out


@pytest.mark.parametrize("build", [tiny_gpt2, tiny_neox], ids=["gpt2", "neox"])
def test_bf16_model_computes_and_decodes_in_fp32(build):
    """A bf16 model must give exactly the bf16-weights/fp32-math gap.

    Pre-fix the forward pass and the LM-head decode of the interpolant ran in
    bf16, so the entropies carried bf16 rounding.
    """
    tok = tokenizer()
    model = build()
    bf16 = inv.run_with_dtype(model, torch.bfloat16)
    ref = inv.bf16_weights_fp32_math(model)
    got, want = _run(bf16, tok), _run(ref, tok)
    inv.assert_invariant(want, got, ["convexity_gap",
                                     "_meta_convexity_gap_endpoint_entropy_mean"], rtol=0.0)
    assert got["_meta_compute_dtype"] == "float32"
    assert got["_meta_promoted_to_fp32"] is True
    assert {p.dtype for p in bf16.parameters()} == {torch.bfloat16}
