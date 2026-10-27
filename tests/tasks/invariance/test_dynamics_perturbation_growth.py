"""dynamics_perturbation_growth: precision and depth invariance on tiny CPU models."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamics_fixtures import dataset, inv, tiny_gpt2, tiny_neox, tokenizer  # noqa: E402

from blme.tasks.dynamics.perturbation_growth import PerturbationGrowthTask  # noqa: E402

CFG = {"num_samples": 4, "rel_eps": 1e-2, "seed": 0}


def _run(model, tok, cfg=CFG):
    with inv.limit_threads(1):
        out = PerturbationGrowthTask(config=dict(cfg)).evaluate(model, tok, dataset())
    assert "error" not in out, out
    return out


@pytest.mark.parametrize("build", [tiny_gpt2, tiny_neox], ids=["gpt2", "neox"])
def test_bf16_model_computes_in_fp32(build):
    """A bf16 model must give exactly the bf16-weights/fp32-math result.

    Pre-fix the forward passes ran in bf16, whose rounding is a noise floor of the
    same order as the 1e-2 perturbation, and the log ratio moved by tens of percent.
    """
    tok = tokenizer()
    model = build()
    bf16 = inv.run_with_dtype(model, torch.bfloat16)
    ref = inv.bf16_weights_fp32_math(model)
    got, want = _run(bf16, tok), _run(ref, tok)
    inv.assert_invariant(want, got, ["growth_log_ratio", "perturbation_growth_per_layer"],
                         rtol=0.0)
    assert got["_meta_compute_dtype"] == "float32"
    assert got["_meta_original_dtype"] == "bfloat16"
    assert got["_meta_promoted_to_fp32"] is True
    # The model is handed back in bf16.
    assert {p.dtype for p in bf16.parameters()} == {torch.bfloat16}


def test_fp32_model_is_not_promoted():
    out = _run(tiny_gpt2(), tokenizer())
    assert out["_meta_promoted_to_fp32"] is False
    assert out["_meta_compute_dtype"] == "float32"
