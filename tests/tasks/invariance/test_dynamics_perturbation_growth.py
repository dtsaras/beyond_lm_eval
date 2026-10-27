"""dynamics_perturbation_growth: precision and depth invariance on tiny CPU models."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamics_fixtures import (dataset, inv, tiny_gpt2, tiny_neox, token_batches,  # noqa: E402
                               tokenizer)

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


# tiny gpt2 has 5 blocks; indices are "insert an identity copy after block i".
DEPTH_LAYOUTS = {"early": [0, 0], "late": [3, 3], "uniform": [0, 1, 2, 3], "end": [4]}


@pytest.mark.parametrize("where", sorted(DEPTH_LAYOUTS))
def test_growth_log_ratio_invariant_under_identity_blocks(where):
    """The end-to-end log ratio is bit-identical under function-preserving
    identity-block insertion anywhere in the stack."""
    tok = tokenizer()
    model = tiny_gpt2()
    deeper = inv.insert_identity_blocks(model, DEPTH_LAYOUTS[where],
                                        verify_on=token_batches(tok))
    base, other = _run(model, tok), _run(deeper, tok)
    inv.assert_invariant(base, other, ["growth_log_ratio"], rtol=0.0)


def test_demoted_forms_move_under_identity_blocks():
    """Known negatives: the per-block rate falls exactly as 1/L and the
    normalised-depth slope moves, so neither is a headline."""
    tok = tokenizer()
    model = tiny_gpt2()
    deeper = inv.insert_identity_blocks(model, DEPTH_LAYOUTS["early"])
    base, other = _run(model, tok), _run(deeper, tok)
    L0, L1 = base["_meta_n_transitions"], other["_meta_n_transitions"]
    assert (L0, L1) == (5, 7)
    assert other["_meta_growth_rate_per_block"] == pytest.approx(
        base["_meta_growth_rate_per_block"] * L0 / L1, rel=1e-12)
    inv.assert_moves(base, other, "_meta_growth_rate_per_block", 0.2)
    inv.assert_moves(base, other, "_meta_growth_slope", 1e-3)


def test_headline_keys_are_depth_free():
    out = _run(tiny_gpt2(), tokenizer())
    for k in ("growth_slope", "growth_rate_per_block", "initial_divergence"):
        assert k not in out, k
        assert f"_meta_{k}" in out, k


def test_token_mean_excludes_position0():
    """The divergence token mean drops position 0 (TOKEN_POLICY); the inclusive
    log ratio is kept as _meta_."""
    tok = tokenizer()
    out = _run(tiny_gpt2(), tok)
    n_tokens = sum(len(tok(d["text"])["input_ids"]) for d in dataset()[:CFG["num_samples"]])
    assert PerturbationGrowthTask.TOKEN_POLICY == "exclude_pos0"
    assert out["_meta_rows_policy"] == "exclude_pos0"
    assert out["_meta_rows_n_rows"] == n_tokens - CFG["num_samples"]
    assert out["growth_log_ratio"] != out["_meta_growth_log_ratio_incl_pos0"]
