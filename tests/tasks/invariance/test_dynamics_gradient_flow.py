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


# tiny gpt2 has 5 blocks; indices are "insert an identity copy after block i".
DEPTH_LAYOUTS = {"early": [0, 0], "late": [3, 3], "uniform": [0, 1, 2, 3], "end": [4, 4]}


@pytest.mark.parametrize("where", sorted(DEPTH_LAYOUTS))
def test_end_to_end_log_ratio_invariant_under_identity_blocks(where):
    """log(g_emb / g_final) is bit-identical under identity blocks anywhere,
    including appended after the last block (its endpoint is the final residual)."""
    tok = tokenizer()
    model = tiny_gpt2()
    deeper = inv.insert_identity_blocks(model, DEPTH_LAYOUTS[where],
                                        verify_on=token_batches(tok))
    base, other = _run(model, tok), _run(deeper, tok)
    inv.assert_invariant(base, other, ["gradient_flow_log_ratio_in_out"], rtol=0.0)
    assert other["_meta_final_endpoint"] == "final_norm_input"


def test_block_input_endpoint_moves_when_a_block_is_appended():
    """Known negative: the former endpoint (input of the last block) moves."""
    tok = tokenizer()
    model = tiny_gpt2()
    deeper = inv.insert_identity_blocks(model, DEPTH_LAYOUTS["end"])
    base, other = _run(model, tok), _run(deeper, tok)
    inv.assert_moves(base, other, "_meta_gradient_flow_log_ratio_block_inputs", 1e-3)


def test_end_to_end_log_ratio_invariant_under_residual_rescale():
    """A common residual-stream rescale (exact for c = 2 with rescaled norm eps)
    halves every gradient norm and cancels in the ratio."""
    tok = tokenizer()
    model = tiny_gpt2()
    scaled = inv.residual_scale_gauge(model, 2.0, rescale_eps=True,
                                      verify_on=token_batches(tok), atol=0.0)
    base, other = _run(model, tok), _run(scaled, tok)
    inv.assert_invariant(base, other, ["gradient_flow_log_ratio_in_out"], rtol=1e-12)
    assert other["gradient_norm_per_layer"][-1] == pytest.approx(
        base["gradient_norm_per_layer"][-1] / 2, rel=1e-12)


def test_profile_summaries_move_under_identity_blocks_and_are_meta():
    """Known negatives: the entropy, slope and vanishing ratio of the profile
    move under function-preserving identity-block insertion; the raw norms
    halve under a residual x2 rescale. None of them is a headline."""
    tok = tokenizer()
    model = tiny_gpt2()
    deeper = inv.insert_identity_blocks(model, DEPTH_LAYOUTS["early"])
    base, other = _run(model, tok), _run(deeper, tok)
    inv.assert_moves(base, other, "_meta_gradient_flow_entropy", 1e-3)
    inv.assert_moves(base, other, "_meta_gradient_flow_slope", 1e-3)
    scaled = inv.residual_scale_gauge(model, 2.0, rescale_eps=True)
    inv.assert_moves(base, _run(scaled, tok), "_meta_gradient_norm_mean", 0.4)
    for k in ("gradient_flow_entropy", "gradient_flow_slope", "gradient_vanishing_ratio",
              "gradient_norm_mean", "gradient_norm_max", "gradient_norm_min"):
        assert k not in base and f"_meta_{k}" in base, k


def test_norms_exclude_position0():
    """The per-layer token pool drops position 0; the loss and the inclusive
    profile (as _meta_) keep it."""
    tok = tokenizer()
    out = _run(tiny_gpt2(), tok)
    n_tokens = sum(len(tok(d["text"])["input_ids"]) for d in dataset()[:CFG["num_samples"]])
    assert GradientFlowTask.TOKEN_POLICY == "exclude_pos0"
    assert out["_meta_rows_n_rows"] == n_tokens - CFG["num_samples"]
    excl = out["gradient_norm_per_layer"]
    incl = out["_meta_gradient_norm_per_layer_incl_pos0"]
    assert all(e < i for e, i in zip(excl, incl))
