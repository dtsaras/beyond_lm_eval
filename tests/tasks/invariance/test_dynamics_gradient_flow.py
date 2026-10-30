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


def test_norms_exclude_sink_rows():
    """The per-layer token pool drops position 0 and the massive rows; the
    loss and the inclusive profile (as _meta_) keep them."""
    tok = tokenizer()
    out = _run(tiny_gpt2(), tok)
    n_tokens = sum(len(tok(d["text"])["input_ids"]) for d in dataset()[:CFG["num_samples"]])
    assert GradientFlowTask.TOKEN_POLICY == "exclude_sinks"
    assert out["_meta_rows_policy"] == "exclude_sinks"
    assert out["_meta_rows_n_rows"] == (n_tokens - CFG["num_samples"]
                                        - out["_meta_sink_n_massive_dropped"])
    excl = out["gradient_norm_per_layer"]
    incl = out["_meta_gradient_norm_per_layer_incl_pos0"]
    assert all(e < i for e, i in zip(excl, incl))


@pytest.fixture(scope="module")
def gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2", attn_implementation="eager", dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"gpt2 not cached: {e}")
    return model, tok


def _run_policy(model, tok, monkeypatch, policy=None, hook=None):
    from _sinks import massive_row_hook
    from blme.tasks.common import get_layers
    with monkeypatch.context() as m:
        if policy is not None:
            m.setattr(GradientFlowTask, "TOKEN_POLICY", policy)
        if hook is None:
            return _run(model, tok)
        with massive_row_hook(model, len(get_layers(model)) // 2, *hook):
            return _run(model, tok)


def test_massive_row_at_one_middle_depth_is_dropped_at_every_point(gpt2, monkeypatch):
    """gpt2 with a first-period-style sink: block L/2's output at position 3 of
    every passage x1000 in every forward (the mask pass and the backward
    passes alike). exclude_pos0 keeps the sink token's gradient at every point
    and its end-to-end log ratio moves 2.835 -> 3.033; exclude_sinks drops that
    token at every point (one mask, from the middle depths) and moves a fifth
    as much -- what is left is the sink's effect on the other tokens through
    attention. With no massive row the two policies agree bit for bit."""
    model, tok = gpt2
    key = "gradient_flow_log_ratio_in_out"
    new, old = _run_policy(model, tok, monkeypatch), _run_policy(model, tok, monkeypatch,
                                                                  "exclude_pos0")
    assert new["_meta_sink_n_massive_dropped"] == 0
    inv.assert_invariant(old, new, [key, "gradient_norm_per_layer"], rtol=0.0)
    new_inj = _run_policy(model, tok, monkeypatch, hook=(3, 1000.0))
    old_inj = _run_policy(model, tok, monkeypatch, "exclude_pos0", hook=(3, 1000.0))
    assert new_inj["_meta_sink_n_massive_dropped"] == CFG["num_samples"]
    assert new_inj["_meta_massive_token_share"] > 0.5 > new["_meta_massive_token_share"]
    # One mask for every point: the token is gone from every pool.
    assert new_inj["_meta_rows_n_rows"] == old_inj["_meta_rows_n_rows"] - CFG["num_samples"]
    d_old = abs(old_inj[key] - old[key])
    d_new = abs(new_inj[key] - new[key])
    assert d_old > 0.1, d_old
    assert d_new < 0.5 * d_old, (d_new, d_old)
