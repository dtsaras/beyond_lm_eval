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


@pytest.mark.parametrize("where", [[0, 0], [4], [0, 1, 2, 3]], ids=["early", "end", "uniform"])
def test_convexity_gap_invariant_under_identity_blocks(where):
    """Only the final state is read, so function-preserving identity blocks
    leave the gap bit-identical."""
    tok = tokenizer()
    model = tiny_gpt2()
    deeper = inv.insert_identity_blocks(model, where, verify_on=token_batches(tok))
    inv.assert_invariant(_run(model, tok), _run(deeper, tok), ["convexity_gap"], rtol=0.0)


def test_convexity_gap_moves_when_depth_changes_the_function():
    """Control: duplicating a real block (not function-preserving) moves the gap,
    so the identity-block test has power."""
    import copy
    tok = tokenizer()
    model = tiny_gpt2()
    other = copy.deepcopy(model)
    blocks = list(other.transformer.h)
    other.transformer.h = torch.nn.ModuleList(blocks[:2] + [copy.deepcopy(blocks[1])] + blocks[2:])
    other.config.n_layer = len(other.transformer.h)
    for i, b in enumerate(other.transformer.h):
        b.attn.layer_idx = i
    inv.assert_moves(_run(model, tok), _run(other, tok), "convexity_gap", 1e-3)


def test_contexts_are_cut_by_tokens_not_characters():
    """Two passages sharing their first 128 characters but not their first 128
    tokens must give two different contexts. Pre-fix the 128-character cut made
    them identical, so the pair's gap was exactly 0."""
    tok = tokenizer()
    shared = "The committee met in the old town hall on Thursday evening to discuss " \
             "the new budget, the repairs to the bridge and the schedule for the summer fair."
    assert len(shared) > 128 and len(tok(shared)["input_ids"]) < 100
    data = [{"text": shared + " The mayor spoke first."},
            {"text": shared + " Nobody came at all, and the doors stayed shut."}]
    cfg = {"num_samples": 2, "num_pairs": 1, "steps": 3, "seed": 0}
    with inv.limit_threads(1):
        out = LatentInterpolationTask(config=cfg).evaluate(tiny_gpt2(), tok, data)
    assert out["convexity_gap"] != 0.0
    n_tok = sorted(len(tok(d["text"])["input_ids"]) for d in data)
    assert out["_meta_context_tokens_mean"] == pytest.approx(sum(n_tok) / 2)
    assert out["_meta_max_tokens"] == 128
