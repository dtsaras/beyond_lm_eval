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


def test_projecting_model_keeps_its_last_block():
    """OPT with word_embed_proj_dim < hidden_size projects HF's last entry to
    the embedding width. The raw last-block capture (hidden width) used to be
    rejected for not matching that entry, and the profile then dropped the last
    block (SIGNOFF_MATRIX codex RED: rate +50% on a 3-block model)."""
    from transformers import OPTConfig, OPTForCausalLM
    torch.manual_seed(0)
    cfg = OPTConfig(vocab_size=1000, hidden_size=32, word_embed_proj_dim=16,
                    num_hidden_layers=3, ffn_dim=64, num_attention_heads=4,
                    max_position_embeddings=128, do_layer_norm_before=False,
                    attn_implementation="eager")
    model = OPTForCausalLM(cfg).eval()
    out = _run(model, tokenizer())
    assert out["_meta_last_entry_is_block_output"] is True
    assert out["_meta_n_transitions"] == 3
    assert len(out["perturbation_growth_per_layer"]) == 4


def test_token_mean_excludes_sink_rows():
    """The divergence token mean drops position 0 and the massive rows
    (TOKEN_POLICY); the inclusive log ratio is kept as _meta_."""
    tok = tokenizer()
    out = _run(tiny_gpt2(), tok)
    n_tokens = sum(len(tok(d["text"])["input_ids"]) for d in dataset()[:CFG["num_samples"]])
    assert PerturbationGrowthTask.TOKEN_POLICY == "exclude_sinks"
    assert out["_meta_rows_policy"] == "exclude_sinks"
    assert out["_meta_rows_n_rows"] == (n_tokens - CFG["num_samples"]
                                        - out["_meta_sink_n_massive_dropped"])
    assert out["growth_log_ratio"] != out["_meta_growth_log_ratio_incl_pos0"]


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
            m.setattr(PerturbationGrowthTask, "TOKEN_POLICY", policy)
        if hook is None:
            return _run(model, tok)
        with massive_row_hook(model, len(get_layers(model)) // 2, *hook):
            return _run(model, tok)


def test_massive_row_at_one_middle_depth_is_dropped_at_every_layer(gpt2, monkeypatch):
    """gpt2 with a first-period-style sink: block L/2's output at position 3 of
    every passage x1000 (clean and perturbed passes alike). The sink's
    divergence is x1000 from there on and dominates exclude_pos0's token mean
    (growth_log_ratio 2.64 -> 6.48); exclude_sinks drops that token at every
    layer and barely moves (the other tokens attend to it). With no massive
    row the two policies agree bit for bit."""
    model, tok = gpt2
    new, old = _run_policy(model, tok, monkeypatch), _run_policy(model, tok, monkeypatch,
                                                                  "exclude_pos0")
    assert new["_meta_sink_n_massive_dropped"] == 0
    inv.assert_invariant(old, new, ["growth_log_ratio", "perturbation_growth_per_layer"],
                         rtol=0.0)
    new_inj = _run_policy(model, tok, monkeypatch, hook=(3, 1000.0))
    old_inj = _run_policy(model, tok, monkeypatch, "exclude_pos0", hook=(3, 1000.0))
    assert new_inj["_meta_sink_n_massive_dropped"] == CFG["num_samples"]
    assert new_inj["_meta_massive_token_share"] > 0.5 > new["_meta_massive_token_share"]
    d_old = old_inj["growth_log_ratio"] - old["growth_log_ratio"]
    d_new = abs(new_inj["growth_log_ratio"] - new["growth_log_ratio"])
    assert d_old > 1.0, d_old
    assert d_new < 0.01 * d_old, (d_new, d_old)


def test_excluded_sink_content_cannot_reach_the_headline(gpt2, monkeypatch):
    """At the LAST position no other token sees the sink (causal attention),
    so exclude_sinks must not depend on its content at all: x1000 and x2000
    give the same value bit for bit, while exclude_pos0 reads the row."""
    model, tok = gpt2
    a = _run_policy(model, tok, monkeypatch, hook=(-1, 1000.0))
    b = _run_policy(model, tok, monkeypatch, hook=(-1, 2000.0))
    assert a["_meta_sink_n_massive_dropped"] == CFG["num_samples"]
    inv.assert_invariant(a, b, ["growth_log_ratio", "perturbation_growth_per_layer"], rtol=0.0)
    oa = _run_policy(model, tok, monkeypatch, "exclude_pos0", hook=(-1, 1000.0))
    ob = _run_policy(model, tok, monkeypatch, "exclude_pos0", hook=(-1, 2000.0))
    inv.assert_moves(oa, ob, "growth_log_ratio", 1e-2)
