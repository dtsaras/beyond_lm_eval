"""dynamics_context_influence: the Jensen-Shannon kernel, exact zero
influence beyond the receptive field of an n-gram-like model, causality,
sinks, determinism and seeding, through the real ``evaluate()``.
"""
import math
import random

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from blme.tasks.config_loader import resolve_task_config  # noqa: E402
from blme.tasks.dynamics.context_influence import (  # noqa: E402
    ContextInfluenceTask, influence_summary, js_rows, lag_bins)

_WORDS = ("the river bank market story light water stone city night road garden "
          "window music paper , . of and to a in that it was").split()


def _docs(n, words=400, seed=0):
    rng = np.random.default_rng(seed)
    return [{"text": " ".join(rng.choice(_WORDS, size=words))} for _ in range(n)]


@pytest.fixture(scope="module")
def gpt2_tok():
    return transformers.AutoTokenizer.from_pretrained("gpt2")


def _run(model, tok, docs, **cfg):
    base = {"documents": "dataset", "num_samples": len(docs), "window": 128,
            "replace_lo": 16, "replace_hi": 32, "n_replacements": 2, "n_bootstrap": 20}
    conf = resolve_task_config("dynamics_context_influence", {**base, **cfg})
    with torch.no_grad():
        res = ContextInfluenceTask(config=conf).evaluate(model, tok, docs)
    assert "error" not in res, res
    return res


def _canon(x):
    if isinstance(x, dict):
        return {k: _canon(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, float):
        return x.hex()
    return x


# ── the kernel ───────────────────────────────────────────────────────────

def test_js_is_exactly_zero_for_identical_rows_and_log2_for_disjoint_ones():
    g = torch.Generator().manual_seed(0)
    lp = torch.log_softmax(torch.randn(5, 300, generator=g, dtype=torch.float64) * 4, -1)
    assert torch.equal(js_rows(lp, lp), torch.zeros(5, dtype=torch.float64))
    # Disjoint supports (probability exactly 0 off the support).
    a = torch.full((2, 6), -2000.0, dtype=torch.float64)
    b = a.clone()
    a[:, :3], b[:, 3:] = 0.0, 0.0
    a, b = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    assert torch.allclose(js_rows(a, b), torch.full((2,), math.log(2.0), dtype=torch.float64),
                          rtol=0, atol=1e-15)


def test_js_is_bounded_by_log2_and_symmetric():
    g = torch.Generator().manual_seed(1)
    for scale in (0.1, 3.0, 30.0, 300.0):
        lp = torch.log_softmax(torch.randn(64, 500, generator=g, dtype=torch.float64) * scale, -1)
        lq = torch.log_softmax(torch.randn(64, 500, generator=g, dtype=torch.float64) * scale, -1)
        d = js_rows(lp, lq)
        assert bool((d >= 0).all()) and bool((d <= math.log(2.0) + 1e-15).all())
        assert torch.allclose(d, js_rows(lq, lp), rtol=1e-12, atol=1e-15)
        # Against the textbook form on the same rows, where that form is
        # itself finite (it divides by an underflowed m at large scales).
        p, q = lp.exp(), lq.exp()
        m = 0.5 * (p + q)
        ref = 0.5 * (p * (p / m).log()).nansum(-1) + 0.5 * (q * (q / m).log()).nansum(-1)
        ok = torch.isfinite(ref)
        assert ok.any()
        assert torch.allclose(d[ok], ref[ok], rtol=1e-9, atol=1e-14)


def test_lag_bins_cover_every_lag_once():
    for K in (11, 16, 17, 961):
        bins = lag_bins(K)
        lags = [k for lo, hi in bins for k in range(lo, hi + 1)]
        assert lags == list(range(1, K + 1))
    assert lag_bins(961)[:5] == [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16)]
    assert lag_bins(961)[-1] == (513, 961)


def test_summary_recovers_a_power_law():
    K = 961
    k = np.arange(1, K + 1, dtype=np.float64)
    s = influence_summary(0.5 * k ** -1.25, bytes_per_token=4.0)
    assert abs(s["influence_decay_exponent"] - 1.25) < 0.01
    assert math.isclose(s["influence_far_mass"], float(np.sum(0.5 * k[8:] ** -1.25)), rel_tol=1e-12)
    assert math.isclose(s["influence_horizon_bytes"], 4.0 * s["horizon_tokens"], rel_tol=1e-12)


# ── through evaluate() ───────────────────────────────────────────────────

def _local_model(sliding_window, n_layers, seed=0):
    """A Mistral whose attention sees only the last ``sliding_window``
    positions: an n-gram model with n = n_layers * (sliding_window - 1) + 1."""
    torch.manual_seed(seed)
    cfg = transformers.MistralConfig(vocab_size=50257, hidden_size=32, intermediate_size=64,
                                     num_hidden_layers=n_layers, num_attention_heads=2,
                                     num_key_value_heads=2, max_position_embeddings=256,
                                     sliding_window=sliding_window, attn_implementation="eager")
    m = transformers.MistralForCausalLM(cfg).eval()
    with torch.no_grad():
        m.lm_head.weight.mul_(30.0)
    return m


@pytest.mark.parametrize("sliding_window,n_layers", [(4, 1), (4, 2), (6, 1)])
def test_zero_influence_beyond_the_receptive_field(gpt2_tok, sliding_window, n_layers):
    n = n_layers * (sliding_window - 1) + 1
    res = _run(_local_model(sliding_window, n_layers), gpt2_tok, _docs(4))
    I = np.array(res["_meta_influence_per_lag"])
    assert np.all(I[:n] > 0), I[:n]
    assert np.all(I[n:] == 0.0), I[n:n + 5]
    assert res["influence_far_mass"] == 0.0 if n < 9 else res["influence_far_mass"] > 0
    assert res["_meta_pre_replacement_max_js"] == 0.0


def _tiny_gpt2(seed=0):
    torch.manual_seed(seed)
    cfg = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=256)
    m = transformers.GPT2LMHeadModel(cfg).eval()
    with torch.no_grad():
        m.transformer.wte.weight.mul_(20.0)
    return m


def test_runs_are_bit_identical_and_the_seed_moves_the_draws(gpt2_tok):
    model, docs = _tiny_gpt2(), _docs(4, seed=1)
    a = _run(model, gpt2_tok, docs)
    b = _run(model, gpt2_tok, docs)
    assert _canon(a) == _canon(b)
    # The study's seed facet adds k to every `seed` key (offset_seeds).
    c = _run(model, gpt2_tok, docs, seed=43)
    assert c["_meta_draw_digest"] != a["_meta_draw_digest"] and c["_meta_seed"] == 43
    moved = [k for k in ContextInfluenceTask.HEADLINE_KEYS if a[k] != c[k]]
    assert moved, "seed 43 moved no headline key"


def test_default_seed_is_pinned():
    assert resolve_task_config("dynamics_context_influence")["seed"] == 42


def test_the_global_rngs_are_left_alone(gpt2_tok):
    model, docs = _tiny_gpt2(), _docs(3, seed=2)
    random.seed(1234), np.random.seed(1234), torch.manual_seed(1234)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    _run(model, gpt2_tok, docs)
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    assert after[0] == before[0]
    assert all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
               for x, y in zip(after[1], before[1]))
    assert torch.equal(after[2], before[2])


def _with_sinks(model, token_id):
    """Every ``token_id`` row gets a massive activation after block 0."""
    def pre(module, args, kwargs):
        hook.ids = kwargs.get("input_ids")

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        sel = (hook.ids[0] == token_id).nonzero().flatten()
        h = h.clone()
        h[0, sel] = h[0, sel] + 500.0
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    model.register_forward_pre_hook(pre, with_kwargs=True)
    model.transformer.h[0].register_forward_hook(hook)
    return model


def test_sinks_are_never_replaced_and_never_scored(gpt2_tok):
    the = gpt2_tok(" the")["input_ids"][0]
    docs = _docs(6, seed=3)
    res = _run(_with_sinks(_tiny_gpt2(), the), gpt2_tok, docs, n_replacements=6)
    assert res["_meta_sink_n_massive_dropped"] > 0
    assert res["_meta_n_replacements_on_sinks"] > 0
    assert res["_meta_n_replacements"] + res["_meta_n_replacements_on_sinks"] == 6 * 6


def test_window_and_lags(gpt2_tok):
    res = _run(_tiny_gpt2(), gpt2_tok, _docs(3), window=128, replace_hi=32)
    assert res["_meta_window_tokens"] == 128 and res["_meta_max_lag"] == 128 - 32 + 1
    assert len(res["_meta_influence_per_lag"]) == res["_meta_max_lag"]
    bpt = res["_meta_bytes_per_token"]
    assert res["_meta_lag_bin_centers_bytes"] == [c * bpt for c in res["_meta_lag_bin_centers_tokens"]]
    # The configured window is capped at the model's context and says so.
    capped = _run(_tiny_gpt2(), gpt2_tok, _docs(3), window=512)
    assert capped["_meta_window_tokens"] == 256 and capped["_meta_window_is_configured"] is False


def test_fp32_compute_on_a_bf16_model_restores_it(gpt2_tok):
    model = _tiny_gpt2().to(torch.bfloat16)
    res = _run(model, gpt2_tok, _docs(2), n_replacements=1)
    assert res["_meta_compute_dtype"] == "float32" and res["_meta_promoted_to_fp32"] is True
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    native = _run(model, gpt2_tok, _docs(2), n_replacements=1, compute_dtype="model")
    assert native["_meta_compute_dtype"] == "bfloat16"
