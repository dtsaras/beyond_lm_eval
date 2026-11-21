"""dynamics_context_influence: the Jensen-Shannon kernel, the base-lag draw,
exact zero influence beyond the receptive field of an n-gram-like model, a
position-dependent but lag-independent model (where scoring every position
after the replacement would report a spurious decay), causality, sinks,
determinism and seeding, through the real ``evaluate()``.
"""
import math
import random
import types

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from blme.tasks.config_loader import resolve_task_config  # noqa: E402
from blme.tasks.dynamics.context_influence import (  # noqa: E402
    ContextInfluenceTask, base_lags, influence_summary, js_rows, lag_bins)

_WORDS = ("the river bank market story light water stone city night road garden "
          "window music paper , . of and to a in that it was").split()


def _docs(n, words=400, seed=0, vocab=_WORDS):
    rng = np.random.default_rng(seed)
    return [{"text": " ".join(rng.choice(vocab, size=words))} for _ in range(n)]


@pytest.fixture(scope="module")
def gpt2_tok():
    return transformers.AutoTokenizer.from_pretrained("gpt2")


def _run(model, tok, docs, **cfg):
    base = {"documents": "dataset", "num_samples": len(docs), "window": 128,
            "n_scored": 64, "replace_lo": 16, "n_replacements": 8, "n_bootstrap": 20}
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


# ── the kernel, the draw and the summary ─────────────────────────────────

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
    for K in (11, 16, 17, 1008):
        bins = lag_bins(K)
        lags = [k for lo, hi in bins for k in range(lo, hi + 1)]
        assert lags == list(range(1, K + 1))
    assert lag_bins(1008)[:5] == [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16)]
    assert lag_bins(1008)[-1] == (513, 1008)


def test_base_lags_are_log_uniform_and_stratified():
    k0_max, R = 945, 32
    rng = np.random.default_rng(0)
    draws = np.stack([base_lags(rng, R, k0_max) for _ in range(4000)])
    assert draws.min() >= 1 and draws.max() <= k0_max
    # One draw per stratum: column r always lies in stratum r of log(k0).
    L = math.log(k0_max + 1)
    lo = np.floor(np.exp(np.arange(R) * L / R))
    hi = np.floor(np.exp((np.arange(R) + 1) * L / R))
    assert np.all(draws >= lo) and np.all(draws <= np.minimum(hi, k0_max))
    # Marginally P(k0 = m) = log((m + 1) / m) / log(k0_max + 1).
    for m in (1, 2, 10, 300):
        want = math.log((m + 1) / m) / L
        assert abs(np.mean(draws == m) - want) < 0.1 * want + 2e-4, m


def test_summary_recovers_a_power_law():
    K = 1008
    bins = lag_bins(K)
    k = np.arange(1, K + 1, dtype=np.float64)
    I = 0.5 * k ** -1.25
    sums = [I[a - 1:b].sum() for a, b in bins]
    cnts = [b - a + 1 for a, b in bins]
    s = influence_summary(sums, cnts, bins, bytes_per_token=4.0)
    assert abs(s["influence_decay_exponent"] - 1.25) < 0.01
    assert math.isclose(s["influence_far_mass"], float(I[8:].sum()), rel_tol=1e-12)
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
    seen = np.array(res["_meta_n_terms_per_lag"]) > 0
    assert seen[:n].all() and seen[n:].sum() > 50
    assert np.all(I[:n] > 0), I[:n]
    assert np.all(I[n:][seen[n:]] == 0.0)
    assert res["influence_far_mass"] == 0.0
    assert res["_meta_pre_replacement_max_js"] == 0.0


class _MeanPoolLM(torch.nn.Module):
    """Logits at t read the mean of the embeddings of tokens 0..t. Replacing
    one token moves the state at t by (e_u - e_x) / (t + 1) for every t after
    it: the response depends on the position t and not on the lag."""

    def __init__(self, vocab=50257, d=16, ctx=1024, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.nn.Embedding(vocab, d)
        self.head = torch.nn.Linear(d, vocab, bias=False)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(vocab, d, generator=g))
            self.head.weight.copy_(torch.randn(vocab, d, generator=g) * 3.0)
        self.config = types.SimpleNamespace(max_position_embeddings=ctx)

    def forward(self, input_ids=None, output_hidden_states=False, **kw):
        e = self.emb(input_ids)
        h = e.cumsum(1) / torch.arange(1, e.shape[1] + 1, dtype=e.dtype)[None, :, None]
        return types.SimpleNamespace(logits=self.head(h),
                                     hidden_states=(e, e, e) if output_hidden_states else None)


def _all_positions_beta(model, tok, docs, W=1024, lo=16, hi=64, R=3, seed=0):
    """The replaced design: s in [lo, hi), every position after s scored."""
    rng = np.random.default_rng(seed)
    K = W - hi + 1
    bins = lag_bins(K)
    sums, cnts = np.zeros(len(bins)), np.zeros(len(bins))
    with torch.no_grad():
        for doc in docs:
            ids = tok(doc["text"], return_tensors="pt")["input_ids"][:, :W]
            lp = torch.log_softmax(model(input_ids=ids).logits[0].double(), -1)
            for s in rng.choice(np.arange(lo, hi), size=R, replace=False):
                x = ids.clone()
                x[0, s] = (int(x[0, s]) + 7919) % 50257
                lq = torch.log_softmax(model(input_ids=x).logits[0].double(), -1)
                d = js_rows(lp[s:s + K], lq[s:s + K]).numpy()
                for b, (a, z) in enumerate(bins):
                    sums[b] += d[a - 1:z].sum()
                    cnts[b] += z - a + 1
    return influence_summary(sums, cnts, bins, 1.0)["influence_decay_exponent"]


def test_a_position_dependent_lag_independent_response_has_no_decay(gpt2_tok):
    """Scoring only the end of the window holds the position fixed, so a
    response that depends on the position alone gives beta ~ 0; scoring every
    position after the replacement reads it as a decay with the lag."""
    model = _MeanPoolLM().eval()
    docs = _docs(6, words=1500, seed=4)
    res = _run(model, gpt2_tok, docs, window=1024, n_replacements=24)
    assert res["_meta_n_bins_empty"] == 0 and res["_meta_pre_replacement_max_js"] == 0.0
    assert abs(res["influence_decay_exponent"]) < 0.1, res["influence_decay_exponent"]
    old = _all_positions_beta(model, gpt2_tok, docs[:3])
    assert old > 0.7, old


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
    docs = _docs(6, seed=3, vocab=["the", "river", "the", "bank", "the", "city"])
    res = _run(_with_sinks(_tiny_gpt2(), the), gpt2_tok, docs, n_replacements=6)
    assert res["_meta_sink_n_massive_dropped"] > 0
    assert res["_meta_n_replacements_on_sinks"] > 0
    assert res["_meta_n_replacements"] + res["_meta_n_replacements_on_sinks"] == 6 * 6
    # Sink predictions are not scored: fewer terms than replacements x positions.
    assert sum(res["_meta_n_terms_per_lag"]) < res["_meta_n_replacements"] * 64


def test_window_and_lags(gpt2_tok):
    res = _run(_tiny_gpt2(), gpt2_tok, _docs(3), window=128)
    assert res["_meta_window_tokens"] == 128 and res["_meta_max_lag"] == 128 - 16
    assert res["_meta_scored_positions"] == [64, 127] and res["_meta_base_lag_range"] == [1, 49]
    assert len(res["_meta_influence_per_lag"]) == res["_meta_max_lag"]
    assert res["_meta_n_forwards"] == 3 * (1 + 8)
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
