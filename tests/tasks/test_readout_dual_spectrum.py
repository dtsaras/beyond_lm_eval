"""geometry_readout_dual_spectrum: the readout identity, its gauge invariance,
the word-final rule, exact sink removal and the random-init control.

Everything that can be is checked through the real ``evaluate()`` on a
float64 model, where the invariances hold to rounding.
"""
import copy
import itertools
import math

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from blme.tasks._tokens import middle_depth_layers, sink_keep_masks  # noqa: E402
from blme.tasks.config_loader import resolve_task_config  # noqa: E402
from blme.tasks.geometry import readout_dual_spectrum as rds  # noqa: E402
from blme.tasks.geometry.readout_dual_spectrum import (  # noqa: E402
    ReadoutDualSpectrumTask, _Moments, _Readout, dual_spectrum, spectrum_summary,
    word_final_positions)

INVARIANT_KEYS = ("jeffreys_dispersion", "dual_coupling", "dual_coherence",
                  "dual_effective_rank", "dual_tail_exponent")

_WORDS = ("the river bank market story light water stone city night road garden "
          "window music paper , . of and to a in that it was").split()


def _texts(n, length=40, seed=0):
    rng = np.random.default_rng(seed)
    return [{"text": " ".join(rng.choice(_WORDS, size=length)).capitalize() + " ."}
            for _ in range(n)]


@pytest.fixture(scope="module")
def gpt2_tok():
    return transformers.AutoTokenizer.from_pretrained("gpt2")


def _tiny_gpt2(head_scale=25.0, seed=0, n_embd=32, dtype=torch.float64, arch="gpt2"):
    """A random 2-block model on the gpt2 vocabulary with an untied, scaled
    head, so that predictions are peaked and the softmax is non-linear."""
    torch.manual_seed(seed)
    if arch == "gpt2":
        cfg = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=n_embd, vocab_size=50257,
                                      n_positions=128, tie_word_embeddings=False)
        m = transformers.GPT2LMHeadModel(cfg).eval()
    else:
        cfg = transformers.LlamaConfig(vocab_size=50257, hidden_size=n_embd, intermediate_size=2 * n_embd,
                                       num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                                       max_position_embeddings=128, tie_word_embeddings=False,
                                       attn_implementation="eager")
        m = transformers.LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        m.lm_head.weight.copy_(torch.randn_like(m.lm_head.weight) * head_scale / math.sqrt(n_embd))
    return m.to(dtype)


def _run(model, tok, docs, **cfg):
    conf = resolve_task_config("geometry_readout_dual_spectrum", cfg)
    with torch.no_grad():
        res = ReadoutDualSpectrumTask(config=conf).evaluate(model, tok, docs)
    assert "error" not in res, res
    return res


def _rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# ── the word-final rule ──────────────────────────────────────────────────

def test_word_final_rule_on_hand_made_offsets():
    text = "ab cd中 e"
    # "ab" | " cd" | piece of 中 | rest of 中 (same span) | " e"
    offs = [(0, 2), (2, 5), (5, 6), (5, 6), (6, 8)]
    assert word_final_positions(text, offs) == [True, False, False, True, False]
    # Special token (empty span) and trimmed whitespace-only token: never word-final.
    text = "ab  cd"
    offs = [(0, 0), (0, 2), (4, 4), (4, 6)]
    assert word_final_positions(text, offs) == [False, True, False, False]


@pytest.mark.parametrize("name", ["gpt2", "EleutherAI/pythia-160m-deduped"])
def test_word_final_rule_drops_mid_character_positions(name):
    tok = transformers.AutoTokenizer.from_pretrained(name)
    text = "The 中文 cat′s  hat.\n\nNew para, here (it)."
    enc = tok(text, return_offsets_mapping=True)
    toks = tok.convert_ids_to_tokens(enc["input_ids"])
    wf = word_final_positions(text, enc["offset_mapping"])
    # The prefix ends at a word end exactly after these strings.
    ends = {text[:e] for (s, e), f in zip(enc["offset_mapping"], wf) if f}
    assert ends == {"The", "The 中文", "The 中文 cat′s",
                    "The 中文 cat′s  hat.", "The 中文 cat′s  hat.\n\nNew",
                    "The 中文 cat′s  hat.\n\nNew para,",
                    "The 中文 cat′s  hat.\n\nNew para, here"}, (toks, wf)


# ── the identity and the spectrum on synthetic readouts ──────────────────

def _synthetic(N=40, d=6, V=30, seed=0, cap=None, scale=None, bias=True):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(N, d, generator=g, dtype=torch.float64) * 1.7 + 0.3
    W = torch.randn(V, d, generator=g, dtype=torch.float64)
    b = torch.randn(V, generator=g, dtype=torch.float64) if bias else None
    return h, _Readout(W, b, scale, cap, torch.float64)


def _pairwise_half_jeffreys(lp):
    p = lp.exp()
    N = lp.shape[0]
    tot = 0.0
    for i, j in itertools.permutations(range(N), 2):
        tot += float((p[i] * (lp[i] - lp[j])).sum() + (p[j] * (lp[j] - lp[i])).sum())
    return 0.5 * tot / (N * (N - 1))


@pytest.mark.parametrize("cap,scale", [(None, None), (None, 0.7), (1.5, None)])
def test_jeffreys_dispersion_equals_half_mean_pairwise_jeffreys(cap, scale):
    h, ro = _synthetic(cap=cap, scale=scale)
    acc = _Moments(h.shape[1], ro.W.shape[0], "cpu")
    _, lp, p, mu = ro.rows(h)
    acc.add_v(lp, p)
    acc.add_d(h, mu)
    dj_v, dj_d = acc.jeffreys()
    direct = _pairwise_half_jeffreys(lp)
    assert _rel(dj_v, direct) < 1e-12
    if cap is None:
        # The d-space form is the same number for an affine head ...
        assert _rel(dj_d, direct) < 1e-12
    else:
        # ... and a first-order approximation under a softcap.
        assert _rel(dj_d, direct) > 1e-6


def test_moments_subtraction_is_exact():
    h, ro = _synthetic(N=30, seed=3)
    _, lp, p, mu = ro.rows(h)
    a, b = _Moments(6, 30, "cpu"), _Moments(6, 30, "cpu")
    a.add_v(lp, p), a.add_d(h, mu)
    a.add_v(lp[5:9], p[5:9], sign=-1.0), a.add_d(h[5:9], mu[5:9], sign=-1.0)
    keep = torch.ones(30, dtype=torch.bool)
    keep[5:9] = False
    b.add_v(lp[keep], p[keep]), b.add_d(h[keep], mu[keep])
    for x, y in zip(a.jeffreys(), b.jeffreys()):
        assert _rel(x, y) < 1e-12
    for x, y in zip(a.covariances(), b.covariances()):
        assert torch.allclose(x, y, rtol=0, atol=1e-12)


def _transformed_summary(h, ro, A, c):
    """The summary of (A h + c, W A^+, b - W A^+ c): every prediction unchanged."""
    Ap = torch.linalg.pinv(A)
    W2 = ro.W @ Ap
    b2 = (ro.b if ro.b is not None else 0) - W2 @ c
    ro2 = _Readout(W2, b2, ro.scale, ro.cap, torch.float64)
    h2 = h @ A.T + c
    out = []
    for r, x in ((ro, h), (ro2, h2)):
        acc = _Moments(x.shape[1], r.W.shape[0], "cpu")
        _, lp, p, mu = r.rows(x)
        acc.add_v(lp, p)
        acc.add_d(x, mu)
        out.append(spectrum_summary(acc, x.shape[1]))
    return out


def _gauge(d, D, seed, sigma):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=torch.float64))
    gains = torch.exp(sigma * torch.randn(d, generator=g, dtype=torch.float64))
    A = Q[:, :d] * gains                       # D x d, full column rank
    c = torch.randn(D, generator=g, dtype=torch.float64) * 3.0
    return A, c


@pytest.mark.parametrize("D", [128, 160])
def test_dual_spectrum_is_invariant_to_affine_rebasing_and_embedding(D):
    """Random rotation with log-normal gains (sigma 1.5) and an offset; D > d
    embeds the state isometrically up to the gains (zero-padding included)."""
    h, ro = _synthetic(N=3000, d=128, V=400, seed=1)
    A, c = _gauge(128, D, seed=2, sigma=1.5)
    base, moved = _transformed_summary(h, ro, A, c)
    k0, k1 = base["kappa"], moved["kappa"][:128]
    assert np.max(np.abs(k0[:100] - k1[:100])) / k0[0] < 1e-10
    for k in INVARIANT_KEYS:
        assert _rel(base[k], moved[k]) < 1e-10, (k, base[k], moved[k])
    # The Euclidean spectrum of h is not a property of the function.
    assert _rel(base["euclidean_participation_ratio"], moved["euclidean_participation_ratio"]) > 0.1


def test_coupling_bounds_the_dispersion():
    h, ro = _synthetic(N=500, d=12, V=200, seed=4)
    acc = _Moments(12, 200, "cpu")
    _, lp, p, mu = ro.rows(h)
    acc.add_v(lp, p), acc.add_d(h, mu)
    s = spectrum_summary(acc, 12)
    assert 0 < s["dual_coherence"] <= 1.0
    # A linear link (tiny logits) attains the bound.
    acc2 = _Moments(12, 200, "cpu")
    _, lp, p, mu = ro.rows(h * 1e-4)
    acc2.add_v(lp, p), acc2.add_d(h * 1e-4, mu)
    assert spectrum_summary(acc2, 12)["dual_coherence"] > 1 - 1e-3


# ── through evaluate() ───────────────────────────────────────────────────

class _Affine(torch.nn.Module):
    def __init__(self, norm, A, c):
        super().__init__()
        self.norm = norm
        self.lin = torch.nn.Linear(A.shape[1], A.shape[0], bias=True, dtype=A.dtype)
        with torch.no_grad():
            self.lin.weight.copy_(A)
            self.lin.bias.copy_(c)

    def forward(self, x):
        return self.lin(self.norm(x))


def _regauge(model, A, c):
    """The same predictive function with the read state A h + c."""
    m = copy.deepcopy(model)
    W = m.lm_head.weight.detach()
    b = m.lm_head.bias.detach() if m.lm_head.bias is not None else torch.zeros(W.shape[0], dtype=W.dtype)
    W2 = W @ torch.linalg.pinv(A)
    head = torch.nn.Linear(A.shape[0], W.shape[0], bias=True, dtype=W.dtype)
    with torch.no_grad():
        head.weight.copy_(W2)
        head.bias.copy_(b - W2 @ c)
    if hasattr(m, "transformer"):
        m.transformer.ln_f = _Affine(m.transformer.ln_f, A, c)
    else:
        m.model.norm = _Affine(m.model.norm, A, c)
    m.lm_head = head
    return m.eval()


@pytest.mark.parametrize("arch,D", [("gpt2", 32), ("llama", 32), ("llama", 48)])
def test_size_invariant_entry(gpt2_tok, arch, D):
    """readout_affine_gauge (D = d: rotation, log-normal gains, offset, bias),
    zero_pad and isometry (D > d), at a fixed number of rows. GPT-2 reshapes
    the final-norm output to the embedding width, so the wider read state is
    checked on Llama."""
    model = _tiny_gpt2(arch=arch)
    docs = _texts(30)
    A, c = _gauge(32, D, seed=5, sigma=1.0)
    base = _run(model, gpt2_tok, docs, n_rows=600)
    moved = _run(_regauge(model, A, c), gpt2_tok, docs, n_rows=600)
    assert base["_meta_n_rows"] == moved["_meta_n_rows"] == 600
    assert moved["_meta_head_bias"] and moved["_meta_d_model"] == D
    for k in ReadoutDualSpectrumTask.SIZE_INVARIANT:
        assert _rel(base[k], moved[k]) < 1e-10, (k, base[k], moved[k])
    k0, k1 = np.array(base["_meta_kappa_top"]), np.array(moved["_meta_kappa_top"])
    assert np.max(np.abs(k0 - k1)) / k0[0] < 1e-10
    if D > 32:
        assert _rel(base["dual_effective_rank_frac"], moved["dual_effective_rank_frac"]) > 0.2


def _with_sinks(model, positions_of):
    """Hook block 0 so that rows selected by ``positions_of(input_ids)`` get
    a norm far above 10x the median at every middle depth."""
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        ids = hook.ids
        sel = positions_of(ids)
        h = h.clone()
        h[0, sel] = h[0, sel] + 500.0
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def pre(module, args, kwargs):
        hook.ids = kwargs.get("input_ids", args[0] if args else None)
    model.transformer.h[0].register_forward_hook(hook)
    model.register_forward_pre_hook(pre, with_kwargs=True)
    return model


@pytest.mark.parametrize("sinks", [False, True])
def test_evaluate_matches_a_brute_force_pairwise_computation(gpt2_tok, sinks):
    """Every word-final row of the passages read, position 0 and sinks out:
    D_J equals 1/2 the mean Jeffreys divergence over distinct pairs. With
    ``sinks`` every " the" carries a massive activation and must be dropped."""
    model = _tiny_gpt2(seed=1)
    if sinks:
        the = gpt2_tok(" the")["input_ids"][0]
        model = _with_sinks(model, lambda ids: (ids[0] == the).nonzero().flatten())
    docs = _texts(6, length=25, seed=2)
    res = _run(model, gpt2_tok, docs, n_rows=10 ** 6)
    assert (res["_meta_sink_rows_removed"] > 0) == sinks
    lps, mids = [], []
    with torch.no_grad():
        for doc in docs:
            enc = gpt2_tok(doc["text"], return_offsets_mapping=True, return_tensors="pt")
            off = enc.pop("offset_mapping")[0].tolist()
            out = model(**enc, output_hidden_states=True)
            wf = word_final_positions(doc["text"], off)
            blocks = out.hidden_states[1:]
            mids.append([blocks[k][0] for k in middle_depth_layers(dict.fromkeys(range(len(blocks))))])
            lp = torch.log_softmax(out.logits[0].double(), -1)
            lps.append((lp, wf))
    masks, _ = sink_keep_masks([[m[k] for m in mids] for k in range(len(mids[0]))])
    rows = torch.cat([lp[[t for t in range(1, lp.shape[0]) if wf[t] and bool(mk[t])]]
                      for (lp, wf), mk in zip(lps, masks)])
    assert rows.shape[0] == res["_meta_n_rows"]
    assert _rel(res["jeffreys_dispersion"], _pairwise_half_jeffreys(rows)) < 1e-10


def test_sink_rows_are_removed_exactly_buffered_or_refetched(gpt2_tok, monkeypatch):
    the = gpt2_tok(" the")["input_ids"][0]
    model = _with_sinks(_tiny_gpt2(seed=2), lambda ids: (ids[0] == the).nonzero().flatten())
    docs = _texts(10, length=30, seed=4)
    buffered = _run(model, gpt2_tok, docs, n_rows=10 ** 6)
    assert buffered["_meta_sink_rows_removed"] > 0
    assert buffered["_meta_sink_rows_refetched"] == 0
    # No candidate kept: every sink row is recomputed by a second forward.
    monkeypatch.setattr(rds, "SINK_CANDIDATE_FACTOR", 1e30)
    refetched = _run(model, gpt2_tok, docs, n_rows=10 ** 6)
    assert refetched["_meta_sink_rows_refetched"] == refetched["_meta_sink_rows_removed"] > 0
    for k in ReadoutDualSpectrumTask.HEADLINE_KEYS:
        assert _rel(buffered[k], refetched[k]) < 1e-10, k
    # And both equal a run on the passages with the sink rows never added.
    assert buffered["_meta_jeffreys_dispersion_incl_sinks"] != buffered["jeffreys_dispersion"]


def test_random_init_control(gpt2_tok):
    """Untrained head (init scale): tiny dispersion, coherence ~ 1."""
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=50257, n_positions=128)
    model = transformers.GPT2LMHeadModel(cfg).eval()
    res = _run(model, gpt2_tok, _texts(40), n_rows=800)
    assert res["jeffreys_dispersion"] < 0.1
    assert res["dual_coherence"] > 0.99
    assert all(math.isfinite(res[k]) for k in ReadoutDualSpectrumTask.HEADLINE_KEYS)


def test_soft_capped_head_reports_the_exact_logit_level(gpt2_tok):
    torch.manual_seed(0)
    cfg = transformers.Gemma2Config(vocab_size=50257, hidden_size=32, intermediate_size=64,
                                    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                    head_dim=16, max_position_embeddings=128, final_logit_softcapping=3.0,
                                    attn_implementation="eager")
    model = transformers.Gemma2ForCausalLM(cfg).eval()
    with torch.no_grad():
        model.lm_head.weight.mul_(40.0)
    model = model.double()
    docs = _texts(6, length=25, seed=7)
    res = _run(model, gpt2_tok, docs, n_rows=10 ** 6)
    assert res["_meta_d_space_exact"] is False and res["_meta_head_form"] == "softcap"
    assert res["_meta_head_guard_max_rel"] < 1e-10
    rows = []
    with torch.no_grad():
        for doc in docs:
            enc = gpt2_tok(doc["text"], return_offsets_mapping=True, return_tensors="pt")
            off = enc.pop("offset_mapping")[0].tolist()
            wf = word_final_positions(doc["text"], off)
            lp = torch.log_softmax(model(**enc).logits[0].double(), -1)
            rows.append(lp[[t for t in range(1, lp.shape[0]) if wf[t]]])
    rows = torch.cat(rows)
    assert rows.shape[0] == res["_meta_n_rows"]
    assert _rel(res["jeffreys_dispersion"], _pairwise_half_jeffreys(rows)) < 1e-10
    assert _rel(res["jeffreys_dispersion"], res["_meta_jeffreys_dispersion_dspace"]) > 1e-6


def test_head_that_does_not_reproduce_the_logits_fails_closed(gpt2_tok):
    model = _tiny_gpt2(dtype=torch.float32)
    model.lm_head.register_forward_hook(lambda m, a, out: out * 1.5)
    conf = resolve_task_config("geometry_readout_dual_spectrum", {"n_rows": 100})
    res = ReadoutDualSpectrumTask(config=conf).evaluate(model, gpt2_tok, _texts(5))
    assert "error" in res and "reproduce" in res["error"]


def test_rows_are_pinned_at_points_per_dim_times_d(gpt2_tok):
    model = _tiny_gpt2(dtype=torch.float32)
    res = _run(model, gpt2_tok, _texts(80), points_per_dim=8)
    assert res["_meta_n_rows"] == res["_meta_n_rows_target"] == 8 * 32
    assert res["_meta_points_per_dim"] == 8.0 and res["_meta_n_target_met"] is True
    assert res["_meta_n_rows_all_positions"] > res["_meta_n_rows"]


def test_compute_dtype_float32_on_a_bf16_model_restores_it(gpt2_tok):
    model = _tiny_gpt2(dtype=torch.bfloat16)
    native = _run(model, gpt2_tok, _texts(10), n_rows=200)
    assert native["_meta_compute_dtype"] == "bfloat16" and native["_meta_promoted_to_fp32"] is False
    fp32 = _run(model, gpt2_tok, _texts(10), n_rows=200, compute_dtype="float32")
    assert fp32["_meta_compute_dtype"] == "float32" and fp32["_meta_promoted_to_fp32"] is True
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())


class _NoOffsets:
    """A tokenizer that cannot give character offsets (a slow tokenizer)."""

    def __init__(self, tok):
        self._tok = tok
        self.eos_token_id, self.sep_token_id = tok.eos_token_id, None

    def __call__(self, text, **kw):
        if kw.pop("return_offsets_mapping", False):
            raise NotImplementedError("return_offset_mapping is not available")
        return self._tok(text, **kw)


def test_a_tokenizer_without_offsets_gives_nan_headlines_and_a_reason(gpt2_tok):
    res = _run(_tiny_gpt2(dtype=torch.float32), _NoOffsets(gpt2_tok), _texts(8), n_rows=150)
    assert all(math.isnan(res[k]) for k in ReadoutDualSpectrumTask.HEADLINE_KEYS)
    assert res["_meta_word_final_available"] is False and "offsets" in res["_meta_word_final_reason"]
    assert math.isfinite(res["_meta_jeffreys_dispersion_all_positions"])
    assert res["_meta_n_rows_all_positions"] == 150
