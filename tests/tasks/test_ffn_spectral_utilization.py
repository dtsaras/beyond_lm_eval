"""geometry_ffn_spectral_utilization: Jha & Reagen's Table 1 formulas, the
streamed covariance, and the hooked site.

No code was released (arXiv:2510.00537), so the checks are the paper's
definitions written out directly, analytic spectra, an explicitly built centred
covariance, and a manual recomputation of the FFN post-activation from each
MLP's input through evaluate()."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))
import _family_d as fd  # noqa: E402

from blme.tasks.geometry.ffn_spectral import (  # noqa: E402
    CovarianceAccumulator, FFNSpectralUtilizationTask, layer_groups, n_nonzero,
    spectral_utilization,
)


def _direct(lam):
    """Table 1 term by term, with no shared code."""
    lam = sorted((float(x) for x in lam), reverse=True)
    D, S = len(lam), sum(lam)
    pr = S * S / sum(x * x for x in lam)
    er = math.exp(-sum(x / S * math.log(x / S) for x in lam if x > 0))
    sc = 2.0 / D * sum(sum(lam[:k]) / S - k / D for k in range(1, D + 1))
    prn, ern = (pr - 1) / (D - 1), (er - 1) / (D - 1)
    sui = 2 * prn * ern / (prn + ern)
    return {"hard_rank": pr, "soft_rank": er, "hard_rank_normalized": prn,
            "soft_rank_normalized": ern, "spectral_concentration": sc, "sui": sui,
            "edim": 1 + (D - 1) * sui}


def test_formulas_match_table1_written_out():
    rng = np.random.default_rng(0)
    for lam in (rng.exponential(size=40), rng.pareto(1.5, size=25) + 1e-3,
                np.r_[rng.random(10), np.zeros(5)]):
        got, ref = spectral_utilization(lam), _direct(lam)
        for k, v in ref.items():
            assert got[k] == pytest.approx(v, rel=1e-12, abs=1e-14), k


@pytest.mark.parametrize("D", [2, 7, 64])
def test_isotropic_spectrum(D):
    m = spectral_utilization(np.full(D, 3.7))
    assert m["hard_rank"] == pytest.approx(D, rel=1e-13)
    assert m["soft_rank"] == pytest.approx(D, rel=1e-13)
    assert m["sui"] == pytest.approx(1.0, rel=1e-13)
    assert m["edim"] == pytest.approx(D, rel=1e-13)
    assert abs(m["spectral_concentration"]) < 1e-13
    assert m["spectral_centroid"] == pytest.approx((D + 1) / 2, rel=1e-13)


@pytest.mark.parametrize("D", [2, 7, 64])
def test_rank_one_spectrum(D):
    m = spectral_utilization(np.r_[2.5, np.zeros(D - 1)])
    assert m["hard_rank"] == 1.0 and m["soft_rank"] == 1.0
    assert m["sui"] == 0.0 and m["edim"] == 1.0
    assert m["spectral_concentration"] == pytest.approx((D - 1) / D, rel=1e-13)


def test_flat_rank_r_spectrum():
    D, r = 50, 8
    m = spectral_utilization(np.r_[np.ones(r), np.zeros(D - r)])
    assert m["hard_rank"] == pytest.approx(r) and m["soft_rank"] == pytest.approx(r)
    assert m["sui"] == pytest.approx((r - 1) / (D - 1))
    assert m["spectral_concentration"] == pytest.approx(1 - r / D)


def test_sc_closed_form_and_scale_invariance():
    lam = np.random.default_rng(1).exponential(size=33)
    m = spectral_utilization(lam)
    D = lam.size
    assert m["spectral_concentration"] == pytest.approx(1 - (2 * m["spectral_centroid"] - 1) / D,
                                                        rel=1e-12)
    for c in (1e-200, 1e-30, 1e30, 1e150):
        mc = spectral_utilization(lam * c)
        for k in m:
            assert mc[k] == pytest.approx(m[k], rel=1e-12), (k, c)


def test_edim_is_the_harmonic_mean_of_the_raw_ranks():
    """1 + (D - 1) SUI = 1 + 2 (PR - 1)(eR - 1) / (PR + eR - 2): D cancels."""
    m = spectral_utilization(np.random.default_rng(5).exponential(size=31))
    a, b = m["hard_rank"] - 1, m["soft_rank"] - 1
    assert m["edim"] == pytest.approx(1 + 2 * a * b / (a + b), rel=1e-13)


def test_appended_zero_eigenvalues_leave_only_the_raw_forms():
    """Dead neurons: PR, eR, eDim and the centroid stay, everything divided by
    D moves."""
    lam = np.random.default_rng(2).exponential(size=20)
    a, b = spectral_utilization(lam), spectral_utilization(np.r_[lam, np.zeros(60)])
    for k in ("hard_rank", "soft_rank", "edim", "spectral_centroid"):
        assert b[k] == pytest.approx(a[k], rel=1e-13), k
    for k in ("hard_rank_normalized", "soft_rank_normalized", "sui", "spectral_concentration"):
        assert abs(b[k] - a[k]) > 0.05 * abs(a[k]), k


def _explicit(X):
    C = np.cov(X, rowvar=False, ddof=1)
    return np.linalg.eigvalsh(C)


def test_streamed_accumulator_equals_explicit_centred_covariance():
    """Uneven chunks, a mean 1e3 times the spread (cancellation), and the
    position-0 downdate against explicit removal of each chunk's first row."""
    rng = np.random.default_rng(3)
    D, sizes = 24, [7, 30, 1, 55, 12, 40]
    A = rng.normal(size=(D, D)) / math.sqrt(D)
    X = 1e3 + rng.normal(size=(sum(sizes), D)) @ A
    acc = CovarianceAccumulator(D)
    starts, s = [], 0
    for n in sizes:
        acc.add(torch.from_numpy(X[s:s + n]).float())
        starts.append(s)
        s += n
    Xf = X.astype(np.float32).astype(np.float64)
    out = acc.spectra()
    ref = _explicit(Xf)
    assert out["n"] == Xf.shape[0]
    assert np.max(np.abs(np.sort(out["all"]) - ref)) <= 1e-9 * ref.max()
    keep = np.setdiff1d(np.arange(Xf.shape[0]), starts)
    ref_ex = _explicit(Xf[keep])
    assert out["n_without_position0"] == keep.size
    assert np.max(np.abs(np.sort(out["without_position0"]) - ref_ex)) <= 1e-9 * ref_ex.max()
    mu = Xf.mean(axis=0)
    share = ((Xf[starts] - mu) ** 2).sum() / ((Xf - mu) ** 2).sum()
    assert out["position0_share"] == pytest.approx(share, rel=1e-9)


def test_rank_deficient_when_tokens_do_not_exceed_width():
    rng = np.random.default_rng(4)
    acc = CovarianceAccumulator(30)
    acc.add(torch.from_numpy(rng.normal(size=(10, 30))))
    lam = acc.spectra()["all"]
    assert n_nonzero(lam) == 9
    assert spectral_utilization(lam)["hard_rank"] <= 9 + 1e-9


def test_layer_groups_fit_the_budget():
    widths = [100] * 6
    per = 8 * (100 * 100 + 100 * 4) + 12 * 100 * 16
    reserve = 8 * 100 * 100
    assert layer_groups(widths, 2, 16, 1e12) == [list(range(6))]
    assert layer_groups(widths, 2, 16, 0) == [[j] for j in range(6)]
    g = layer_groups(widths, 2, 16, reserve + 2 * per)
    assert g == [[0, 1], [2, 3], [4, 5]]


def _evaluate(model, tok, cfg):
    return FFNSpectralUtilizationTask({"use_dataset": True, "max_length": 64, **cfg}).evaluate(
        model, tok, fd.dataset())


def test_grouped_passes_equal_one_pass():
    """A budget below one layer forces one group per layer; every value is
    bit-identical to the single-group run."""
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    one = _evaluate(model, tok, {"memory_budget_gb": 8.0})
    many = _evaluate(model, tok, {"memory_budget_gb": 0.0})
    assert one["_meta_n_layer_groups"] == 1
    assert many["_meta_n_layer_groups"] == len(one["ffn_sui_profile"])
    for k, v in one.items():
        if k not in ("_meta_n_layer_groups", "_meta_memory_budget_gb"):
            assert many[k] == v, k


def _manual_post_activations(model, tok, texts, max_length):
    """The FFN post-activation recomputed in float64 from each MLP's input:
    act(h W_fc + b) for GPT-2, SiLU(h W_g^T) * (h W_u^T) for Llama."""
    gpt2 = model.config.model_type == "gpt2"
    blocks = model.transformer.h if gpt2 else model.model.layers
    rows = {i: [] for i in range(len(blocks))}
    hooks = [b.mlp.register_forward_pre_hook(
        lambda m, a, i=i: rows[i].append(a[0][0].detach().double()))
        for i, b in enumerate(blocks)]
    try:
        with torch.no_grad():
            for t in texts:
                ids = tok(t, return_tensors="pt", truncation=True,
                          max_length=max_length)["input_ids"]
                model(input_ids=ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    out = []
    for i, b in enumerate(blocks):
        h = torch.cat(rows[i])
        if gpt2:
            z = h @ b.mlp.c_fc.weight.double() + b.mlp.c_fc.bias.double()
            assert model.config.activation_function == "gelu_new"
            a = 0.5 * z * (1 + torch.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z ** 3)))
        else:
            assert model.config.hidden_act == "silu"
            g = h @ b.mlp.gate_proj.weight.double().T
            u = h @ b.mlp.up_proj.weight.double().T
            a = g * torch.sigmoid(g) * u
        out.append(a.detach().numpy())
    return out


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_readout_is_the_manual_post_activation(arch):
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=3)
    tok = fd.tokenizer()
    res = _evaluate(model, tok, {"n_tokens": 10 ** 6})
    acts = _manual_post_activations(model, tok, fd.TEXTS, 64)
    assert res["_meta_n_tokens"] == acts[0].shape[0]
    assert res["_meta_ffn_width_profile"] == [a.shape[1] for a in acts]
    assert res["_meta_n_layers_token_mismatch"] == 0
    assert res["_meta_n_layers_measured"] == len(acts)
    starts = np.cumsum([0] + [len(tok(t)["input_ids"]) for t in fd.TEXTS])[:-1]
    for j, A in enumerate(acts):
        ref = spectral_utilization(_explicit(A))
        keep = np.setdiff1d(np.arange(A.shape[0]), starts)
        ref_ex = spectral_utilization(_explicit(A[keep]))
        for k in ("hard_rank", "soft_rank", "sui", "spectral_concentration", "edim"):
            assert res[f"ffn_{k}_profile"][j] == pytest.approx(ref[k], rel=1e-6), (j, k)
        for k in ("hard_rank", "soft_rank", "sui"):
            assert res[f"_meta_ffn_{k}_profile_without_position0"][j] == pytest.approx(
                ref_ex[k], rel=1e-6), (j, k)
    assert res["ffn_sui_median"] == pytest.approx(float(np.median(res["ffn_sui_profile"])))


def test_token_budget_is_pinned_and_independent_of_the_shared_corpus():
    """N is exact and the passages are the library corpus's, whatever dataset
    the run passes in."""
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    cfg = {"n_tokens": 1500, "max_length": 64}
    a = FFNSpectralUtilizationTask(cfg).evaluate(model, tok, fd.dataset(2))
    b = FFNSpectralUtilizationTask(cfg).evaluate(model, tok, None)
    assert a["_meta_n_tokens"] == b["_meta_n_tokens"] == 1500
    assert a["ffn_sui_profile"] == b["ffn_sui_profile"]


class _Routed(torch.nn.Module):
    """A routed block: no named down projection; its last projection is the gate."""

    def __init__(self, mlp, hidden):
        super().__init__()
        self.experts = torch.nn.ModuleList([mlp])
        self.gate = torch.nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        return self.experts[0](x)


def test_a_routed_block_is_skipped_not_read_at_its_gate():
    from blme.tasks.geometry.schatten import _mlp_down_projections
    model, tok = fd.tiny_llama(n_layers=3), fd.tokenizer()
    blk = model.model.layers[1]
    blk.mlp = _Routed(blk.mlp, model.config.hidden_size)
    assert dict(_mlp_down_projections(model))[1] is blk.mlp.gate
    res = _evaluate(model, tok, {})
    assert res["_meta_ffn_block_indices"] == [0, 2]
    assert res["_meta_n_layers_not_dense"] == 1
    assert res["_meta_ffn_width_profile"] == [model.config.intermediate_size] * 2
    for b in model.model.layers:
        b.mlp = _Routed(b.mlp, model.config.hidden_size)
    assert "error" in _evaluate(model, tok, {})
