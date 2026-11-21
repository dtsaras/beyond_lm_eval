"""geometry_readout_dual_spectrum: the conditioning ``_meta_`` keys (0.2.7).

A blind review (2026-11-21, F3) found the kappa keys exactly invariant only in
exact arithmetic: under a dense re-basing with cond(A) = 2e7, sum kappa moved
4% even in float64, and nothing in the result reported conditioning. The task
now reports, for each balanced covariance the eigensolver reads, the
eigenvalue range [smallest kept, largest], their ratio and the count of
eigenvalues the rank tolerance set to 0, and the limit 1 / (d eps) at which
that tolerance starts zeroing. Recording them changes no other number.
"""
import math

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from blme.tasks.config_loader import resolve_task_config  # noqa: E402
from blme.tasks.geometry.readout_dual_spectrum import (  # noqa: E402
    ReadoutDualSpectrumTask, _Moments, _Readout, conditioning_meta, dual_spectrum, spectrum_summary)

KEYS = ("_meta_cov_h_eig_range", "_meta_cov_h_condition", "_meta_cov_h_eigs_zeroed",
        "_meta_cov_mu_eig_range", "_meta_cov_mu_condition", "_meta_cov_mu_eigs_zeroed",
        "_meta_cov_condition_limit")


def _balanced(S_hh, S_mm):
    """The balanced pair dual_spectrum hands the eigensolver."""
    a, b = torch.diagonal(S_hh), torch.diagonal(S_mm)
    g = (b / a) ** 0.25
    G = torch.outer(g, g)
    return S_hh * G, S_mm / G


def _psd(d, eigs, seed):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return (Q * torch.as_tensor(eigs, dtype=torch.float64)) @ Q.T


def test_info_is_read_only_and_matches_the_balanced_spectra():
    d = 24
    S_hh = _psd(d, [0.0] + list(np.geomspace(1e-3, 10.0, d - 1)), seed=0)   # one exact null direction
    S_mm = _psd(d, np.geomspace(1e-2, 3.0, d), seed=1)
    info = {}
    with_info = dual_spectrum(S_hh, S_mm, info=info)
    assert np.array_equal(with_info, dual_spectrum(S_hh, S_mm))     # the spectrum is unchanged
    eps = float(torch.finfo(torch.float64).eps)
    for side, S in zip(("h", "mu"), _balanced(S_hh, S_mm)):
        e = torch.linalg.eigvalsh(S)
        tol = float(e.max()) * d * eps
        kept = e[e > tol]
        assert info[side]["n_zeroed"] == d - kept.numel()
        assert math.isclose(info[side]["eig_max"], float(e.max()), rel_tol=1e-12)
        assert math.isclose(info[side]["eig_min_kept"], float(kept.min()), rel_tol=1e-6)
    assert info["h"]["n_zeroed"] == 1 and info["mu"]["n_zeroed"] == 0


def test_meta_keys_from_a_summary():
    g = torch.Generator().manual_seed(3)
    h = torch.randn(400, 8, generator=g, dtype=torch.float64)
    ro = _Readout(torch.randn(50, 8, generator=g, dtype=torch.float64), None, None, None, torch.float64)
    acc = _Moments(8, 50, "cpu")
    _, lp, p, mu = ro.rows(h)
    acc.add_v(lp, p), acc.add_d(h, mu)
    s = spectrum_summary(acc, 8)
    meta = conditioning_meta(s["conditioning"], 8)
    assert set(meta) == set(KEYS)
    for side in ("h", "mu"):
        lo, hi = meta[f"_meta_cov_{side}_eig_range"]
        assert 0 < lo <= hi and meta[f"_meta_cov_{side}_condition"] == hi / lo
        assert meta[f"_meta_cov_{side}_eigs_zeroed"] == 0
    assert meta["_meta_cov_condition_limit"] == 1.0 / (8 * np.finfo(np.float64).eps)
    # Not computed: NaN ranges and no counts.
    empty = conditioning_meta(None, 8)
    assert all(math.isnan(x) for x in empty["_meta_cov_h_eig_range"])
    assert empty["_meta_cov_h_eigs_zeroed"] is None and math.isnan(empty["_meta_cov_mu_condition"])


def test_an_ill_conditioned_rebasing_shows_in_the_condition_number():
    """The regime F3 describes: a dense re-basing with cond(A) = 1e6 raises the
    reported covariance condition number by orders of magnitude."""
    d = 16
    S_hh = _psd(d, np.geomspace(0.5, 2.0, d), seed=4)
    S_mm = _psd(d, np.geomspace(0.5, 2.0, d), seed=5)
    g = torch.Generator().manual_seed(6)
    U, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    A = (U * torch.as_tensor(np.geomspace(1.0, 1e6, d))) @ V.T
    Ai = torch.linalg.inv(A)
    base, moved = {}, {}
    dual_spectrum(S_hh, S_mm, info=base)
    dual_spectrum(A @ S_hh @ A.T, Ai.T @ S_mm @ Ai, info=moved)
    cond = lambda i: i["eig_max"] / i["eig_min_kept"]  # noqa: E731
    assert cond(base["h"]) < 10
    assert cond(moved["h"]) > 1e6 and cond(moved["mu"]) > 1e6


def test_task_reports_the_conditioning(tmp_path):
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=128,
                                  tie_word_embeddings=False)
    model = transformers.GPT2LMHeadModel(cfg).eval().to(torch.float64)
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    words = "the river bank market story light water stone city night road garden".split()
    rng = np.random.default_rng(0)
    docs = [{"text": " ".join(rng.choice(words, size=40)).capitalize() + " ."} for _ in range(20)]
    conf = resolve_task_config("geometry_readout_dual_spectrum", {"n_rows": 300})
    res = ReadoutDualSpectrumTask(conf).evaluate(model, tok, docs)
    assert "error" not in res, res
    assert set(KEYS) <= set(res)
    d = res["_meta_d_model"]
    for side in ("h", "mu"):
        lo, hi = res[f"_meta_cov_{side}_eig_range"]
        assert 0 < lo <= hi and math.isclose(res[f"_meta_cov_{side}_condition"], hi / lo)
        assert isinstance(res[f"_meta_cov_{side}_eigs_zeroed"], int)
        assert 0 <= res[f"_meta_cov_{side}_eigs_zeroed"] < d
    # GPT-2's final LayerNorm puts the read state on a hyperplane: one null direction.
    assert res["_meta_cov_h_eigs_zeroed"] >= 1
    assert res["_meta_cov_condition_limit"] == 1.0 / (d * np.finfo(np.float64).eps)
