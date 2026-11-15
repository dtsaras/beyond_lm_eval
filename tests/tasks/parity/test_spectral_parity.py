"""geometry_spectral's fixed-tail Hill proxy (``_meta_avg_alpha``) vs WeightWatcher.

Scope: the headline is WeightWatcher's own alpha (``avg_alpha_official``,
``median_alpha_official``, ``avg_alpha_hat``), pinned against the package in
tests/tasks/test_comprehensive_parity.py::test_geometry_spectral and
tests/tasks/parity/test_spectral_sharpness_values.py. This file covers only the
Hill estimator over a FIXED 20% tail of the SINGULAR values, which the task
still emits under ``_meta_`` (``_meta_avg_alpha`` and its summaries).

Paper: Martin & Mahoney, Heavy-Tailed Self-Regularization (2019/2021);
       Martin, Peng & Mahoney, Nature Communications 12:4122 (2021).
       Official code: CalculatedContent/WeightWatcher (pip `weightwatcher`).

WeightWatcher uses the same Hill/MLE formula but scans every candidate xmin,
keeps the KS-optimal one, and fits the EIGENVALUES lambda = sigma^2. So the
proxy differs from WW alpha by (i) a ~factor-2 sigma-vs-lambda scale
[alpha_sigma ~ 2*alpha_lambda - 1] and (ii) fixed-tail vs KS-xmin: it is not
value-comparable to a published WW alpha, which is why it is _meta_.

What is pinned for the proxy:

  (a) known-exponent recovery: singular values with an ESD tail exponent a
      -> Hill alpha ~= a (statistical recovery, loose tolerance).
  (b) EXACT PARITY: the Hill kernel == an INDEPENDENT transcription of the
      Hill estimator, to < 1e-9, over many toy spectra.
  (c) the proxy-vs-WeightWatcher delta on the SAME toy matrices is RECORDED
      (not forced to 0); the fixture holds the measured WW alpha, the
      sigma-map, and the residual gap.
  (d) anchor: a random (Marchenko-Pastur) matrix -> LARGE alpha in both; the
      proxy lands in the sigma-scale [3,11] band.

(a)-(d) run on a transcription of the kernel, `_blme_hill_alpha`;
test_real_task_matches_transcribed_kernel ties it to evaluate()'s
``_meta_avg_alpha``. WeightWatcher's numbers in (c) and (d) were produced once
and frozen into the fixture.
"""

import json
from pathlib import Path

import numpy as np
import pytest


FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures/reference_parity/parity/spectral.json"
)


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The Hill proxy behind ``_meta_avg_alpha``, transcribed from
# src/blme/tasks/geometry/spectral.py (tied to evaluate() by
# test_real_task_matches_transcribed_kernel). S = singular values, descending.
# ---------------------------------------------------------------------------
def _blme_hill_alpha(S_np, tail_fraction=0.2):
    S_np = np.sort(np.asarray(S_np, dtype=np.float64))[::-1]  # svdvals: descending
    k = max(2, int(tail_fraction * len(S_np)))
    top_k = S_np[:k]
    if k > 0 and top_k[-1] > 1e-6:
        x_min = top_k[-1]
        log_sum = np.sum(np.log(top_k / x_min))
        if log_sum > 0:
            alpha = float(np.clip(1 + k / log_sum, 0, 20))
        else:
            alpha = 0.0
    else:
        alpha = 0.0
    return alpha


# ---------------------------------------------------------------------------
# INDEPENDENT Hill estimator (Hill 1975), transcribed from the textbook
# definition (order statistics x_(1) >= ... >= x_(k+1)):
#     gamma_hat = (1/k) * sum ln(x_(i)/x_(k+1)),  alpha = 1 + 1/gamma_hat.
# BLME divides by len(top_k) and uses top_k[-1] as the threshold, so we match
# its exact arithmetic (same mathematics, independent transcription).
# ---------------------------------------------------------------------------
def _independent_hill_alpha(S_np, tail_fraction=0.2):
    x = np.sort(np.asarray(S_np, dtype=np.float64))[::-1]  # descending
    n = len(x)
    kk = max(2, int(tail_fraction * n))
    tail = x[:kk]
    thresh = tail[-1]
    if thresh <= 1e-6:
        return 0.0
    logs = np.log(tail / thresh)
    denom = float(logs.sum())
    if denom <= 0:
        return 0.0
    return float(min(max(1.0 + kk / denom, 0.0), 20.0))


def _sample_powerlaw(a, n, xmin=1.0, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    return xmin * (1.0 - u) ** (-1.0 / (a - 1.0))


# ===========================================================================
# (a) known-exponent recovery: BLME Hill alpha ~= true tail exponent a.
# ===========================================================================
def test_known_exponent_recovery():
    fx = _fixture()["check_a_known_exponent_recovery"]
    for row in fx["rows"]:
        a_true = row["a_true"]
        svals = _sample_powerlaw(a_true, n=5000, xmin=1.0, seed=int(a_true * 10))
        alpha_hat = _blme_hill_alpha(svals, tail_fraction=fx["tail_fraction"])
        # bit-reproducible against the recorded value
        assert alpha_hat == pytest.approx(row["blme_hill_alpha_tail0.05"], abs=1e-6)
        # statistical recovery of the true exponent
        assert abs(alpha_hat - a_true) < 0.3
    assert fx["pass"] is True


# ===========================================================================
# (b) EXACT PARITY: BLME Hill == independent Hill, < 1e-9  (the exact verdict).
# ===========================================================================
def test_blme_hill_equals_independent_hill_exact():
    rng = np.random.default_rng(11)
    max_diff = 0.0
    for i in range(6):
        S = np.abs(rng.standard_normal(200)) + 0.01 * (i + 1)
        for tf in (0.1, 0.2, 0.35):
            a_blme = _blme_hill_alpha(S, tail_fraction=tf)
            a_indep = _independent_hill_alpha(S, tail_fraction=tf)
            max_diff = max(max_diff, abs(a_blme - a_indep))

    assert max_diff < 1e-9

    fx = _fixture()["check_b_independent_hill"]
    assert max_diff == pytest.approx(fx["max_abs_diff"], abs=1e-12)
    assert fx["max_abs_diff"] < fx["tol"]
    assert fx["pass"] is True
    # spot-check a recorded row reproduces bit-for-bit
    row = fx["sample_rows"][0]
    S0 = np.abs(np.random.default_rng(11).standard_normal(200)) + 0.01
    a0 = _blme_hill_alpha(S0, tail_fraction=row["tail_fraction"])
    assert a0 == pytest.approx(row["blme"], abs=1e-9)


# ===========================================================================
# (c) BLME-vs-WeightWatcher delta is RECORDED (proxy, not forced-equal).
# ===========================================================================
def test_blme_vs_weightwatcher_delta_recorded():
    fx = _fixture()["check_c_blme_vs_ww_delta"]
    assert fx["verdict"].startswith("PROXY")

    for row in fx["rows"]:
        # sigma-map consistency: ww_mapped == 2*ww_alpha - 1
        assert row["ww_mapped_2a_minus_1"] == pytest.approx(
            2.0 * row["ww_alpha_lambda_ksxmin"] - 1.0, abs=1e-3
        )
        # delta_mapped == blme - ww_mapped
        assert row["delta_blme_vs_ww_mapped"] == pytest.approx(
            row["blme_alpha_sigma_tail0.2"] - row["ww_mapped_2a_minus_1"], abs=1e-3
        )

    # The heavy-tailed layer, once the sigma-vs-lambda factor-2 is removed, is
    # within ~0.1 of WW's KS-xmin alpha -> BLME tracks WW closely there.
    heavy = next(r for r in fx["rows"] if r["matrix"] == "heavy_tailed_linear")
    assert abs(heavy["delta_blme_vs_ww_mapped"]) < 0.2
    # BLME and WW are genuinely different estimators: the raw (unmapped) delta
    # is non-trivial, confirming this is a PROXY not an exact reproduction.
    assert abs(heavy["delta_blme_vs_ww_raw"]) > 0.1


# ===========================================================================
# (d) anchor: random Marchenko-Pastur matrix -> LARGE alpha in BLME and WW.
# ===========================================================================
def test_marchenko_pastur_anchor():
    fx = _fixture()["check_d_anchor_marchenko_pastur"]
    rng = np.random.default_rng(5)
    W = rng.standard_normal((400, 300)) / np.sqrt(300)
    svals = np.linalg.svd(W, compute_uv=False)
    mp_alpha = _blme_hill_alpha(svals, tail_fraction=0.2)

    # reproduces the recorded BLME value
    assert mp_alpha == pytest.approx(fx["blme_alpha_sigma_tail0.2"], abs=1e-4)
    # inside the sigma-scale HTSR band
    lo, hi = fx["expected_band_sigma_scale"]
    assert lo <= mp_alpha <= hi
    # BLME and WW agree it is a high-alpha random layer (same order of magnitude)
    assert abs(mp_alpha - fx["ww_alpha_lambda_ksxmin"]) < 2.0
    assert fx["pass"] is True


# ===========================================================================
# fixture-level verdict pin.
# ===========================================================================
def test_fixture_records_proxy_verdict():
    """The PROXY verdict is the Hill proxy's (``_meta_avg_alpha``), not the
    headline's: ``avg_alpha_official`` is WeightWatcher's alpha itself."""
    from blme.tasks.geometry.spectral import WeightSpectralTask
    assert "avg_alpha_official" in WeightSpectralTask.HEADLINE_KEYS
    assert not [k for k in WeightSpectralTask.HEADLINE_KEYS if k.endswith("avg_alpha")]
    fx = _fixture()
    assert fx["task"] == "geometry_spectral"
    assert fx["label"].startswith("REFINED-ADAPTATION")
    assert fx["verdict"] == "PROXY"
    assert fx["reference_impl"]["library"] == "weightwatcher"
    assert fx["reference_impl"]["ww_version"] == "0.7.7"
    assert fx["check_a_known_exponent_recovery"]["pass"] is True
    assert fx["check_b_independent_hill"]["pass"] is True
    assert fx["check_d_anchor_marchenko_pastur"]["pass"] is True


def test_real_task_matches_transcribed_kernel():
    """Anti-drift guard (campaign-4 B4): every other test in this file runs
    on the TRANSCRIBED kernel above, so a formula change in spectral.py
    would leave them green. Drive the REAL WeightSpectralTask on a Linear
    layer with a PLANTED spectrum and assert it equals _blme_hill_alpha on
    the same singular values (and the analytic stable rank)."""
    torch = pytest.importorskip("torch")
    from blme.tasks.geometry.spectral import WeightSpectralTask

    # Planted Pareto-quantile spectrum (a=2.5), n=300, embedded via QR bases.
    rng = np.random.default_rng(0)
    n = 300
    q = (1.0 - (np.arange(1, n + 1) - 0.5) / n) ** (-1.0 / 2.5)
    s = np.sort(q)[::-1]
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Vt, _ = np.linalg.qr(rng.standard_normal((n, n)))
    W = (U * s) @ Vt.T

    lin = torch.nn.Linear(n, n, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor(W, dtype=torch.float32))

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = lin

    out = WeightSpectralTask(config={"tail_fraction": 0.2}).evaluate(
        _M(), None, dataset=None
    )
    assert "error" not in out

    # The task recomputes svdvals in float32; feed the SAME float32 svdvals
    # to the transcribed kernel so the comparison isolates the formula.
    S32 = torch.linalg.svdvals(lin.weight.detach()).numpy()
    expected_alpha = _blme_hill_alpha(S32, tail_fraction=0.2)
    assert out["_meta_avg_alpha"] == pytest.approx(expected_alpha, rel=1e-6), (
        "real task diverged from the transcribed kernel — update BOTH"
    )
    # The proxy is not the headline: "avg_alpha" exists only under _meta_, and
    # the KS-optimal fit of the eigenvalues gives another number.
    assert "avg_alpha" not in out
    assert np.isfinite(out["avg_alpha_official"])
    assert abs(out["avg_alpha_official"] - expected_alpha) > 1e-2

    expected_srank = float((S32 ** 2).sum() / (S32 ** 2).max())
    assert out["avg_stable_rank"] == pytest.approx(expected_srank, rel=1e-5)
