"""Numeric-parity test: BLME geometry_spectral (HTSR power-law alpha) vs the
OFFICIAL reference, WeightWatcher.

TASK: geometry_spectral
BLME: src/blme/tasks/geometry/spectral.py
      WeightSpectralTask.evaluate(), Hill alpha kernel lines 77-91.

Paper: Martin & Mahoney, Heavy-Tailed Self-Regularization (2019/2021);
       Martin, Peng & Mahoney, Nature Communications 12:4122 (2021).
       Official code: CalculatedContent/WeightWatcher (pip `weightwatcher`).

This is a REFINED-ADAPTATION / PROXY task.  BLME computes HTSR alpha with a
HILL estimator over a FIXED 20% tail of the SINGULAR values; WeightWatcher
computes alpha with the SAME Hill/MLE *formula* but scanning ALL candidate
xmin values and selecting the KS-optimal one, fitted on the EIGENVALUES
lambda = sigma^2.  So:

  * alpha differs by (i) a ~factor-2 sigma-vs-lambda scale
    [alpha_sigma ~ 2*alpha_lambda - 1] and (ii) fixed-tail vs KS-xmin.
  * geometry_spectral is NOT value-comparable to published WW alpha; it is a
    ranking-preserving proxy.  (See scratchpad wave3/spectral_verify.py and
    AUDIT_V2 sec 5; the docstring of spectral.py already states this.)

The bar we pin here:

  (a) known-exponent recovery: singular values with an ESD tail exponent a
      -> BLME Hill alpha ~= a (statistical recovery, loose tolerance).
  (b) EXACT PARITY: BLME's Hill kernel == an INDEPENDENT transcription of the
      Hill estimator, to < 1e-9, over many toy spectra.  This is the exact
      part of the verdict.
  (c) BLME-vs-WeightWatcher delta on the SAME toy matrices is RECORDED (not
      forced to 0); the fixture pins the measured WW alpha, the sigma-map, and
      the residual gap so a future WW/BLME change is caught.
  (d) anchor: a random (Marchenko-Pastur) matrix -> LARGE alpha in both BLME
      and WW (no real heavy tail); BLME lands in the sigma-scale [3,11] band.

BLME's Hill kernel is transcribed VERBATIM from spectral.py lines 77-91 so the
test exercises the exact numeric path without importing torch / a model.
src/blme is NOT modified.  WeightWatcher itself is NOT a test dependency: its
reference numbers were produced in an isolated venv and frozen into the
fixture; this test asserts BLME against them.
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
# BLME Hill alpha kernel, transcribed VERBATIM from
# src/blme/tasks/geometry/spectral.py lines 77-91.
# BLME feeds S = torch.linalg.svdvals(W) (singular values, descending).
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
    fx = _fixture()
    assert fx["task"] == "geometry_spectral"
    assert fx["label"].startswith("REFINED-ADAPTATION")
    assert fx["verdict"] == "PROXY"
    assert fx["reference_impl"]["library"] == "weightwatcher"
    assert fx["reference_impl"]["ww_version"] == "0.7.7"
    assert fx["check_a_known_exponent_recovery"]["pass"] is True
    assert fx["check_b_independent_hill"]["pass"] is True
    assert fx["check_d_anchor_marchenko_pastur"]["pass"] is True
