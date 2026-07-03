"""Numeric-parity test: BLME interpretability_activation_kurtosis vs scipy.

TASK: interpretability_activation_kurtosis
BLME: src/blme/tasks/interpretability/activation_kurtosis.py
      ActivationKurtosisTask.evaluate() + kernel `_activation_kurtosis_stats`.

The canonical numeric reference for kurtosis is scipy.stats.kurtosis with
scipy's DEFAULT convention:

    fisher=True  -> "excess" kurtosis: normal distribution -> 0
    bias=True    -> population central moments,
                    E[(x-mu)^4] / E[(x-mu)^2]^2 - 3
                    (NO n-based sample-size / Fisher correction).

BLME's `_activation_kurtosis_stats` computes per-channel excess kurtosis
along the token axis (axis=0). We assert BIT-EXACT parity (< 1e-9 max abs
diff) with scipy on synthetic activations, and recover the textbook
anchors (Gaussian ~0, Laplace ~3, uniform ~-1.2, single spike -> huge)
while ALSO agreeing with scipy element-for-element on the same array.

Primary paper: Akhondzadeh et al., "KurTail: Kurtosis-based LLM
Quantization", Findings of EMNLP 2025 (arXiv:2503.01483) — the kurtosis
minimised for 4-bit quantization. Motivation: Sun et al. 2024 "Massive
Activations" (arXiv:2402.17762); Dettmers et al. 2022 "LLM.int8()"
(arXiv:2208.07339).

src/blme is NOT modified; the ACTUAL kernel is imported.
"""
import json
from pathlib import Path

import numpy as np
import pytest

stats = pytest.importorskip("scipy.stats")

from blme.tasks.interpretability.activation_kurtosis import (  # noqa: E402
    _activation_kurtosis_stats,
)

TOL = 1e-9

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures/reference_parity/parity/activation_kurtosis.json"
)


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _parity_array():
    """The deterministic mixed-tailedness activation cloud used by both the
    test and the fixture. cols: student-t / laplace / uniform / normal."""
    rng = np.random.default_rng(20250703)
    N, D = 3000, 24
    A = rng.standard_normal((N, D)).astype(np.float64)
    A[:, :6] = rng.standard_t(df=5, size=(N, 6))
    A[:, 6:12] = rng.laplace(0.0, 1.0, size=(N, 6))
    A[:, 12:18] = rng.uniform(-1, 1, size=(N, 6))
    return A


# ===========================================================================
# PARITY: BLME per-channel excess kurtosis == scipy (bit-exact).
# ===========================================================================
def test_per_channel_matches_scipy_bit_exact():
    A = _parity_array()
    blme = _activation_kurtosis_stats(A, threshold=10.0)
    scipy_pc = stats.kurtosis(A, axis=0, fisher=True, bias=True)

    max_abs = float(np.max(np.abs(blme["per_channel_kurtosis"] - scipy_pc)))
    assert max_abs < TOL, f"per-channel max abs diff {max_abs:.3e} >= {TOL}"

    fx = _fixture()["parity_array"]
    assert max_abs == pytest.approx(fx["per_channel_max_abs_diff_vs_scipy"], abs=1e-12)
    assert blme["mean"] == pytest.approx(fx["blme_mean"], abs=1e-12)
    assert blme["max"] == pytest.approx(fx["blme_max"], abs=1e-12)


def test_summaries_match_scipy():
    A = _parity_array()
    blme = _activation_kurtosis_stats(A, threshold=10.0)
    scipy_pc = stats.kurtosis(A, axis=0, fisher=True, bias=True)

    assert abs(blme["mean"] - float(np.mean(scipy_pc))) < TOL
    assert abs(blme["max"] - float(np.max(scipy_pc))) < TOL
    assert abs(blme["frac_above_threshold"] - float(np.mean(scipy_pc > 10.0))) < TOL


def test_tensor_kurtosis_matches_scipy():
    A = _parity_array()
    blme = _activation_kurtosis_stats(A)
    scipy_tensor = float(stats.kurtosis(A.reshape(-1), fisher=True, bias=True))
    assert abs(blme["tensor_kurtosis"] - scipy_tensor) < TOL


# ===========================================================================
# BIAS FLAG: BLME matches scipy bias=True (default), NOT bias=False.
# ===========================================================================
def test_matches_bias_true_not_bias_false():
    rng = np.random.default_rng(9)
    A = rng.standard_normal((30, 5)).astype(np.float64)  # small N -> flags differ
    blme_pc = _activation_kurtosis_stats(A)["per_channel_kurtosis"]
    k_true = stats.kurtosis(A, axis=0, fisher=True, bias=True)
    k_false = stats.kurtosis(A, axis=0, fisher=True, bias=False)

    assert float(np.max(np.abs(blme_pc - k_true))) < TOL
    # bias=False is a materially different number on small N.
    assert float(np.max(np.abs(blme_pc - k_false))) > 1e-3


# ===========================================================================
# ANCHORS: textbook excess kurtoses recovered AND scipy-exact.
# ===========================================================================
@pytest.mark.parametrize(
    "name, sampler, expected, recover_tol",
    [
        ("gaussian", lambda r, n: r.standard_normal(n), 0.0, 0.2),
        ("laplace", lambda r, n: r.laplace(0.0, 1.0, n), 3.0, 0.2),
        ("uniform", lambda r, n: r.uniform(-1.0, 1.0, n), -1.2, 0.05),
    ],
)
def test_anchor_recovery_and_scipy_parity(name, sampler, expected, recover_tol):
    rng = np.random.default_rng(2024)
    N = 400_000
    x = sampler(rng, N).astype(np.float64).reshape(N, 1)

    blme_k = float(_activation_kurtosis_stats(x)["per_channel_kurtosis"][0])
    scipy_k = float(stats.kurtosis(x[:, 0], fisher=True, bias=True))

    # Recovery of the known kurtosis at large N.
    assert abs(blme_k - expected) < recover_tol, f"{name}: {blme_k} vs {expected}"
    # Exact agreement with scipy on the SAME array.
    assert abs(blme_k - scipy_k) < TOL

    fx = _fixture()["anchors"][name]
    assert blme_k == pytest.approx(fx["recovered"], abs=1e-9)


def test_anchor_spike_is_very_high_and_scipy_exact():
    rng = np.random.default_rng(2024)
    N = 400_000
    x = (rng.standard_normal(N) * 1e-3).astype(np.float64)
    x[0] = 1000.0  # single massive spike
    x = x.reshape(N, 1)

    blme_k = float(_activation_kurtosis_stats(x)["per_channel_kurtosis"][0])
    scipy_k = float(stats.kurtosis(x[:, 0], fisher=True, bias=True))

    assert blme_k > 1e4, f"spike kurtosis not high: {blme_k}"
    assert abs(blme_k - scipy_k) < 1e-3  # huge magnitude -> relax abs tol


# ===========================================================================
# EDGE CASES: zero-variance channel guarded; tiny/empty inputs -> NaN.
# ===========================================================================
def test_zero_variance_channel_guarded():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((500, 4)).astype(np.float64)
    A[:, 1] = 7.0  # constant channel -> undefined kurtosis
    blme = _activation_kurtosis_stats(A)
    pc = blme["per_channel_kurtosis"]

    assert np.isnan(pc[1])
    assert np.all(np.isfinite(pc[[0, 2, 3]]))
    assert blme["n_finite_channels"] == 3


def test_empty_and_single_row_inputs():
    empty = _activation_kurtosis_stats(np.empty((0, 4)))
    one = _activation_kurtosis_stats(np.ones((1, 4)))
    assert np.isnan(empty["mean"]) and empty["n_finite_channels"] == 0
    assert np.isnan(one["mean"])


def test_fixture_records_parity_verdict():
    fx = _fixture()
    assert fx["task"] == "interpretability_activation_kurtosis"
    assert fx["verdict"] == "PARITY"
    assert fx["reference"].startswith("scipy.stats.kurtosis")
    assert fx["parity_array"]["per_channel_max_abs_diff_vs_scipy"] == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
