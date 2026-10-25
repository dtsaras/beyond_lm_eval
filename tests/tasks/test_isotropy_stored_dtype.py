"""The rank tolerance must follow the INCOMING precision, not the promoted one.

`_svd_metrics_for_layer` promotes torch inputs to float64 (so a float64 cloud
at 1e6 is not destroyed). Reading the array's dtype AFTER that promotion made
the tolerance tau = sigma_1 * max(N, D) * eps use float64's eps for every
torch input, shrinking it by 536870912x and admitting noise into the rank.
"""
import numpy as np
import pytest
import torch

from blme.tasks.geometry.isotropy import _svd_metrics_for_layer


def _exact_fixture():
    """100x5, exactly representable in float32; sigma = 1, .5, .25, 2^-27, 2^-28.

    The last two sit below float32's noise floor relative to sigma_1 and above
    float64's, so the numerical rank is 3 for float32 input and 5 for float64.
    """
    d = np.diag([1.0, 0.5, 0.25, 2.0 ** -27, 2.0 ** -28])
    return np.tile(np.vstack([d, -d]), (10, 1))


def test_float32_tensor_matches_float32_numpy():
    X = _exact_fixture()
    # FIXTURE INTEGRITY: nothing is lost by the float32 container itself.
    assert np.array_equal(X, X.astype(np.float32).astype(np.float64))

    r_np = _svd_metrics_for_layer(X.astype(np.float32))
    r_t = _svd_metrics_for_layer(torch.from_numpy(X).float())
    assert r_t["_meta_numerical_rank"] == 3
    assert r_t["cond_number"] == pytest.approx(4.0, rel=1e-12)
    assert r_t["_meta_numerical_rank"] == r_np["_meta_numerical_rank"]
    assert r_t["cond_number"] == pytest.approx(r_np["cond_number"], rel=1e-12)


def test_float64_tensor_keeps_its_own_resolution():
    """2^-27 IS signal in float64, so rank 5 there is correct, not a defect."""
    X = _exact_fixture()
    r = _svd_metrics_for_layer(torch.from_numpy(X).double())
    assert r["_meta_numerical_rank"] == 5
    assert r["cond_number"] == pytest.approx(2.0 ** 28, rel=1e-12)


def test_promotion_does_not_move_scale_free_metrics():
    """erank / PR must be identical on every route -- they carry no tolerance."""
    X = _exact_fixture()
    routes = [_svd_metrics_for_layer(X.astype(np.float32)),
              _svd_metrics_for_layer(torch.from_numpy(X).float()),
              _svd_metrics_for_layer(torch.from_numpy(X).double())]
    for key in ("effective_rank", "participation_ratio"):
        vals = [r[key] for r in routes]
        assert all(v == pytest.approx(vals[0], rel=1e-12) for v in vals), (
            f"{key} differs across routes: {vals}")


def test_bfloat16_reports_its_own_eps():
    """torch.finfo is used directly so bf16 is not flattened to a numpy dtype."""
    X = torch.from_numpy(_exact_fixture()).to(torch.bfloat16)
    r = _svd_metrics_for_layer(X)
    # bf16 eps is 0.0078125 -- far coarser than float32's, so the rank does not
    # exceed float32's on the same fixture.
    assert 1 <= r["_meta_numerical_rank"] <= 3


def test_bfloat16_tolerance_does_not_discard_everything():
    """The max(m, n) factor exceeds 1 at bf16 and would zero the rank.

    sigma_1 * max(m, n) * eps is calibrated for eps ~ 1e-16. At bf16's
    0.0078125 and width 768 the factor is 6.0, so the tolerance exceeds
    sigma_1 and EVERY singular value is discarded -- measured rank 0 and
    cond_number inf before the sqrt(max(m, n)) fallback was applied.
    """
    X = np.random.default_rng(0).standard_normal((200, 768))
    r16 = _svd_metrics_for_layer(torch.from_numpy(X).to(torch.bfloat16))
    r32 = _svd_metrics_for_layer(torch.from_numpy(X).float())
    assert r16["_meta_numerical_rank"] > 0, "bf16 tolerance discarded everything"
    assert np.isfinite(r16["cond_number"])
    # It should land near the float32 answer, not at a degenerate extreme.
    assert r16["cond_number"] == pytest.approx(r32["cond_number"], rel=0.05)


def test_float32_tolerance_is_unchanged_by_the_bf16_guard():
    """Every shipped route is float32; its established tolerance must not move.

    The guard only fires when max(m, n) * eps >= 1. At float32 and width 768
    that factor is 9.2e-5, so the primary envelope is kept untouched.
    """
    assert 768 * float(np.finfo(np.float32).eps) < 1.0
    X = np.random.default_rng(0).standard_normal((200, 768))
    r = _svd_metrics_for_layer(torch.from_numpy(X).float())
    # Pinned from the pre-guard code on this exact fixture.
    assert r["_meta_numerical_rank_tol"] == pytest.approx(
        0.003803996991049668, rel=1e-12)
    assert r["_meta_numerical_rank"] == 199
