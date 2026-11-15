"""topology_persistence_landscape: values of the exact landscape summaries.

``_landscape_stats`` (the critical-point quadrature behind every headline key)
is checked against persim's exact landscapes, ``PersLandscapeExact``, whose
critical points are integrated exactly here. ``evaluate()`` is then driven on
injected last-token clouds that are squares: a square of side a has one H1 bar
[a, a sqrt(2)], so every headline key has a closed form.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[2]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import invariance as inv  # noqa: E402
from blme.tasks.topology.persistence_landscape import (  # noqa: E402
    _compute_landscape, _landscape_stats)

K, N_POINTS = 5, 200          # the task's defaults (n_landscapes, n_points)


def _persim_exact(dgm, k_max):
    """Per-depth integral, maximum and L2 norm of persim's exact landscape:
    every lambda_k is linear between its critical points."""
    pl = pytest.importorskip("persim.landscapes")
    crit = pl.PersLandscapeExact(dgms=[np.asarray(dgm, dtype=np.float64)], hom_deg=0).critical_pairs
    integ, maxes, norms = [], [], []
    for k in range(k_max):
        if k >= len(crit):
            integ.append(0.0), maxes.append(0.0), norms.append(0.0)
            continue
        cp = np.asarray(crit[k], dtype=np.float64)
        t, y = cp[:, 0], cp[:, 1]
        dt = np.diff(t)
        integ.append(float(np.sum(dt * (y[:-1] + y[1:]) / 2.0)))
        maxes.append(float(y.max()))
        norms.append(float(np.sqrt(np.sum(dt * (y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2) / 3.0))))
    return integ, maxes, norms


def _blme_stats(dgm):
    """The call _layer_landscape makes for one diagram."""
    dgm = np.asarray(dgm, dtype=np.float64)
    land = _compute_landscape(dgm, K, N_POINTS)
    dt = (dgm[:, 1].max() - dgm[:, 0].min()) / (N_POINTS - 1)
    return _landscape_stats(land, dt, dgm=dgm)


def _diagrams():
    rng = np.random.default_rng(0)
    out = [np.array([[1.0, 6.0], [2.0, 8.0], [3.0, 4.0]])]
    for n in (7, 40):                       # overlapping tents, more bars than K
        b = rng.uniform(0.0, 3.0, n)
        out.append(np.stack([b, b + rng.exponential(1.0, n) + 1e-3], axis=1))
    d = np.sort(rng.exponential(1.0, 60))   # H0-like: every bar born at 0
    out.append(np.stack([np.zeros_like(d), d], axis=1))
    return out


@pytest.mark.parametrize("case", range(4))
def test_landscape_stats_match_persim_exact_landscapes(case):
    dgm = _diagrams()[case]
    got = _blme_stats(dgm)
    integ, maxes, norms = _persim_exact(dgm, K)
    assert got["_meta_landscape_quadrature"] == "exact"
    assert got["_meta_landscape_max_status"] == "exact"
    np.testing.assert_allclose(got["landscape_integrals"], integ, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(got["landscape_maxes"], maxes, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got["landscape_norms"], norms, rtol=1e-10, atol=1e-12)
    assert got["mean_landscape_integral"] == pytest.approx(np.mean(integ), rel=1e-10)


def test_landscape_integrals_sum_to_the_tent_areas():
    """With K >= #bars the lambda_k sum to the sum of all tents, so the
    integrals add up to sum (d - b)^2 / 4 (Bubenik 2015 Eq. 3.4)."""
    dgm = _diagrams()[0]
    got = _blme_stats(dgm)
    want = float(np.sum((dgm[:, 1] - dgm[:, 0]) ** 2) / 4.0)
    assert sum(got["landscape_integrals"]) == pytest.approx(want, rel=1e-12)
    assert got["mean_landscape_integral"] == pytest.approx(want / K, rel=1e-12)
    assert got["landscape_maxes"][:3] == pytest.approx([3.0, 2.0, 0.5], rel=1e-12)


def _square(a):
    return np.array([[0.0, 0.0], [a, 0.0], [0.0, a], [a, a]])


SIDES = {"first": 1.0, "mid": 2.0, "last": 3.0}


def _square_states():
    """Three blocks; document i ends at vertex i of a square whose side grows
    with depth (the earlier rows are not read: TOKEN_POLICY last_per_sequence)."""
    rng = np.random.default_rng(1)
    blocks = {li: [] for li in range(3)}
    for i in range(4):
        for li, a in enumerate(SIDES.values()):
            doc = rng.standard_normal((3, 2))
            doc[-1] = _square(a)[i]
            blocks[li].append(doc)
    return blocks


def test_evaluate_headline_keys_on_square_clouds():
    """One H1 bar of persistence p = a (sqrt 2 - 1): integral of lambda_1 is
    p^2 / 4 (mean over K = 5 landscapes), sup p / 2, L2 norm
    sqrt(2/3) (p/2)^1.5; the twins divide by the mean pairwise distance
    a (4 + 2 sqrt 2) / 6 to the matching power. ripser's diagrams are float32."""
    res = inv.run_task_on_states("topology_persistence_landscape", _square_states())
    assert res["_meta_landscape_n_subsamples"] == 1
    for role, a in SIDES.items():
        p = a * (np.sqrt(2.0) - 1.0)
        mpd = a * (4.0 + 2.0 * np.sqrt(2.0)) / 6.0
        mean_int = p ** 2 / 4.0 / K
        max1 = p / 2.0
        norm1 = np.sqrt(2.0 / 3.0) * (p / 2.0) ** 1.5
        want = {
            "mean_landscape_integral": mean_int,
            "landscape_max_1": max1,
            "mean_landscape_integral_mpd2": mean_int / mpd ** 2,
            "landscape_max_1_mpd": max1 / mpd,
            "landscape_norm_1_mpd15": norm1 / mpd ** 1.5,
        }
        for key, v in want.items():
            assert res[f"{role}_layer_h1_{key}"] == pytest.approx(v, rel=1e-6), (role, key)
        assert res[f"{role}_layer_h1_landscape_norm_1"] == pytest.approx(norm1, rel=1e-6)
