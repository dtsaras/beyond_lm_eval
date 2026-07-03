"""Numeric-parity test: BLME geometry_magnitude `_magnitude(D, t)` vs the
OFFICIAL magnipy reference (aidos-lab/magnipy).

TASK: geometry_magnitude
BLME: src/blme/tasks/geometry/magnitude.py :: _magnitude(D, t)
      |tX| = 1^T (exp(-t*D))^{-1} 1 = sum(np.linalg.solve(exp(-t*D), ones))

Reference: Limbeck, Andreeva, Sarkar, Rieck (2024), "Metric Space Magnitude
for Evaluating the Diversity of Latent Representations", NeurIPS 2024,
arXiv:2311.16054. Official code: aidos-lab/magnipy.

We pin BLME against TWO independent magnipy entry points on the SAME (D, t):
  * compute_magnitude_no_gpu(W, t)  -- pinv(exp(-t*distance_matrix(W,W))).sum()
  * compute_magnitude_from_distances(D, ts, method='cholesky') -- the
    weights_cholesky triangular-solve path, sum of the weight vector.

magnipy is imported from an OFFICIAL checkout at $BLME_MAGNIPY_DIR (or a
sibling scratchpad clone); if it is not importable the parity checks are
skipped, and only the self-contained anchor + fixture checks run. The bar is
< 1e-6 (float64 closed-form solve).

Reference source files are loaded by path, registered under their real dotted
names, WITHOUT executing magnipy/__init__.py -- that __init__ pulls a plotting
module which imports `trapz` from scipy.integrate (removed in scipy>=1.14) and
is unrelated to the magnitude solve. compute.py/weights.py themselves are the
exact upstream source.
"""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest


FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures/reference_parity/parity/magnitude.json"
)

TOL = 1e-6


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Locate an OFFICIAL magnipy checkout.
# ---------------------------------------------------------------------------
def _find_magnipy_dir():
    cand = os.environ.get("BLME_MAGNIPY_DIR")
    if cand and (Path(cand) / "magnipy" / "magnitude" / "compute.py").exists():
        return cand
    # scratchpad clone used during development
    scratch = (
        "/tmp/claude-1736197890/-home-dtsaras-projects-beyond-lm-eval/"
        "42dc090a-aa6d-4f5b-bd0f-fb7213022dc9/scratchpad/refrepos/magnipy"
    )
    if (Path(scratch) / "magnipy" / "magnitude" / "compute.py").exists():
        return scratch
    return None


MAGNIPY_DIR = _find_magnipy_dir()


def _load_from_source(magnipy_dir, dotted, relpath):
    spec = importlib.util.spec_from_file_location(
        dotted, os.path.join(magnipy_dir, relpath)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_magnipy_refs(magnipy_dir):
    """Return (compute_magnitude_no_gpu, compute_magnitude_from_distances)
    from the OFFICIAL magnipy source, or (None, None) if unavailable."""
    try:
        approx = _load_from_source(
            magnipy_dir, "mag_approx_ref",
            "magnipy/magnitude/approximation.py",
        )
        for pkg in ("magnipy", "magnipy.magnitude"):
            if pkg not in sys.modules:
                p = types.ModuleType(pkg)
                p.__path__ = [os.path.join(magnipy_dir, *pkg.split("."))]
                sys.modules[pkg] = p
        _load_from_source(magnipy_dir, "magnipy.magnitude.distances",
                          "magnipy/magnitude/distances.py")
        _load_from_source(magnipy_dir, "magnipy.magnitude.scales",
                          "magnipy/magnitude/scales.py")
        _load_from_source(magnipy_dir, "magnipy.magnitude.weights",
                          "magnipy/magnitude/weights.py")
        _load_from_source(magnipy_dir, "magnipy.magnitude.convergence",
                          "magnipy/magnitude/convergence.py")
        comp = _load_from_source(magnipy_dir, "magnipy.magnitude.compute",
                                 "magnipy/magnitude/compute.py")
        return (approx.compute_magnitude_no_gpu,
                comp.compute_magnitude_from_distances)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"magnipy reference not importable: {type(e).__name__}: {e}")
        return None, None


# ---------------------------------------------------------------------------
# BLME artifact under test (imported from src, NOT transcribed).
# ---------------------------------------------------------------------------
from blme.tasks.geometry.magnitude import (  # noqa: E402
    _magnitude,
    _magnitude_dimension,
    _median_heuristic_scale,
)


def _dist_matrix(X):
    X = np.asarray(X, dtype=np.float64)
    sq = np.sum(X * X, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    D = np.sqrt(d2)
    np.fill_diagonal(D, 0.0)
    return D


needs_magnipy = pytest.mark.skipif(
    MAGNIPY_DIR is None,
    reason="No OFFICIAL magnipy checkout (set BLME_MAGNIPY_DIR).",
)


# ===========================================================================
# (a) BLME _magnitude == magnipy (both no_gpu pinv and cholesky), same D & t.
# ===========================================================================
@needs_magnipy
def test_blme_matches_magnipy_both_paths():
    no_gpu, from_dist = _load_magnipy_refs(MAGNIPY_DIR)

    rng = np.random.default_rng(0)
    n, d = 25, 4
    W = rng.standard_normal((n, d))
    D = _dist_matrix(W)
    t_med = _median_heuristic_scale(D)
    ts = np.geomspace(t_med / 10, t_med * 10, 7)

    max_diff = 0.0
    for t in ts:
        m_nogpu = float(no_gpu(W, t))
        m_chol = float(
            from_dist(D, ts=np.array([t]), method="cholesky",
                      one_point_property=True, perturb_singularities=True)[0]
        )
        m_blme = _magnitude(D, float(t))
        assert abs(m_blme - m_nogpu) < TOL
        assert abs(m_blme - m_chol) < TOL
        max_diff = max(max_diff, abs(m_blme - m_nogpu), abs(m_blme - m_chol))

    assert max_diff < TOL
    fx = _fixture()["check_a_parity"]
    assert fx["pass"] is True
    # fixture records the same-order-of-magnitude fp residual
    assert max_diff < 10 * fx["max_abs_diff"] + 1e-12


# ===========================================================================
# (c) Full-ladder magnitude function parity (cholesky, all scales at once).
# ===========================================================================
@needs_magnipy
def test_full_ladder_magnitude_function_parity():
    _, from_dist = _load_magnipy_refs(MAGNIPY_DIR)

    rng = np.random.default_rng(0)
    W = rng.standard_normal((25, 4))
    D = _dist_matrix(W)
    t_med = _median_heuristic_scale(D)
    ladder = np.geomspace(t_med / 10, t_med * 10, 10)

    ref_curve = np.asarray(
        from_dist(D, ts=ladder, method="cholesky",
                  one_point_property=True, perturb_singularities=True),
        dtype=np.float64,
    )
    blme_curve = np.array([_magnitude(D, float(t)) for t in ladder])
    assert float(np.max(np.abs(ref_curve - blme_curve))) < TOL

    fx = _fixture()["check_c_full_ladder"]
    assert ref_curve[0] == pytest.approx(fx["magnipy_cholesky_first"], abs=1e-9)
    assert ref_curve[-1] == pytest.approx(fx["magnipy_cholesky_last"], abs=1e-9)


# ===========================================================================
# (b) Analytic anchors -- self-contained, run even without magnipy.
# ===========================================================================
def test_anchor_single_point_is_one():
    assert _magnitude(np.array([[0.0]]), 3.7) == pytest.approx(1.0, abs=1e-12)
    # any distance matrix of a single point, any t
    assert _magnitude(np.zeros((1, 1)), 100.0) == pytest.approx(1.0, abs=1e-12)


def test_anchor_two_points_large_t_tends_to_2():
    L = 5.0
    D2 = np.array([[0.0, L], [L, 0.0]])
    t = 20.0
    closed = 2.0 / (1.0 + np.exp(-t * L))
    m = _magnitude(D2, t)
    assert m == pytest.approx(closed, abs=TOL)
    assert m == pytest.approx(2.0, abs=1e-6)
    # and small t -> tends to 1 (whole space looks like one point)
    assert _magnitude(D2, 1e-6) == pytest.approx(1.0, abs=1e-3)


def test_anchor_n_separated_points_large_t_tends_to_n():
    # n well-separated points at large t -> magnitude ~ n (each point becomes
    # perfectly distinguishable, zeta -> I, |tX| -> n). Any far-apart cloud.
    pts = np.random.default_rng(1).standard_normal((8, 3)) * 50.0
    D3 = _dist_matrix(pts)
    m = _magnitude(D3, 5.0)
    assert m == pytest.approx(8.0, abs=1e-3)


def test_anchor_monotonic_nondecreasing():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((25, 4))
    D = _dist_matrix(W)
    t_med = _median_heuristic_scale(D)
    ts = np.geomspace(t_med / 20, t_med * 20, 30)
    vals = np.array([_magnitude(D, float(t)) for t in ts])
    diffs = np.diff(vals)
    assert int(np.sum(diffs < -1e-6)) == 0
    # spans from ~1 up toward n=25
    assert vals[0] < vals[-1]
    assert vals[-1] <= 25.0 + 1e-6


def test_anchor_magnitude_dimension_positive():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((25, 4))
    D = _dist_matrix(W)
    t_med = _median_heuristic_scale(D)
    ts = np.geomspace(t_med / 10, t_med * 10, 7)
    mags = np.array([_magnitude(D, float(t)) for t in ts])
    md = _magnitude_dimension(ts, mags)
    assert np.isfinite(md) and md > 0
    fx = _fixture()["check_b_anchors"]["magnitude_dimension"]
    assert md == pytest.approx(fx["value"], abs=1e-6)


def test_singular_zeta_perturbation_fallback():
    """Duplicate points -> zeta has repeated rows -> singular. The solve must
    not raise; the perturb/pinv fallback returns a finite magnitude."""
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])  # first two identical
    D = _dist_matrix(X)
    m = _magnitude(D, 1.0)
    assert np.isfinite(m)


# ===========================================================================
# Fixture bookkeeping.
# ===========================================================================
def test_fixture_records_parity_verdict():
    fx = _fixture()
    assert fx["task"] == "geometry_magnitude"
    assert fx["verdict"] == "PARITY"
    assert fx["check_a_parity"]["pass"] is True
    assert fx["check_c_full_ladder"]["pass"] is True
    anchors = fx["check_b_anchors"]
    assert anchors["single_point"]["pass"] is True
    assert anchors["two_points_large_t"]["pass"] is True
    assert anchors["n_separated_large_t"]["pass"] is True
