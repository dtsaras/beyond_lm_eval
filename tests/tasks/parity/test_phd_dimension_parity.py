"""Numeric-parity test: BLME geometry_phd_dimension vs GPTID PHD (official).

TASK: geometry_phd_dimension
BLME: src/blme/tasks/geometry/phd_dimension.py
      PHDimensionTask.evaluate(); core math in `_phd_dimension` (and the Prim
      MST kernel `_prim_tree`).

Reference (Persistent-Homology Dimension):
    Tulchinskii, Kuznetsov, Kushnareva, Cherniavskii, Nikolenko, Burnaev,
    Barannikov, Piontkovskaya (2023). "Intrinsic Dimension Estimation for
    Robust Detection of AI-Generated Texts." NeurIPS 2023, arXiv:2306.04723.
    Official code: github.com/ArGintum/GPTID @ 8c8759e — IntrinsicDim.py,
    class ``PHD`` (``fit_transform`` / ``_calc_ph_dim_single`` / ``prim_tree``).

Verdict: PARITY (bit-exact when seed-matched).

Parity strategy
---------------
The reference draws subsamples from the GLOBAL legacy ``np.random`` inside
``n_reruns`` threads; thread interleaving makes the draw order irreproducible.
For a DETERMINISTIC comparison we run the reference with ``n_reruns=1`` (a
single, immediately-joined thread) after ``np.random.seed(SEED)``, and drive
BLME's ``_phd_dimension`` with ``np.random.RandomState(SEED)`` issuing the
identical ``np.random.choice(n_cloud, size=n, replace=False)`` sequence
(reruns -> test_n -> restarts). Both then consume the same RNG stream and must
agree to < 1e-9 (observed: 0.0, bit-exact).

The official ``IntrinsicDim.py`` is imported from the cloned GPTID repo if
present; otherwise these reference-backed cases skip (the fixture still pins
the recorded numbers).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from blme.tasks.geometry.phd_dimension import _phd_dimension, _prim_tree

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures/reference_parity/parity/phd_dimension.json"
)

# Candidate locations for the cloned official GPTID repo (holds IntrinsicDim.py).
_REF_CANDIDATES = [
    "/tmp/claude-1736197890/-home-dtsaras-projects-beyond-lm-eval/"
    "42dc090a-aa6d-4f5b-bd0f-fb7213022dc9/scratchpad/refrepos/GPTID",
]


def _load_reference():
    for d in _REF_CANDIDATES:
        if (Path(d) / "IntrinsicDim.py").exists():
            if d not in sys.path:
                sys.path.insert(0, d)
            import IntrinsicDim  # noqa: WPS433

            return IntrinsicDim.PHD
    return None


RefPHD = _load_reference()
_HAVE_REF = RefPHD is not None
_needs_ref = pytest.mark.skipif(
    not _HAVE_REF, reason="official GPTID/IntrinsicDim.py not present"
)


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _ref_single_thread(X, seed):
    """Official PHD with n_reruns=1 under a fixed global seed (deterministic)."""
    ph = RefPHD(alpha=1.0, metric="euclidean", n_reruns=1, n_points=7, n_points_min=3)
    np.random.seed(seed)
    return ph.fit_transform(
        np.asarray(X, dtype=np.float64),
        min_points=50, max_points=512, point_jump=40,
    )


def _blme_single(X, seed):
    return _phd_dimension(np.asarray(X, dtype=np.float64), n_reruns=1, seed=seed)


def _uniform_cloud(k, n, seed):
    return np.random.RandomState(seed).rand(n, k)


def _line_cloud(n, ambient, seed):
    r = np.random.RandomState(seed)
    t = r.rand(n)
    direction = r.randn(ambient)
    return t[:, None] * direction[None, :]


# ===========================================================================
# (0) Prim MST kernel matches the reference transcription on a fixed matrix.
# ===========================================================================
@_needs_ref
def test_prim_tree_matches_reference_kernel():
    import IntrinsicDim as ref  # noqa: WPS433

    rng = np.random.RandomState(0)
    pts = rng.rand(40, 4)
    from scipy.spatial.distance import cdist

    D = cdist(pts, pts, metric="euclidean")
    assert _prim_tree(D, alpha=1.0) == pytest.approx(ref.prim_tree(D, 1.0), abs=0.0)
    assert _prim_tree(D, alpha=0.5) == pytest.approx(ref.prim_tree(D, 0.5), abs=0.0)


# ===========================================================================
# (A) EXACT parity: n_reruns=1, single-thread, seed-matched (bit-exact).
# ===========================================================================
@_needs_ref
def test_exact_parity_seed_matched():
    fx = _fixture()
    tol = fx["tol_abs"]
    max_diff = 0.0
    for k in (1, 2, 3):
        X = _uniform_cloud(k, 600, seed=100 + k)
        for seed in (0, 1, 7, 42):
            ref = _ref_single_thread(X, seed)
            blme = _blme_single(X, seed)
            diff = abs(ref - blme)
            max_diff = max(max_diff, diff)
            assert diff < tol, f"R^{k} seed={seed}: |{ref}-{blme}|={diff}"
    # Bit-exact against the recorded run.
    assert max_diff <= fx["exact_max_abs_diff"] + 1e-15
    assert fx["exact_max_abs_diff"] == pytest.approx(0.0, abs=1e-15)


@_needs_ref
def test_exact_parity_matches_fixture_values():
    fx = _fixture()
    for case in fx["exact_parity_cases"]:
        X = _uniform_cloud(case["k"], case["N"], seed=case["cloud_seed"])
        seed = case["sample_seed"]
        ref = _ref_single_thread(X, seed)
        blme = _blme_single(X, seed)
        assert ref == pytest.approx(case["ref"], abs=1e-12)
        assert blme == pytest.approx(case["blme"], abs=1e-12)
        assert abs(ref - blme) == pytest.approx(case["abs_diff"], abs=1e-15)


# ===========================================================================
# (B) BLME alone reproduces the recorded numbers without the reference repo.
# ===========================================================================
def test_blme_reproduces_recorded_numbers():
    fx = _fixture()
    for case in fx["exact_parity_cases"]:
        X = _uniform_cloud(case["k"], case["N"], seed=case["cloud_seed"])
        blme = _blme_single(X, case["sample_seed"])
        assert blme == pytest.approx(case["blme"], abs=1e-12)


# ===========================================================================
# (C) Anchors — recover known intrinsic dimensions; BLME tracks the reference.
# ===========================================================================
@_needs_ref
def test_anchors_track_reference():
    fx = _fixture()["anchors"]
    for k in (1, 2, 3):
        X = _uniform_cloud(k, 1000, seed=2000 + k)
        ref = _ref_single_thread(X, 42)
        blme = _blme_single(X, 42)
        assert abs(ref - blme) < fx["uniform_R%d" % k]["abs_diff"] + 1e-9
        assert blme == pytest.approx(fx["uniform_R%d" % k]["blme"], abs=1e-9)
    Xl = _line_cloud(1000, 5, 3000)
    assert _blme_single(Xl, 42) == pytest.approx(fx["line_in_R5"]["blme"], abs=1e-9)


def test_anchors_recover_known_dimensions():
    """PHD recovers intrinsic dim within its known (downward) bias:
    R^k uniform -> ~k, a line -> ~1, and the estimates are monotone in k."""
    fx = _fixture()["anchors"]
    d1 = _blme_single(_uniform_cloud(1, 1000, 2001), 42)
    d2 = _blme_single(_uniform_cloud(2, 1000, 2002), 42)
    d3 = _blme_single(_uniform_cloud(3, 1000, 2003), 42)
    dline = _blme_single(_line_cloud(1000, 5, 3000), 42)

    # 1-D clouds pin tightly to 1; 2-D and 3-D sit under the true dim (known
    # small-sample PHD bias) but stay close and strictly ordered.
    assert abs(d1 - 1.0) < 0.1
    assert abs(dline - 1.0) < 0.1
    assert 1.7 < d2 < 2.1
    assert 2.5 < d3 < 3.1
    assert d1 < d2 < d3

    # Match the recorded anchor values.
    assert d1 == pytest.approx(fx["uniform_R1"]["blme"], abs=1e-9)
    assert d3 == pytest.approx(fx["uniform_R3"]["blme"], abs=1e-9)


# ===========================================================================
# (D) Fixture records the PARITY verdict.
# ===========================================================================
def test_fixture_records_parity_verdict():
    fx = _fixture()
    assert fx["task"] == "geometry_phd_dimension"
    assert fx["verdict"] == "PARITY"
    assert fx["reference_impl"]["commit"].startswith("8c8759e")
    assert fx["exact_max_abs_diff"] == 0.0
