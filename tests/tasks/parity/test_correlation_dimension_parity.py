"""Numeric-parity test: BLME geometry_correlation_dimension vs Grassberger-Procaccia.

TASK: geometry_correlation_dimension
BLME: src/blme/tasks/geometry/correlation_dimension.py
      CorrelationDimensionTask.evaluate(), GP kernel lines 83-119.

Reference: Grassberger & Procaccia, Phys. Rev. Lett. 50(5) 346 (1983) and
           Physica D 9(1) 189-208 (1983); official code nolds.corr_dim.

This is a FORMULA-FAITHFUL / PROXY task (see scratchpad
wave1/correlation_dimension_verify.py). The bar we pin here:

  (a) BLME's GP kernel == an INDEPENDENT GP implementation (scipy.pdist ->
      C(r) -> log-log slope) on the SAME radii, to < 1e-6.
  (b) BLME's slope == nolds' INTENDED GP kernel (self-matches excluded, per
      nolds' own docstring) on the SAME delay-embedded cloud + SAME rvals, to
      < 1e-6.  NOTE: the raw nolds.corr_dim API leaves the distance-matrix
      diagonal at 0 and counts those n self-pairs, inflating each C(r) by
      n/(n(n-1)); that is an upstream artifact, not a BLME discrepancy.
  (c) BLME recovers known dimensions DIRECTIONALLY: line in R^d -> ~1 (tight),
      and line < plane < ambient.  The plane lands ~1.5 (not 2.0) because
      BLME's 5th-95th percentile radius window includes the large-r saturating
      regime, biasing the GP slope down; this is a documented property of the
      scaling-region choice, not a code bug.

The BLME GP kernel below is transcribed VERBATIM from
correlation_dimension.py lines 83-119 so the test exercises the exact numeric
path without importing torch / a model. src/blme is NOT modified.
"""

import json
from pathlib import Path

import numpy as np
import pytest

nolds = pytest.importorskip("nolds")
import nolds.measures as nm  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402


FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures/reference_parity/parity/correlation_dimension.json"
)


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# BLME GP kernel, transcribed from src/blme/tasks/geometry/correlation_dimension.py
# lines 83-119 (compute pairwise L2 dists -> upper triangle -> percentile-spaced
# logspace radii -> C(r) = fraction of pairs with dist < r -> log-log polyfit).
# rvals override pins it against a reference on an identical radius grid.
# ---------------------------------------------------------------------------
def _blme_corr_dim(H, num_radii=30, rvals=None):
    H = np.asarray(H, dtype=np.float64)
    N = H.shape[0]
    diff = H[:, None, :] - H[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    distances = dist_matrix[iu]

    if rvals is None:
        r_min = np.percentile(distances, 5)
        r_max = np.percentile(distances, 95)
        assert r_min > 0 and r_max > 0 and r_min < r_max
        radii = np.logspace(np.log10(r_min), np.log10(r_max), num=num_radii)
    else:
        radii = np.asarray(rvals, dtype=np.float64)

    total_pairs = len(distances)
    C_r, valid_radii = [], []
    for r in radii:
        count = np.sum(distances < r)        # strict '<' as in BLME
        c = count / total_pairs
        if c > 0:
            C_r.append(c)
            valid_radii.append(r)
    assert len(valid_radii) >= 3

    log_r = np.log(valid_radii)
    log_Cr = np.log(C_r)
    slope, _intercept = np.polyfit(log_r, log_Cr, 1)
    return float(slope), np.asarray(valid_radii), np.asarray(C_r)


def _independent_gp(H, rvals):
    """GP straight from the 1983 PRL: C(r)=fraction of i<j pairs with dist<r."""
    d = pdist(np.asarray(H, dtype=np.float64), metric="euclidean")
    rvals = np.asarray(rvals, dtype=np.float64)
    C = np.array([np.sum(d < r) / d.size for r in rvals])
    keep = C > 0
    slope, _ = np.polyfit(np.log(rvals[keep]), np.log(C[keep]), 1)
    return float(slope), C


def _logistic_series(n=500, x0=0.371, a=3.97):
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = a * x[i - 1] * (1.0 - x[i - 1])
    return x


# ===========================================================================
# (a) BLME kernel == independent GP on the same radii (bit-exact).
# ===========================================================================
def test_blme_matches_independent_gp():
    rng = np.random.default_rng(7)
    H = rng.standard_normal((60, 6))
    slope_blme, used_radii, C_blme = _blme_corr_dim(H, num_radii=30)
    slope_indep, C_indep = _independent_gp(H, used_radii)

    assert abs(slope_blme - slope_indep) < 1e-6
    assert float(np.max(np.abs(C_blme - C_indep))) < 1e-12

    fx = _fixture()["check_a_independent_gp"]
    assert slope_blme == pytest.approx(fx["blme_slope"], abs=1e-9)
    assert slope_indep == pytest.approx(fx["independent_slope"], abs=1e-9)


# ===========================================================================
# (b) BLME slope == nolds' INTENDED GP kernel (self-matches excluded), same rvals.
# ===========================================================================
def test_blme_matches_nolds_intended_kernel():
    x = _logistic_series(n=500)
    emb_dim, lag = 3, 1
    sd = np.std(x, ddof=1)
    rvals = np.asarray(nm.logarithmic_r(0.1 * sd, 0.5 * sd, 1.03))
    orbit = nm.delay_embedding(x, emb_dim, lag=lag)
    n = len(orbit)

    # nolds INTENDED: diagonal excluded (inf) per nolds' own docstring (j != i).
    dists = np.full((n, n), np.inf)
    for i in range(n):
        dd = nm.rowwise_euclidean(orbit[i + 1:], orbit[i])
        dists[i + 1:, i] = dd
        dists[i, i + 1:] = dd
    csums = np.array([1.0 / (n * (n - 1)) * np.sum(dists <= r) for r in rvals])
    nz = csums > 0
    nolds_intended = float(np.polyfit(np.log(rvals[nz]), np.log(csums[nz]), 1)[0])

    slope_blme, _, _ = _blme_corr_dim(orbit, rvals=rvals)

    assert abs(slope_blme - nolds_intended) < 1e-6

    fx = _fixture()["check_b_nolds"]
    assert nolds_intended == pytest.approx(fx["nolds_intended_slope"], abs=1e-9)
    assert slope_blme == pytest.approx(fx["blme_slope"], abs=1e-9)


def test_nolds_raw_api_self_match_artifact_is_documented():
    """Pin the upstream nolds diagonal self-match artifact so a future nolds
    fix that changes the raw-API number is caught and the fixture updated.

    BLME is correct (self-matches excluded); the raw API slope differs from
    BLME by the recorded artifact gap.  This test asserts the artifact, not a
    BLME discrepancy.
    """
    x = _logistic_series(n=500)
    emb_dim, lag = 3, 1
    sd = np.std(x, ddof=1)
    rvals = np.asarray(nm.logarithmic_r(0.1 * sd, 0.5 * sd, 1.03))

    api_slope = nolds.corr_dim(x, emb_dim, lag=lag, rvals=rvals, fit="poly")

    fx = _fixture()["check_b_nolds"]
    assert float(api_slope) == pytest.approx(fx["nolds_raw_api_slope"], abs=1e-9)
    gap = abs(float(api_slope) - fx["blme_slope"])
    assert gap == pytest.approx(fx["artifact_gap_vs_raw_api"], abs=1e-9)


# ===========================================================================
# (c) BLME recovers known dimensions directionally: line ~1, line < plane < amb.
# ===========================================================================
def test_blme_recovers_known_dimensions_directionally():
    rng = np.random.default_rng(123)
    N = 3000

    t = rng.random(N)
    direction = rng.standard_normal(5)
    line_cloud = t[:, None] * direction[None, :]
    slope_line, _, _ = _blme_corr_dim(line_cloud, num_radii=30)

    uv = rng.random((N, 2))
    basis = rng.standard_normal((2, 5))
    plane_cloud = uv @ basis
    slope_plane, _, _ = _blme_corr_dim(plane_cloud, num_radii=30)

    # line recovers ~1 tightly
    assert abs(slope_line - 1.0) < 0.15
    # directional separation: line clearly below plane, plane below ambient (5)
    assert (slope_line + 0.2) < slope_plane < 5.0

    fx = _fixture()["check_c_known_dims"]
    assert slope_line == pytest.approx(fx["line_R5_slope"], abs=1e-6)
    assert slope_plane == pytest.approx(fx["plane_R5_slope"], abs=1e-6)


def test_fixture_records_parity_verdict():
    fx = _fixture()
    assert fx["task"] == "geometry_correlation_dimension"
    assert fx["verdict"] == "PARITY"
    assert fx["check_a_independent_gp"]["pass"] is True
    assert fx["check_b_nolds"]["pass"] is True
    assert fx["check_c_known_dims"]["pass"] is True
