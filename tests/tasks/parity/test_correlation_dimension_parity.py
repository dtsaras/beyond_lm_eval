"""geometry_correlation_dimension: the GP correlation integral vs nolds and the 1983 definition.

The headline ``correlation_dimension`` is the slope of log C(r) on log r over
Du & Tanaka-Ishii's fixed window, C in [20 / (N (N - 1)), 1 / N], fitted at
every positive sorted pair distance in it (``_fixed_window_dimension``). The
5th-95th percentile radius grid (``_gp_correlation_dimension``) is kept only as
``_meta_correlation_dimension_percentile_window``: it lies in the saturation
regime and under-reads (a plane reads 1.48).

* the headline slope equals a fit of nolds' intended correlation sum (its
  docstring's j != i; the raw ``nolds.corr_dim`` API also counts the n
  self-pairs) evaluated at the window's own radii, and recovers a line, a
  plane and a cube;
* the ``_meta_`` percentile fit equals an independent pdist GP and nolds'
  intended kernel on the same radii (fixture values recorded at transcription).

The window itself and the GP 1983 Lorenz / Henon constants are pinned in
tests/tasks/test_correlation_dimension_protocol.py.
"""

import json
from pathlib import Path

import numpy as np
import pytest

nolds = pytest.importorskip("nolds")
import nolds.measures as nm  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402

from blme.tasks.geometry.correlation_dimension import (  # noqa: E402
    _fixed_window_dimension,
    _gp_correlation_dimension,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures/reference_parity/parity/correlation_dimension.json"
)


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _logistic_series(n=500, x0=0.371, a=3.97):
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = a * x[i - 1] * (1.0 - x[i - 1])
    return x


def _nolds_intended_csums(orbit, rvals):
    """nolds.corr_dim's correlation sum with the diagonal excluded (j != i),
    built from its own rowwise_euclidean."""
    n = len(orbit)
    dists = np.full((n, n), np.inf)
    for i in range(n):
        dd = nm.rowwise_euclidean(orbit[i + 1:], orbit[i])
        dists[i + 1:, i] = dd
        dists[i, i + 1:] = dd
    flat = np.sort(dists[np.isfinite(dists)])
    return np.array([np.searchsorted(flat, r, side="right") / (n * (n - 1)) for r in rvals])


def _orbit():
    return nm.delay_embedding(_logistic_series(n=500), 3, lag=1)


# ===========================================================================
# The headline: the fixed-window fit.
# ===========================================================================
def test_headline_slope_equals_nolds_intended_sum_on_the_window():
    """At r = the k-th smallest distance (no ties), nolds' intended sum is
    k / M, the headline's C; the fit over the window's radii is the same line."""
    orbit = _orbit()
    n = len(orbit)
    d = np.sort(pdist(orbit))
    assert np.unique(d).size == d.size
    slope, r2, n_fit, status, _ = _fixed_window_dimension(d, n)
    assert status == "ok" and n_fit == (n - 1) // 2 - 10 + 1
    radii = d[9:n_fit + 9]
    cs = _nolds_intended_csums(orbit, radii)
    np.testing.assert_allclose(cs, np.arange(10, n_fit + 10) / d.size, rtol=1e-12)
    ref = float(np.polyfit(np.log(radii), np.log(cs), 1)[0])
    assert slope == pytest.approx(ref, rel=1e-10)


def test_headline_recovers_known_dimensions():
    """N = 3000 points uniform on a segment, a square and a cube (the first two
    linearly embedded in R^5): 1.036 / 1.969 / 3.078, within 0.1 of the
    dimension, while the _meta_ percentile window reads 0.891 / 1.480 / 2.112."""
    rng = np.random.default_rng(123)
    N = 3000
    line = rng.random(N)[:, None] * rng.standard_normal(5)[None, :]
    plane = rng.random((N, 2)) @ rng.standard_normal((2, 5))
    cube = rng.random((N, 3))
    for dim, X in ((1, line), (2, plane), (3, cube)):
        d = pdist(X)
        slope, r2, _, status, _ = _fixed_window_dimension(d, N)
        assert status == "ok" and r2 > 0.99
        assert slope == pytest.approx(dim, abs=0.1), dim
        pct, _, _ = _gp_correlation_dimension(d)
        assert pct < slope - 0.1, dim


# ===========================================================================
# The _meta_ percentile-window fit.
# ===========================================================================
def test_percentile_fit_matches_independent_gp():
    rng = np.random.default_rng(7)
    H = rng.standard_normal((60, 6))
    d = pdist(H)
    radii = np.logspace(np.log10(np.percentile(d, 5)), np.log10(np.percentile(d, 95)), num=30)
    C = np.array([np.sum(d < r) / d.size for r in radii])
    keep = C > 0
    ref = float(np.polyfit(np.log(radii[keep]), np.log(C[keep]), 1)[0])
    slope, _, status = _gp_correlation_dimension(d, num_radii=30)
    assert status == "ok"
    assert slope == pytest.approx(ref, abs=1e-12)
    assert slope == pytest.approx(_fixture()["check_a_independent_gp"]["blme_slope"], abs=1e-9)


def test_percentile_kernel_matches_nolds_intended_kernel():
    """On nolds' own radius grid, logarithmic_r(0.1 sd, 0.5 sd, 1.03)."""
    x = _logistic_series(n=500)
    sd = np.std(x, ddof=1)
    rvals = np.asarray(nm.logarithmic_r(0.1 * sd, 0.5 * sd, 1.03))
    orbit = _orbit()
    cs = _nolds_intended_csums(orbit, rvals)
    nz = cs > 0
    ref = float(np.polyfit(np.log(rvals[nz]), np.log(cs[nz]), 1)[0])
    slope, _, status = _gp_correlation_dimension(pdist(orbit), rvals=rvals)
    assert status == "ok"
    assert slope == pytest.approx(ref, abs=1e-6)
    assert ref == pytest.approx(_fixture()["check_b_nolds"]["nolds_intended_slope"], abs=1e-9)
