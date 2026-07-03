"""Numeric-parity test: BLME geometry_hubness vs scikit-hubness.

The canonical "hubness skewness" S_k is the skewness of the k-occurrence
distribution N_k (Radovanovic et al. 2010 JMLR; Tomasev et al. 2014 TKDE).

OFFICIAL reference (transcribed verbatim, provenance below):
    repo:   https://github.com/VarIr/scikit-hubness
    commit: c36a058e67c34696182d3b8c5da089e287eb60bf
    file:   skhubness/analysis/estimation.py
      - _k_neighbors (line 231): k neighbors per point with the self hit
        filtered (start=1, end=k+1 when querying within the indexed set).
      - line 434:  k_occurrence = np.bincount(k_neighbors.ravel(), minlength=n)
      - line 442:  hubness_measures["k_skewness"] = stats.skew(k_occurrence)

    i.e. S_k == scipy.stats.skew(k_occurrence), with k_occurrence the
    self-excluded k-NN occurrence-count vector. That is exactly what BLME's
    `_hubness_stats_from_occurrences` computes via `scipy.stats.skew`.

The toy k_occurrence vector below is produced (and embedded) by the wave1
verification harness using sklearn NearestNeighbors on 60 points in R^8
(seed 0, k=5, two deliberate hubs at indices 0 and 1).
"""
import numpy as np
import pytest
from scipy import stats

from blme.tasks.geometry.hubness import (
    _hubness_stats_from_occurrences,
    _gini_from_counts,
)

# --- Toy self-excluded k-occurrence vector (n=60, d=8, k=5, seed=0) ---------
# Reproduced from the wave1 harness; sum == n*k == 300.
K = 5
K_OCCURRENCE = np.array(
    [26, 25, 2, 0, 2, 1, 7, 7, 6, 6, 1, 5, 4, 1, 5, 3, 12, 7, 1, 3, 2, 8, 2,
     1, 3, 2, 10, 1, 6, 2, 5, 9, 6, 0, 1, 6, 2, 3, 3, 18, 1, 1, 0, 8, 7, 13,
     13, 1, 3, 5, 1, 4, 0, 0, 7, 10, 6, 3, 1, 2],
    dtype=np.int64,
)

# OFFICIAL expected values (scikit-hubness line 442 + independent Gini).
OFFICIAL_K_SKEWNESS = 2.1327252317535574
OFFICIAL_GINI = 0.5146666666666667
TOL = 1e-9


def _gini_independent(x):
    """Clean mean-absolute-difference Gini (independent of BLME's impl)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    mad = np.abs(x.reshape(-1, 1) - x.reshape(1, -1)).sum()
    return float(mad / (2.0 * n * x.sum()))


def test_sum_invariant():
    # Sanity: self-excluded k-occurrence must sum to n*k.
    assert K_OCCURRENCE.sum() == K_OCCURRENCE.size * K


def test_hubness_skewness_parity():
    """BLME S_k == scikit-hubness stats.skew(k_occurrence)."""
    official = float(stats.skew(K_OCCURRENCE))
    # guard: embedded constant matches a freshly recomputed scipy.skew
    assert abs(official - OFFICIAL_K_SKEWNESS) < TOL

    blme = _hubness_stats_from_occurrences(K_OCCURRENCE, K)[f"hubness_k{K}_skew"]
    assert abs(blme - official) < TOL, f"BLME={blme} OFFICIAL={official}"


def test_hubness_gini_parity():
    """BLME Gini == independent mean-absolute-difference Gini."""
    official = _gini_independent(K_OCCURRENCE)
    assert abs(official - OFFICIAL_GINI) < TOL

    blme = _hubness_stats_from_occurrences(K_OCCURRENCE, K)[f"hubness_k{K}_gini"]
    assert abs(blme - official) < TOL, f"BLME={blme} OFFICIAL={official}"
    # the standalone helper agrees too
    assert abs(_gini_from_counts(K_OCCURRENCE) - official) < TOL


def test_zero_variance_skew_is_zero():
    """BLME's documented special-case: constant occurrences -> skew 0."""
    flat = np.full(20, 3, dtype=np.int64)
    out = _hubness_stats_from_occurrences(flat, K)
    assert out[f"hubness_k{K}_skew"] == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
