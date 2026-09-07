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
import torch
from scipy import stats

from blme.tasks.geometry.hubness import (
    GlobalHubnessTask,
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


def test_zero_variance_skew_is_nan_like_scipy():
    """A CONSTANT occurrence vector has no skew: S_Nk is the standardized third
    moment, so it is 0/0 there, and `scipy.stats.skew` returns NaN.

    BLME used to special-case this to 0.0, which asserts "perfectly symmetric"
    -- a measurement the data does not support and one that is indistinguishable
    from a genuine symmetric distribution (adversarial review round 2,
    2026-09-04). It now matches scipy.
    """
    from scipy.stats import skew as _scipy_skew
    flat = np.full(20, 3, dtype=np.int64)
    out = _hubness_stats_from_occurrences(flat, K)
    assert np.isnan(out[f"hubness_k{K}_skew"])
    with np.errstate(invalid="ignore"):
        assert np.isnan(float(_scipy_skew(flat)))



# ---------------------------------------------------------------------------
# Regression: neighbour ranking under a large common offset.
#
# Codex round 9 (2026-09-07) flagged that no test drove `evaluate()` on a
# vocabulary sharing a strong common direction, so the suite was structurally
# unable to see the defect it found. It was right.
#
# Cosine similarity is NOT translation invariant, so a common offset is a
# genuinely different measurement -- the bug was never "the answer changed",
# it was that the float32 DOT PRODUCT could no longer resolve the answer.
# With every unit row nearly parallel, each u.v sits just below 1 and the
# separating quantity 1 - u.v is obtained by cancelling two ~1.0 sums. At
# offset 1e3 that left 41 distinct off-diagonal similarities out of 32888 and
# put 212/257 occupancy bins wrong. Ranking direct chord distances instead
# differences the coordinates BEFORE squaring, which is exact for nearby
# floats, so the common component cancels harmlessly.
# ---------------------------------------------------------------------------

def _reference_occurrences(X, k):
    """Independent float64 k-NN occurrence counts by direct chord distance.

    Deliberately shares no code with the task: an explicit per-row loop, and
    `lexsort` for the (distance, index) stable tie rule.
    """
    U = np.asarray(X, dtype=np.float64)
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    n = len(U)
    idx = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        d = ((U - U[i]) ** 2).sum(axis=1)
        d[i] = np.inf                      # self-excluded
        idx[i] = np.lexsort((np.arange(n), d))[:k]
    return np.bincount(idx.ravel(), minlength=n).astype(np.int64)


class _EmbeddingOnly(torch.nn.Module):
    def __init__(self, X, dtype):
        super().__init__()
        self.emb = torch.nn.Embedding(*X.shape)
        self.emb.weight.data = torch.from_numpy(X).to(dtype)

    def get_input_embeddings(self):
        return self.emb


@pytest.mark.parametrize("offset", [0.0, 1e1, 1e3, 1e6, 1e8])
def test_evaluate_matches_reference_under_common_offset(offset):
    """`evaluate()` must equal the float64 reference at every common offset.

    The reference is built from the float32-ROUNDED input, because that is what
    the task is actually handed. Comparing against the un-rounded float64 array
    would charge the task for precision the checkpoint never contained: at
    offset 1e6 that phantom gap is 3.09e-02 in skew, and at 1e8 it is 2.92 --
    entirely storage, nothing the ranking can recover.
    """
    from scipy.stats import skew as _scipy_skew

    rng = np.random.default_rng(20260907)
    X = rng.standard_normal((257, 24)) + offset

    model = _EmbeddingOnly(X, torch.float32)
    got = GlobalHubnessTask({"k_values": [K], "batch_size": 2000}).evaluate(
        model, None, None)

    seen = torch.from_numpy(X).to(torch.float32).numpy().astype(np.float64)
    expected = _reference_occurrences(seen, K)

    assert got[f"hubness_k{K}_skew"] == pytest.approx(
        float(_scipy_skew(expected)), abs=1e-12), (
        f"offset={offset:g}: neighbour ranking lost resolution")
    assert got[f"hubness_k{K}_max"] == int(expected.max())
    assert got[f"hubness_k{K}_gini"] == pytest.approx(
        _gini_from_counts(expected), abs=1e-12)


def test_offset_fixture_would_catch_a_dot_product_ranking():
    """Guard on the guard: the offset fixture must actually be discriminating.

    If someone reverts to normalized-dot-product ranking, the test above has to
    fail. That only holds if the fixture is genuinely ill-conditioned for it, so
    assert the condition directly rather than trusting the constant 1e3.
    """
    rng = np.random.default_rng(20260907)
    X = rng.standard_normal((257, 24)) + 1e3
    E = torch.from_numpy(X).to(torch.float32)
    E = E / E.abs().amax(dim=1, keepdim=True)
    E = E / E.norm(dim=1, keepdim=True)
    sims = (E @ E.T).numpy()
    off = sims[~np.eye(len(X), dtype=bool)]
    # float32 dot products collapse this fixture to a handful of levels.
    assert len(np.unique(off)) < 200, (
        f"fixture no longer ill-conditioned for dot-product ranking "
        f"({len(np.unique(off))} distinct similarities); it can no longer "
        f"detect a regression")
    # and the collapse must actually corrupt the ranking, not merely the values
    dot_occ = np.bincount(
        torch.argsort(torch.from_numpy(sims) - torch.eye(len(X)) * 1e9,
                      dim=1, descending=True, stable=True)[:, :K]
        .reshape(-1).numpy(), minlength=len(X))
    seen = E.numpy().astype(np.float64)
    assert int((dot_occ != _reference_occurrences(seen, K)).sum()) > 100


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
