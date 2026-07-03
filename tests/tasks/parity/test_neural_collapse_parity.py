"""Numeric-parity test: BLME geometry_neural_collapse vs official NC library.

Neural Collapse NC1 (within-class variability collapse), Papyan, Han & Donoho
2020 PNAS (arXiv:2008.08186):

    NC1 = tr( Sigma_W @ Sigma_B^+ ) / K

with Sigma_W the within-class covariance (normalized by N_total), Sigma_B the
between-class covariance (M_centred^T M_centred / K), and (.)^+ the Moore-Penrose
pseudo-inverse.

OFFICIAL reference (run directly in the wave1 harness; transcribed verbatim here):
    repo:   https://github.com/rhubarbwu/neural-collapse
    commit: c05a0b80d6bc6e8e2e102f2180f1f33b2c9605fd
    file:   neural_collapse/measure.py
      - covariance_ratio (lines 16-54). For metric="pinv" (Papyan et al. 2020),
        lines 37-43:
            (K, _), M_centred = M.shape, M - m_G
            V_inter = M_centred.mT @ M_centred / K
            prod    = la.pinv(V_inter) @ V_intra
            return  pt.trace(prod).item() / K
      - Sigma_W (V_intra) comes from accumulate.py::CovarAccumulator.compute
        (lines 153-159): totals / ns_samples.sum()  -> /N_total, matching BLME's
        `sigma_w /= n`.
      - Global mean m_G from accumulate.py::Accumulator.compute (weighted=False,
        lines 82-87): avg.mean(dim=0) == unweighted mean of class means; equals
        the data mean for balanced classes (used here).

trace is order-invariant, so tr(pinv(Sb) @ Sw) == tr(Sw @ pinv(Sb)); the BLME
implementation computes the same scalar via a top-(K-1) eigenvalue subspace
restriction of Sb, which on well-conditioned synthetic data is numerically
identical to the full Moore-Penrose pinv (and to the reference's "svd" metric).

The toy features (K=3 tight Gaussian blobs in R^5, 30 pts/class, seed 0,
spread 0.15) and the OFFICIAL expected NC1 are reproduced from the wave1
harness, which RAN the rhubarbwu library directly.

NC2-ETF caveat: BLME's `nc2_etf_cosine_deviation_proxy` is an explicitly
documented PROXY (mean |cos - (-1/(K-1))| over centered class-mean pairs), a
DIFFERENT definition from the reference `simplex_etf_error` (Frobenius norm of
the normalized cross-class Gram vs the ETF target, Kothapalli 2023). It is not
numerically comparable and is verified only against its own stated definition.
"""
import numpy as np
import pytest

from blme.tasks.geometry.neural_collapse import _neural_collapse_metrics

# --- Toy data: K=3 Gaussian blobs in R^5, 30 pts/class, balanced, seed 0 ----
# Identical construction to the wave1 harness (make_toy).
SEED, K, N_PER, D, SPREAD = 0, 3, 30, 5, 0.15
TOL = 1e-9

# OFFICIAL NC1 from the rhubarbwu library, metric="pinv" (Papyan et al. 2020),
# computed in the wave1 harness on the toy features below.
OFFICIAL_NC1_PINV = 0.00409094387203806
# metric="svd" (Han et al. 2022) agreed to machine epsilon.
OFFICIAL_NC1_SVD = 0.004090943872038059
# BLME's own stated NC2 metrics on this toy (verified against their definitions).
BLME_NC2_EQUINORM_CV = 0.00784386467284082
BLME_NC2_ETF_PROXY = 0.011144638743029825


def make_toy():
    rng = np.random.default_rng(SEED)
    centers = np.array(
        [
            [3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )[:K]
    feats, labs = [], []
    for k in range(K):
        feats.append(centers[k] + SPREAD * rng.standard_normal((N_PER, D)))
        labs.append(np.full(N_PER, k, dtype=np.int64))
    return np.concatenate(feats, 0).astype(np.float64), np.concatenate(labs, 0)


def reference_nc1_pinv(X, y):
    """Independent reimplementation of covariance_ratio(metric='pinv').

    Verbatim algebra from neural_collapse/measure.py lines 37-43 plus the
    Sigma_W normalization (/N_total) from CovarAccumulator.compute and the
    unweighted global mean from Accumulator.compute(weighted=False).
    """
    classes = np.unique(y)
    Kc = len(classes)
    n, d = X.shape
    mus = np.stack([X[y == c].mean(axis=0) for c in classes])  # (K,D)
    m_G = mus.mean(axis=0)  # unweighted mean of class means (== data mean, balanced)
    # V_intra (Sigma_W): sum of within-class outer products / N_total
    V_intra = np.zeros((d, d))
    for ci, c in enumerate(classes):
        diff = X[y == c] - mus[ci]
        V_intra += diff.T @ diff
    V_intra /= n
    M_centred = mus - m_G
    V_inter = M_centred.T @ M_centred / Kc  # Sigma_B
    prod = np.linalg.pinv(V_inter) @ V_intra
    return float(np.trace(prod) / Kc)


def test_toy_is_balanced_and_separated():
    X, y = make_toy()
    assert X.shape == (K * N_PER, D)
    # balanced => unweighted mean of class means == data mean (to fp eps)
    mus = np.stack([X[y == c].mean(axis=0) for c in np.unique(y)])
    assert np.max(np.abs(mus.mean(axis=0) - X.mean(axis=0))) < 1e-12


def test_reference_constant_matches_independent_recompute():
    """Guard: embedded OFFICIAL_NC1_PINV == freshly recomputed reference algebra."""
    X, y = make_toy()
    assert abs(reference_nc1_pinv(X, y) - OFFICIAL_NC1_PINV) < TOL


def test_nc1_parity_blme_vs_reference():
    """BLME NC1 == official covariance_ratio(metric='pinv') to <1e-9."""
    X, y = make_toy()
    official = reference_nc1_pinv(X, y)
    blme = _neural_collapse_metrics(X, y)["nc1_within_class_collapse"]
    assert abs(blme - official) < TOL, f"BLME={blme} OFFICIAL={official}"
    # also matches the embedded official constants (pinv and svd variants)
    assert abs(blme - OFFICIAL_NC1_PINV) < TOL
    assert abs(blme - OFFICIAL_NC1_SVD) < TOL


def test_nc1_subspace_rank_is_K_minus_1():
    """Sigma_B has exactly rank K-1 here, so BLME's truncation keeps K-1 dirs."""
    X, y = make_toy()
    out = _neural_collapse_metrics(X, y)
    assert out["nc1_subspace_rank"] == K - 1


def test_nc1_trace_order_invariance():
    """tr(Sw @ pinv(Sb)) == tr(pinv(Sb) @ Sw): BLME's ordering is equivalent."""
    X, y = make_toy()
    classes = np.unique(y)
    n, d = X.shape
    mus = np.stack([X[y == c].mean(axis=0) for c in classes])
    m_G = X.mean(axis=0)
    Sw = np.zeros((d, d))
    for ci, c in enumerate(classes):
        diff = X[y == c] - mus[ci]
        Sw += diff.T @ diff
    Sw /= n
    Mc = mus - m_G
    Sb = Mc.T @ Mc / K
    a = np.trace(Sw @ np.linalg.pinv(Sb)) / K
    b = np.trace(np.linalg.pinv(Sb) @ Sw) / K
    assert abs(a - b) < TOL


def test_nc2_proxies_match_stated_definitions():
    """NC2 metrics are PROXIES verified against their own stated definitions."""
    X, y = make_toy()
    out = _neural_collapse_metrics(X, y)

    classes = np.unique(y)
    Kc = len(classes)
    mus = np.stack([X[y == c].mean(axis=0) for c in classes])
    M = mus - X.mean(axis=0)

    # NC2-equinorm: CV of centered class-mean norms (std/mean).
    norms = np.linalg.norm(M, axis=1)
    equinorm_cv = float(np.std(norms) / norms.mean())
    assert abs(out["nc2_equinorm_cv"] - equinorm_cv) < TOL
    assert abs(out["nc2_equinorm_cv"] - BLME_NC2_EQUINORM_CV) < TOL

    # NC2-ETF proxy: mean |cos - (-1/(K-1))| over centered class-mean pairs.
    Mu = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    cos = Mu @ Mu.T
    iu = np.triu_indices(Kc, k=1)
    target = -1.0 / (Kc - 1)
    etf_proxy = float(np.mean(np.abs(cos[iu] - target)))
    assert abs(out["nc2_etf_cosine_deviation_proxy"] - etf_proxy) < TOL
    assert abs(out["nc2_etf_cosine_deviation_proxy"] - BLME_NC2_ETF_PROXY) < TOL


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
