"""Numeric-parity test: BLME geometry_vendi_score vs the OFFICIAL Vendi
Score library.

The Vendi Score (Friedman & Dieng, TMLR 2023, arXiv:2210.02410; repo
vertaix/Vendi-Score, pip ``vendi_score`` v0.0.3) of order q=1 is

    VS = exp( - Σ_i λ_i log λ_i ),

where λ_1..λ_n are the eigenvalues of K/n and K is an n×n PSD similarity
matrix with unit diagonal. It is the effective number of distinct samples.

OFFICIAL reference (installed and RUN directly; see the companion verify
script $SCRATCH/newtasks/vendi_verify.py):
    package: vendi_score == 0.0.3
    file:    <site-packages>/vendi_score/vendi.py
    fn:      score_K(K, q=1, p=None, normalize=False)  (lines 30-38)
      weight_K(K, p=None) -> K / K.shape[0]              (lines 8-12)
      w = scipy.linalg.eigvalsh(K / n)                   (dense branch)
      entropy_q(w, q=1) -> -(w[w>0] * log(w[w>0])).sum() (lines 22-27)
      return np.exp(entropy)

The shipped default (``kernel="cosine"``) is the released ``score_X``
protocol: rows L2-normalised, K = Xn Xn^T (an uncentred linear Gram), then
``score_K``. It is pinned twice below against the live package:

  * ``_vendi_score(X)`` == ``vendi.score_X(X)`` (the released entry point, so
    the kernel construction is the library's, not BLME's);
  * through ``evaluate()``: the four headline keys == ``vendi.score_X`` on the
    one-point-per-sequence cloud of the first / middle / last layer, and its
    maximum over layers.

The remaining tests feed BLME's own kernel matrix to ``score_K`` (same K, same
algebra, bit-exact) and cover the ``rbf`` option, which is not the default.

The OFFICIAL_* constants below were produced by RUNNING vendi.score_K on
the kernel matrices BLME builds (vendi_score 0.0.3, scipy 1.18.0, numpy 2.4.6).
"""
import numpy as np
import pytest

# OFFICIAL library, run live in these tests. Guarded so a fresh clone / CI
# without vendi_score SKIPS this module instead of aborting collection.
vendi = pytest.importorskip("vendi_score.vendi")

from blme.tasks.geometry.vendi import (
    _vendi_score,
    _cosine_kernel,
    _rbf_kernel,
)

# Bit-exact tolerance for "same kernel, same algebra" parity; a looser
# 1e-9 for the closed-form anchors (eigensolver round-off).
BITEXACT = 1e-12
TOL = 1e-9

# --- OFFICIAL-derived constants (from RUNNING vendi.score_K) ------------
# Random cloud n=8 d=5 seed=0, cosine kernel.
OFFICIAL_COSINE_RANDOM = 3.751352462688749
# Same cloud, RBF kernel gamma=0.5.
OFFICIAL_RBF_RANDOM = 7.123163160295763
# Mid case: 4 clusters x 2 near-duplicate rows (cosine), seed-0 rng chain.
OFFICIAL_COSINE_MID = 3.0033817835431083


def _random_cloud():
    rng = np.random.default_rng(0)
    return rng.standard_normal((8, 5))


def _mid_cloud():
    """Reproduce the verify-script rng chain exactly: the (8,5) draw is
    consumed first, THEN base(4,5) and the 1e-3 jitter(8,5)."""
    rng = np.random.default_rng(0)
    _ = rng.standard_normal((8, 5))          # consumed first (Check A cloud)
    base = rng.standard_normal((4, 5))
    return np.repeat(base, 2, axis=0) + 1e-3 * rng.standard_normal((8, 5))


@pytest.mark.parametrize("shape,seed", [((8, 5), 0), ((40, 12), 1), ((30, 64), 2)])
def test_default_kernel_is_the_released_score_X(shape, seed):
    """The released entry point builds its own kernel from X, so this pins
    BLME's kernel construction (row-normalised, uncentred) and not only the
    eigenvalue algebra."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(shape) + 2.0          # a shared mean direction
    official = float(vendi.score_X(X, q=1, normalize=True))
    assert _vendi_score(X) == pytest.approx(official, rel=1e-10)
    assert _vendi_score(X, kernel="cosine") == _vendi_score(X)
    # Neither the centred cloud nor the unnormalised Gram gives this number.
    assert abs(_vendi_score(X - X.mean(axis=0)) - official) > 1e-2
    assert abs(float(vendi.score_X(X * np.arange(1, shape[0] + 1)[:, None],
                                   q=1, normalize=False)) - official) > 1e-2


def test_evaluate_headline_is_score_X_on_one_point_per_sequence():
    """geometry_vendi_score on injected states with the packaged defaults: each
    headline key is the released score_X on the last row of every document at
    that layer (the first / middle / last block output), and vendi_max the
    largest of the per-layer scores."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import invariance as inv

    L, n_docs, d = 5, 24, 16
    g = np.random.default_rng(7)
    docs = [g.standard_normal((int(t), d)) + 1.5 for t in g.integers(4, 9, size=n_docs)]
    blocks = {}
    for li in range(L):
        blocks[li] = [x.astype(np.float32) for x in docs]
        W = np.eye(d) + 0.6 * g.standard_normal((d, d)) / np.sqrt(d)
        docs = [np.tanh(x @ W) + 0.3 * g.standard_normal(x.shape) for x in docs]
    res = inv.run_task_on_states("geometry_vendi_score", blocks)
    assert res["_meta_n_points"] == n_docs

    per_layer = [float(vendi.score_X(
        np.stack([np.asarray(x, dtype=np.float64)[-1] for x in blocks[li]]),
        q=1, normalize=True)) for li in range(L)]
    assert np.ptp(per_layer) > 1e-2
    assert res["vendi_mean_first_layer"] == pytest.approx(per_layer[0], rel=1e-9)
    assert res["vendi_mean_mid_layer"] == pytest.approx(per_layer[L // 2], rel=1e-9)
    assert res["vendi_mean_last_layer"] == pytest.approx(per_layer[-1], rel=1e-9)
    assert res["vendi_max"] == pytest.approx(max(per_layer), rel=1e-9)
    assert not [k for k in res if "null_ratio" in k]


def test_cosine_parity_vs_official_score_K():
    """BLME cosine _vendi_score == vendi.score_K on the SAME kernel, bit-exact."""
    X = _random_cloud()
    K = _cosine_kernel(X)
    official = vendi.score_K(K, q=1, normalize=False)
    blme = _vendi_score(X, kernel="cosine")
    assert abs(blme - official) < BITEXACT, f"BLME={blme} OFFICIAL={official}"
    assert abs(blme - OFFICIAL_COSINE_RANDOM) < BITEXACT


def test_rbf_parity_vs_official_score_K():
    """BLME rbf _vendi_score == vendi.score_K on the SAME RBF kernel, bit-exact."""
    X = _random_cloud()
    K = _rbf_kernel(X, gamma=0.5)
    official = vendi.score_K(K, q=1, normalize=False)
    blme = _vendi_score(X, kernel="rbf", gamma=0.5)
    assert abs(blme - official) < BITEXACT, f"BLME={blme} OFFICIAL={official}"
    assert abs(blme - OFFICIAL_RBF_RANDOM) < BITEXACT


def test_mid_case_parity_and_range():
    """Controlled mid case: bit-exact parity AND 1 < VS < n."""
    X = _mid_cloud()
    K = _cosine_kernel(X)
    official = vendi.score_K(K, q=1, normalize=False)
    blme = _vendi_score(X, kernel="cosine")
    assert abs(blme - official) < BITEXACT, f"BLME={blme} OFFICIAL={official}"
    assert abs(blme - OFFICIAL_COSINE_MID) < BITEXACT
    assert 1.0 < blme < 8.0


def test_kernel_has_unit_diagonal():
    """Both kernels must have K_ii = 1 (the definition's normalization)."""
    X = _random_cloud()
    assert np.max(np.abs(np.diagonal(_cosine_kernel(X)) - 1.0)) < 1e-12
    assert np.max(np.abs(np.diagonal(_rbf_kernel(X, gamma=0.5)) - 1.0)) < 1e-12


# -------------------- GROUND-TRUTH CLOSED-FORM ANCHORS ------------------

def test_anchor_identical_rows_gives_one():
    """n identical rows -> K all-ones (rank 1) -> VS = 1 exactly."""
    Xid = np.ones((6, 4))
    K = _cosine_kernel(Xid)
    assert np.allclose(K, 1.0)  # all-ones kernel
    vs = _vendi_score(Xid, kernel="cosine")
    assert abs(vs - 1.0) < TOL, f"VS={vs}"
    # official agrees on the same all-ones K
    assert abs(vendi.score_K(K, q=1, normalize=False) - 1.0) < TOL


def test_anchor_orthonormal_rows_gives_n():
    """n mutually orthogonal unit rows -> K = I -> VS = n exactly."""
    n = 7
    Xorth = np.eye(n)  # e_1..e_n are orthonormal, cosine kernel = I
    K = _cosine_kernel(Xorth)
    assert np.allclose(K, np.eye(n))
    vs = _vendi_score(Xorth, kernel="cosine")
    assert abs(vs - n) < TOL, f"VS={vs}"
    assert abs(vendi.score_K(np.eye(n), q=1, normalize=False) - n) < TOL


def test_anchor_score_K_identity_equals_n():
    """Direct: vendi.score_K(I_n) == n and BLME algebra matches for K=I."""
    for n in (5, 20):
        assert abs(vendi.score_K(np.eye(n), q=1, normalize=False) - n) < TOL


def test_score_bounded_between_one_and_n():
    """VS is always in [1, n] for any valid similarity kernel."""
    rng = np.random.default_rng(3)
    for n in (4, 10, 25):
        X = rng.standard_normal((n, 6))
        vs = _vendi_score(X, kernel="cosine")
        assert 1.0 - TOL <= vs <= n + TOL, f"n={n} VS={vs}"


def test_single_sample_is_one():
    assert _vendi_score(np.array([[1.0, 2.0, 3.0]]), kernel="cosine") == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
