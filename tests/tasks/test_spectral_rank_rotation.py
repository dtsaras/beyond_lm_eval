"""alpha is a function of the SPECTRUM, so orthogonal conjugation cannot move it.

Exact positivity (`lam > 0`) cannot separate a true small eigenvalue from SVD
backward-error residue. For H = hadamard(64)/8 and D = diag(8,...,1, 0 x 56),
H D H^T has the same eigenvalues as D by construction, but its computed 9th
singular value is ~4.59e-15 rather than 0 -- so 64 numerical positives replaced
the true 8 and flipped the released small-N estimator branch.
"""
import numpy as np
import pytest
import torch
from scipy.linalg import hadamard

from blme.tasks.geometry.spectral import _pl_mle_alpha_ks
from blme.tasks.geometry.utils import safe_svdvals

_H = hadamard(64) / 8.0


def _alpha(W):
    """The shipped boundary: relative rank tolerance, then square."""
    S = np.asarray(safe_svdvals(torch.from_numpy(np.asarray(W, dtype=np.float64))),
                   dtype=np.float64)
    tol = float(np.finfo(np.float64).eps) * max(W.shape) * float(S[0])
    return _pl_mle_alpha_ks(S[S > tol] ** 2)[0]


def _rank_k(k):
    d = np.zeros(64)
    d[:k] = np.arange(k, 0, -1)
    return np.diag(d)


@pytest.mark.parametrize("rank", [2, 3, 8, 19, 20, 32, 48, 63, 64])
def test_orthogonal_conjugation_preserves_alpha(rank):
    D = _rank_k(rank)
    W = _H @ D @ _H.T
    # FIXTURE INTEGRITY: the conjugation is exactly invertible.
    assert np.allclose(_H.T @ W @ _H, D, atol=0, rtol=0) or \
        np.abs(_H.T @ W @ _H - D).max() < 1e-14
    assert _alpha(W) == pytest.approx(_alpha(D), rel=1e-9), (
        f"alpha moved under an exact orthogonal conjugation at rank {rank}")


def test_rank_8_matches_released_weightwatcher():
    """The released full analyzer returns 1.330045361632 on BOTH matrices."""
    D = _rank_k(8)
    assert _alpha(D) == pytest.approx(1.330045361632, rel=1e-11)
    assert _alpha(_H @ D @ _H.T) == pytest.approx(1.330045361632, rel=1e-11)


def test_signed_permutations_are_stable():
    """All 32 signed coordinate permutations give one value, not a range."""
    D = _rank_k(8)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(32):
        sg = np.where(rng.random(64) < 0.5, -1.0, 1.0)
        Hs = _H * sg[None, :]
        vals.append(_alpha(Hs @ D @ Hs.T))
    assert max(vals) - min(vals) < 1e-11, f"range {min(vals)}..{max(vals)}"
    assert vals[0] == pytest.approx(1.330045361632, rel=1e-11)


def test_tolerance_is_scale_invariant():
    """A relative envelope must not reintroduce the absolute floor's bug."""
    W = _H @ _rank_k(8) @ _H.T
    base = _alpha(W)
    for scale in (1e-23, 1e-8, 1.0, 1e8, 1e23):
        assert _alpha(W * scale) == pytest.approx(base, rel=1e-11), (
            f"alpha moved at scale {scale}")
