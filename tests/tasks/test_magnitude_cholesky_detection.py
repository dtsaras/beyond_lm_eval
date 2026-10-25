"""magnitude must use magnipy's Cholesky failure test, not LU.

`np.linalg.solve` factorises any nonsingular matrix, so it raises only when
zeta is EXACTLY singular. magnipy's `weights_cholesky` uses `cho_factor`,
which fails whenever zeta is not positive definite -- and `perturb_singularities`
then adds 0.01*I and retries. A zeta that is numerically indefinite but
LU-invertible therefore used to return from an INVALID solve instead of
reaching the perturbation.
"""
import numpy as np
import pytest
from scipy.linalg import cho_factor

from blme.tasks.geometry.magnitude import _magnitude, _pairwise_distance_matrix


def _near_duplicate_cloud():
    """100x768 with 4 near-duplicate groups, separated just below the ULP band.

    This is the regime the real GPT-2 activation cloud reaches: cond(zeta)
    ~1e17, Cholesky pivots go negative on rounding, LU proceeds regardless.
    """
    rng = np.random.default_rng(2)
    X = rng.standard_normal((100, 768))
    off = np.random.default_rng(99)
    for a, b in ((5, 6), (5, 61), (6, 61), (11, 65), (44, 59), (85, 90)):
        X[b] = X[a] + off.standard_normal(768) * 2.16e-17
    return X


def test_indefinite_zeta_reaches_the_perturbation_fallback():
    X = _near_duplicate_cloud()
    D = _pairwise_distance_matrix(X)
    t = 0.1
    Z = np.exp(-t * D)

    # FIXTURE INTEGRITY: we are in the target regime -- Cholesky fails and LU
    # does not. If this stops holding the test below proves nothing.
    with pytest.raises(Exception):
        cho_factor(Z)
    w_lu = np.linalg.solve(Z, np.ones(100))       # the old primary path
    residual = float(np.abs(Z @ w_lu - 1).max())
    assert residual > 1.0, (
        f"fixture no longer produces an invalid LU solve (residual={residual})")
    # The old path returned a NEGATIVE magnitude, which is impossible for a
    # finite metric space at t > 0.
    assert float(w_lu.sum()) < 0.0

    got = _magnitude(D, t)
    # magnipy: weights_cholesky(Z) raises -> weights_cholesky(Z + 0.01*I).
    from scipy.linalg import solve_triangular
    Zp = Z + 0.01 * np.eye(100)
    c, _ = cho_factor(Zp)
    ones = np.ones(100)
    ref = float(solve_triangular(c, solve_triangular(c, ones, trans=1).T,
                                 trans=0).sum())
    assert got == pytest.approx(ref, rel=1e-12), f"{got!r} != reference {ref!r}"
    assert got == pytest.approx(32.758578926947735, rel=1e-9)
    assert got > 0.0


def test_positive_definite_zeta_is_unaffected():
    """Where Cholesky succeeds the fix must not move the answer."""
    X = _near_duplicate_cloud()
    D = _pairwise_distance_matrix(X)
    Z = np.exp(-0.5 * D)
    cho_factor(Z)                                  # in regime: PD, no raise
    assert _magnitude(D, 0.5) == pytest.approx(94.9999664369684, rel=1e-12)


@pytest.mark.parametrize("pts,pinned", [
    ([[0.], [1.]], 1.4621171572600098),
    ([[0.], [0.], [1.]], 1.4541545040460961),
])
def test_pinned_magnipy_parity_values_unchanged(pts, pinned):
    """The released duplicate policy is preserved bit-for-bit."""
    D = _pairwise_distance_matrix(np.array(pts))
    assert _magnitude(D, 1.0) == pinned
