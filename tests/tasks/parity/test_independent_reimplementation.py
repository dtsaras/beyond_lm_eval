"""Independent REIMPLEMENTATIONS of three widely-used references.

Most round-13 Opus verdicts rest on executed parity -- running the reference
checks that already exist. That inherits whatever those tests fail to ask. The
six coverage-gap tasks were instead checked against references derived from the
cited definitions, and that immediately caught a real trap (GUDHI's landscape
is sqrt(2) x Bubenik's). These three do the same for the most widely-used
kernels in the suite.
"""
import numpy as np
import pytest
import torch


# --------------------------------------------------------------------------
# Linear CKA — Kornblith et al. 2019, Eq. 2
# --------------------------------------------------------------------------
def _ref_linear_cka(X, Y):
    """CKA(X,Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F), columns centred."""
    X = np.asarray(X, dtype=np.float64) - np.asarray(X, dtype=np.float64).mean(0)
    Y = np.asarray(Y, dtype=np.float64) - np.asarray(Y, dtype=np.float64).mean(0)
    num = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    den = (np.linalg.norm(X.T @ X, ord="fro")
           * np.linalg.norm(Y.T @ Y, ord="fro"))
    return float(num / den) if den > 0 else float("nan")


def test_linear_cka_matches_kornblith_definition():
    from blme.tasks.geometry.cka import _linear_cka
    rng = np.random.default_rng(0)
    for _ in range(20):
        n, d1, d2 = int(rng.integers(20, 60)), int(rng.integers(4, 24)), int(rng.integers(4, 24))
        X, Y = rng.standard_normal((n, d1)), rng.standard_normal((n, d2))
        got = _linear_cka(torch.from_numpy(X), torch.from_numpy(Y))
        got = float(got) if not isinstance(got, tuple) else float(got[0])
        # 1e-6, and deliberately not tighter. cka.py narrows to float32 inside
        # `_linear_cka` (M = M.float()), by design: float32 and float64 INPUTS
        # return the identical value, and it agrees with this exact float64
        # closed form to ~7e-8 relative -- float32 precision. Promoting it would
        # be a REGRESSION against a bit-pinned reference, which is this repo's
        # documented rule, so the tolerance records the precision instead.
        assert got == pytest.approx(_ref_linear_cka(X, Y), rel=1e-6)


def test_linear_cka_invariants():
    """CKA is 1 on identical inputs, invariant to orthogonal maps and to scale."""
    from blme.tasks.geometry.cka import _linear_cka
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 12))
    f = lambda A, B: float(_linear_cka(torch.from_numpy(A), torch.from_numpy(B)))
    assert f(X, X) == pytest.approx(1.0, rel=1e-9)
    Q, _ = np.linalg.qr(rng.standard_normal((12, 12)))
    assert f(X, X @ Q) == pytest.approx(1.0, rel=1e-6), "not orthogonally invariant"
    Y = rng.standard_normal((50, 9))
    base = f(X, Y)
    for s in (1e-6, 1e3, 1e6):
        assert f(X * s, Y) == pytest.approx(base, rel=1e-6), f"scale {s} moved CKA"


# --------------------------------------------------------------------------
# Debiased HSIC — Song et al. 2012, the unbiased estimator
# --------------------------------------------------------------------------
def _ref_debiased_hsic(K, L):
    """Unbiased HSIC (Song et al. 2012 Eq. 5), zero diagonals."""
    K = np.array(K, dtype=np.float64); L = np.array(L, dtype=np.float64)
    n = K.shape[0]
    np.fill_diagonal(K, 0.0); np.fill_diagonal(L, 0.0)
    t1 = float((K * L).sum())
    t2 = float(K.sum() * L.sum()) / ((n - 1) * (n - 2))
    t3 = 2.0 * float(K.sum(0) @ L.sum(0)) / (n - 2)
    return (t1 + t2 - t3) / (n * (n - 3))


def test_debiased_hsic_matches_song_estimator():
    from blme.tasks.geometry.cka import _debiased_hsic
    rng = np.random.default_rng(0)
    for _ in range(15):
        n = int(rng.integers(8, 40))
        A, B = rng.standard_normal((n, 6)), rng.standard_normal((n, 6))
        K, L = A @ A.T, B @ B.T
        got = float(_debiased_hsic(torch.from_numpy(K.copy()),
                                   torch.from_numpy(L.copy())))
        assert got == pytest.approx(_ref_debiased_hsic(K, L), rel=1e-8)


def test_debiased_hsic_is_unbiased_under_independence():
    """The debiased estimator must average to ~0 for independent inputs."""
    from blme.tasks.geometry.cka import _debiased_hsic
    rng = np.random.default_rng(3)
    vals = []
    for _ in range(200):
        A, B = rng.standard_normal((30, 4)), rng.standard_normal((30, 4))
        K, L = A @ A.T, B @ B.T
        vals.append(float(_debiased_hsic(torch.from_numpy(K), torch.from_numpy(L))))
    m = float(np.mean(vals))
    se = float(np.std(vals) / np.sqrt(len(vals)))
    assert abs(m) < 4 * se, f"mean {m:.4g} is {abs(m)/se:.1f} SE from 0 -- biased"


# --------------------------------------------------------------------------
# IsoScore — Rudman et al. 2022, Algorithm 1
# --------------------------------------------------------------------------
def _ref_isoscore(X):
    """IsoScore per Rudman et al. 2022, Alg. 1, implemented from the paper."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    Xc = X - X.mean(0)                                  # 1. mean-centre
    C = np.cov(Xc, rowvar=False, bias=False)            # 2. covariance
    # 3. The DIAGONAL IS TAKEN IN THE PCA BASIS, i.e. it is the eigenvalue
    # spectrum of the covariance -- not the raw diagonal of C. Omitting the
    # reorientation is what made my first reference disagree with BLME by 2.5%;
    # BLME was right and the reference was incomplete.
    diag = np.sort(np.linalg.eigvalsh(C))[::-1]
    nd = np.linalg.norm(diag)
    if nd == 0:
        return 0.0
    dn = diag * np.sqrt(d) / nd                         # 4. normalise to length sqrt(d)
    delta = np.linalg.norm(dn - np.ones(d)) / np.sqrt(2 * (d - np.sqrt(d)))  # 5. distance
    phi = (d - delta ** 2 * (d - np.sqrt(d))) ** 2      # 6. isotropy defect -> dims used
    phi = phi / (d ** 2)
    return float((phi * d - 1) / (d - 1))               # 7. rescale to [0,1]


def test_isoscore_matches_rudman_algorithm():
    from blme.tasks.geometry.isotropy import _isoscore
    rng = np.random.default_rng(0)
    for _ in range(15):
        n, d = int(rng.integers(60, 200)), int(rng.integers(3, 16))
        X = rng.standard_normal((n, d)) * rng.uniform(0.2, 3.0, size=d)
        assert _isoscore(X) == pytest.approx(_ref_isoscore(X), rel=1e-6)


def test_isoscore_endpoints():
    """Isotropic Gaussian -> near 1; a single dominant axis -> near 0."""
    from blme.tasks.geometry.isotropy import _isoscore
    rng = np.random.default_rng(2)
    iso = _isoscore(rng.standard_normal((4000, 8)))
    assert iso > 0.85, f"isotropic cloud scored {iso}"
    X = rng.standard_normal((4000, 8)) * np.array([1e3] + [1e-3] * 7)
    assert _isoscore(X) < 0.10, f"degenerate cloud scored {_isoscore(X)}"
