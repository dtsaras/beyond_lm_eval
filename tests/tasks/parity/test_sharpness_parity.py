"""Numeric-parity test: BLME dynamics_sharpness vs PyHessian (official).

The canonical loss-landscape "sharpness" used in BLME is the top Hessian
eigenvalue lambda_max(H) of the loss w.r.t. the selected parameters, estimated
by power iteration (Yao et al. 2020, "PyHessian"; Foret et al. 2021, SAM).

OFFICIAL reference (pip-installed `pyhessian`, file
site-packages/pyhessian/hessian.py; package exposes no __version__):

    from pyhessian import hessian
    eigvals, _ = hessian(model, criterion, data=(x, y), cuda=False)\
                     .eigenvalues(maxIter=..., tol=..., top_n=1)
    lambda_max == eigvals[0]

PyHessian.eigenvalues() runs power iteration on the Hessian-vector product
(double backward) and returns the Rayleigh quotient <Hv, v> for the dominant
eigenvalue.

BLME kernel under test (src/blme/tasks/dynamics/sharpness.py):
    - _hvp (line ~69): Hessian-vector product via double backward.
    - _make_random_vec (line ~87): Gaussian init for power iteration.
    - power-iteration loop in compute() (lines ~195-211): normalize v, then
      n_power_iter steps of v <- Hv/||Hv||; reports top_eig = ||Hv|| from the
      final iteration.

We run BOTH on the SAME tiny float64 model + fixed MSE batch + seed, and ALSO
compute the EXACT Hessian (torch.autograd.functional.hessian) and its true
largest eigenvalue (numpy.linalg.eigh) to pin ground truth.

Tolerances (documented):
  - BLME vs EXACT lambda_max: 1e-9 (BLME converges to ground truth at f64
    machine precision with enough power-iteration steps).
  - PyHessian vs EXACT, and BLME vs PyHessian: 1e-5. PyHessian's own accuracy
    floor on this tiny problem is ~2e-6, an artifact of its off-by-one
    Rayleigh-quotient reporting (eigenvalues() returns <Hv,v> computed before
    the final renormalization of v), NOT a BLME/reference divergence.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip("pyhessian")
from pyhessian import hessian  # noqa: E402

# BLME's ACTUAL kernel helpers — imported, not reimplemented.
from blme.tasks.dynamics.sharpness import (  # noqa: E402
    _hvp,
    _make_random_vec,
    _flatten_grads,
)

TOL_EXACT = 1e-9
TOL_REF = 1e-5

# Expected, from the wave1 verification harness (pinned ground truth).
EXACT_LAMBDA_MAX = 2.0562113169633243


def _build():
    """Tiny deterministic 2-layer MLP + fixed MSE batch (float64, seed 0)."""
    torch.manual_seed(0)
    np.random.seed(0)
    model = nn.Sequential(
        nn.Linear(4, 6),
        nn.Tanh(),
        nn.Linear(6, 3),
    ).double()
    x = torch.randn(5, 4, dtype=torch.float64)
    y = torch.randn(5, 3, dtype=torch.float64)
    criterion = nn.MSELoss()
    return model, criterion, x, y


def _exact_lambda_max(model, criterion, x, y):
    """EXACT lambda_max via full Hessian + numpy.linalg.eigh (ground truth)."""
    params = [p for p in model.parameters() if p.requires_grad]
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]
    flat0 = torch.cat([p.detach().reshape(-1) for p in params]).clone()

    def loss_of_flat(flat):
        op, idx = [], 0
        for sh, ne in zip(shapes, numels):
            op.append(flat[idx:idx + ne].reshape(sh))
            idx += ne
        W0, b0, W2, b2 = op
        h = torch.tanh(x @ W0.t() + b0)
        return criterion(h @ W2.t() + b2, y)

    H = torch.autograd.functional.hessian(loss_of_flat, flat0).detach().numpy()
    H = 0.5 * (H + H.T)
    return float(np.linalg.eigh(H)[0][-1])


def _pyhessian_lambda_max(model, criterion, x, y):
    torch.manual_seed(12345)
    hess = hessian(model, criterion, data=(x, y), cuda=False)
    eigvals, _ = hess.eigenvalues(maxIter=2000, tol=1e-9, top_n=1)
    return float(eigvals[0])


def _blme_lambda_max(model, criterion, x, y, n_power_iter=200):
    """Replicate compute()'s top-eigenvalue power iteration using BLME helpers."""
    torch.manual_seed(999)
    params = [p for p in model.parameters() if p.requires_grad]

    def compute_loss():
        return criterion(model(x), y)

    v = _make_random_vec(params, rademacher=False)  # Gaussian init, as BLME
    flat_v = _flatten_grads(v)
    v_norm = float(flat_v.norm().item())
    if v_norm > 0:
        v = [vi / v_norm for vi in v]
    top_eig = 0.0
    for _ in range(n_power_iter):
        loss_fresh = compute_loss()
        hv = _hvp(loss_fresh, params, v)
        flat_hv = _flatten_grads(hv)
        top_eig = float(flat_hv.norm().item())
        if top_eig > 0:
            v = [h / top_eig for h in hv]
        else:
            break
    return top_eig


def test_blme_matches_exact_lambda_max():
    """BLME power iteration converges to the EXACT top Hessian eigenvalue."""
    model, criterion, x, y = _build()
    exact = _exact_lambda_max(model, criterion, x, y)
    # embedded ground truth is reproducible
    assert abs(exact - EXACT_LAMBDA_MAX) < TOL_EXACT, exact

    blme = _blme_lambda_max(model, criterion, x, y)
    rel = abs(blme - exact) / abs(exact)
    assert rel < TOL_EXACT, f"BLME={blme} EXACT={exact} rel={rel:.3e}"


def test_pyhessian_matches_exact_lambda_max():
    """Official PyHessian converges to the EXACT top Hessian eigenvalue."""
    model, criterion, x, y = _build()
    exact = _exact_lambda_max(model, criterion, x, y)
    py = _pyhessian_lambda_max(model, criterion, x, y)
    rel = abs(py - exact) / abs(exact)
    assert rel < TOL_REF, f"PyHessian={py} EXACT={exact} rel={rel:.3e}"


def test_blme_matches_pyhessian():
    """BLME's sharpness == PyHessian's eigenvalues(top_n=1) on same model/loss/data."""
    model, criterion, x, y = _build()
    py = _pyhessian_lambda_max(model, criterion, x, y)
    blme = _blme_lambda_max(model, criterion, x, y)
    rel = abs(blme - py) / abs(py)
    assert rel < TOL_REF, f"BLME={blme} PyHessian={py} rel={rel:.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
