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

BLME kernel under test (src/blme/tasks/dynamics/sharpness.py) — the REAL
module-level helpers evaluate() calls, imported (not transcribed):
    - _hvp: Hessian-vector product via double backward.
    - _make_random_vec: Gaussian init for power iteration.
    - _power_iteration_top_eigenvalue: power iteration returning the SIGNED
      Rayleigh quotient <Hv, v> at the converged eigenvector, matching
      PyHessian's eigenvalues(top_n=1). (The old code reported ||Hv||,
      which drops the sign; the negative-dominant test below pins the fix.)
    - _sam_sharpness: SAM delta L(theta + rho*g/||g||) - L(theta) with
      exact parameter restore (Foret et al. 2021, Eq. 2, p=2).

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
    _HessianOperator,
    _hutchinson_trace,
    _hvp,
    _make_random_vec,
    _flatten_grads,
    _power_iteration_top_eigenvalue,
    _sam_sharpness,
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


def _blme_lambda_max(model, criterion, x, y, n_power_iter=200, tol=0.0, seed=999,
                     status=None):
    """Drive the REAL evaluate() kernel: _power_iteration_top_eigenvalue.

    tol=0 runs the full budget (convergence of the kernel itself); a positive
    tol applies PyHessian's stopping rule.
    """
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]

    def compute_loss():
        return criterion(model(x), y)

    # PyHessian's own draw: float32 torch.randn(p.size()) per parameter, global RNG.
    v = [torch.randn(p.size()).to(p.dtype) for p in params]
    return _power_iteration_top_eigenvalue(
        _HessianOperator(compute_loss, params), v, n_power_iter, status=status, tol=tol
    )


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
    """BLME's sharpness == PyHessian's eigenvalues(top_n=1) on same model/loss/data,
    under the same stopping rule (maxIter 2000, tol 1e-9)."""
    model, criterion, x, y = _build()
    py = _pyhessian_lambda_max(model, criterion, x, y)
    blme = _blme_lambda_max(model, criterion, x, y, n_power_iter=2000, tol=1e-9)
    rel = abs(blme - py) / abs(py)
    assert rel < TOL_REF, f"BLME={blme} PyHessian={py} rel={rel:.3e}"


def test_default_loop_matches_pyhessian_defaults_from_the_same_start():
    """eigenvalues(maxIter=100, tol=1e-3) vs BLME's defaults, same start vector:
    the same loop up to PyHessian's +1e-6 in its normalisation."""
    model, criterion, x, y = _build()
    torch.manual_seed(7)
    hess = hessian(model, criterion, data=(x, y), cuda=False)
    py = float(hess.eigenvalues(maxIter=100, tol=1e-3, top_n=1)[0][0])
    status = {}
    blme = _blme_lambda_max(model, criterion, x, y, n_power_iter=100, tol=1e-3, seed=7,
                            status=status)
    assert abs(blme - py) / abs(py) < TOL_REF, (blme, py)
    assert status["converged"] and status["n_iter"] < 100
    assert abs(blme - EXACT_LAMBDA_MAX) / EXACT_LAMBDA_MAX < 1e-2


def test_one_hvp_per_iteration():
    """PyHessian's cost: one HVP per power iteration and per trace probe."""
    model, criterion, x, y = _build()
    params = list(model.parameters())
    H = _HessianOperator(lambda: criterion(model(x), y), params)
    calls = []

    def counted(v):
        calls.append(1)
        return H(v)

    st = {}
    _power_iteration_top_eigenvalue(counted, _make_random_vec(params, rademacher=False),
                                    100, status=st, tol=1e-3)
    assert len(calls) == st["n_iter"]
    calls.clear()
    tr = {}
    samples = _hutchinson_trace(counted, params, n_iter=100, tol=1e-3, status=tr)
    assert len(calls) == len(samples) == tr["n_iter"]


def test_hessian_operator_matches_fresh_graph_hvp():
    model, criterion, x, y = _build()
    params = list(model.parameters())
    H = _HessianOperator(lambda: criterion(model(x), y), params)
    torch.manual_seed(3)
    v = _make_random_vec(params, rademacher=False)
    a = _flatten_grads(H(v))
    b = _flatten_grads(_hvp(criterion(model(x), y), params, v))
    assert torch.allclose(a, b, rtol=1e-12, atol=1e-14)


def test_hutchinson_matches_pyhessian_trace_on_the_same_probes():
    """PyHessian trace(maxIter=100, tol=1e-3) and BLME's loop draw the same
    Rademacher probes from the global RNG and stop at the same probe."""
    model, criterion, x, y = _build()
    torch.manual_seed(11)
    hess = hessian(model, criterion, data=(x, y), cuda=False)
    torch.manual_seed(11)
    py = hess.trace(maxIter=100, tol=1e-3)
    params = list(model.parameters())
    H = _HessianOperator(lambda: criterion(model(x), y), params)
    torch.manual_seed(11)
    blme = _hutchinson_trace(H, params, generator=None, n_iter=100, tol=1e-3)
    assert len(blme) == len(py)
    assert np.allclose(blme, py, rtol=1e-12, atol=1e-14)


class _Quad(nn.Module):
    """Quadratic loss L(theta) = 0.5 * sum(diag_i * theta_i^2) with a KNOWN
    Hessian diag(diag) — analytic ground truth for sign and SAM tests."""

    def __init__(self, theta, diag):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float64))
        self.register_buffer(
            "diag", torch.tensor(diag, dtype=torch.float64)
        )

    def loss(self):
        return 0.5 * (self.diag * self.theta ** 2).sum()


def test_signed_eigenvalue_on_negative_dominant_hessian():
    """When the dominant-magnitude curvature is NEGATIVE (a saddle), the
    kernel must report the SIGNED Rayleigh quotient (-5), matching
    PyHessian — not ||Hv|| = +5 (the pre-fix behavior). Hessian
    diag(-5, 3): |−5| > |3|, true lambda_max = +3."""
    torch.manual_seed(0)
    m = _Quad([1.0, 1.0], [-5.0, 3.0])
    params = [m.theta]
    v = _make_random_vec(params, rademacher=False)
    eig = _power_iteration_top_eigenvalue(
        lambda vv: _hvp(m.loss(), params, vv), v, 200, tol=0.0
    )
    assert eig == pytest.approx(-5.0, abs=1e-9), (
        f"expected signed -5.0 (PyHessian convention), got {eig}"
    )


def test_sam_sharpness_matches_analytic_quadratic():
    """For L = 0.5*||theta||^2: g = theta, so the SAM perturbation moves
    theta to theta*(1 + rho/||theta||) and the delta is EXACTLY
    rho*||theta|| + rho^2/2. Also pins that parameters are restored."""
    m = _Quad([3.0, 4.0], [1.0, 1.0])  # ||theta|| = 5
    params = [m.theta]
    baseline = float(m.loss().item())
    assert baseline == pytest.approx(12.5, abs=1e-12)

    rho = 0.1
    before = m.theta.detach().clone()
    delta, perturbed, grad_norm = _sam_sharpness(m.loss, params, rho, baseline)

    assert grad_norm == pytest.approx(5.0, abs=1e-12)
    assert delta == pytest.approx(rho * 5.0 + rho ** 2 / 2.0, abs=1e-12)
    assert perturbed == pytest.approx(baseline + delta, abs=1e-12)
    assert torch.equal(m.theta.detach(), before), "params not restored"


def test_asam_matches_analytic_quadratic():
    """L = 0.5 * sum(d_i theta_i^2): g = d*theta, T_w g = |theta| d theta and
    eps* = rho |theta|^2 d theta / ||T_w g||, so the perturbed loss is exact."""
    from blme.tasks.dynamics.sharpness import _asam_sharpness
    theta = [3.0, -4.0, 0.5]
    diag = [1.0, 2.0, 5.0]
    m = _Quad(theta, diag)
    params = [m.theta]
    baseline = float(m.loss().item())
    rho = 0.1
    t = torch.tensor(theta, dtype=torch.float64)
    d = torch.tensor(diag, dtype=torch.float64)
    twg = t.abs() * d * t
    eps = rho * t.abs() * twg / twg.norm()
    want = float(0.5 * (d * (t + eps) ** 2).sum()) - baseline
    before = m.theta.detach().clone()
    got, perturbed, norm = _asam_sharpness(m.loss, params, rho, baseline)
    assert got == pytest.approx(want, rel=1e-12)
    assert norm == pytest.approx(float(twg.norm()), rel=1e-12)
    assert torch.equal(m.theta.detach(), before), "params not restored"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
