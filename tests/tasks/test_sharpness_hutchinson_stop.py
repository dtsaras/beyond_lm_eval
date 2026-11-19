"""dynamics_sharpness: the Hutchinson stopping rule is a parameter.

"pyhessian" (default) is PyHessian's relative-change rule, which does not
control precision (phi-2: 29 probes at a 27% relative SE; pythia-160m: the
100-probe cap at 38%). "rse" stops at a relative SE target after a minimum
number of probes, on the same seeded probe sequence, so PyHessian's stopping
point is a prefix of it and is recorded.
"""
import numpy as np
import torch

from blme.tasks.dynamics.sharpness import _hutchinson_trace


def _problem(seed=0, d=30):
    g = torch.Generator().manual_seed(seed)
    B = torch.randn(d, d, generator=g, dtype=torch.float64)
    A = (B + B.T) / 2 + d * torch.eye(d, dtype=torch.float64)   # trace > 0, real off-diagonal
    params = [torch.zeros(d, dtype=torch.float64)]
    return A, params, (lambda v: [A @ v[0]])


def test_rse_rule_meets_its_target_and_extends_the_pyhessian_sequence():
    A, params, hvp = _problem()
    st_p, st_r = {}, {}
    p = _hutchinson_trace(hvp, params, torch.Generator().manual_seed(3), n_iter=500, status=st_p)
    r = _hutchinson_trace(hvp, params, torch.Generator().manual_seed(3), n_iter=500, status=st_r,
                          stop="rse", rse_target=0.01, min_probes=10)
    assert st_r["converged"] and len(r) >= 10
    rse = np.std(r, ddof=1) / np.sqrt(len(r)) / abs(np.mean(r))
    assert rse <= 0.01
    n = min(len(r), len(p))
    assert r[:n] == p[:n]                                    # same probe sequence
    assert st_p["pyhessian_n"] == len(p)
    # The RSE run records PyHessian's stopping point whenever it went past it.
    assert st_r["pyhessian_n"] == (len(p) if len(r) >= len(p) else None)
    # A tighter target runs longer than PyHessian's rule and still records it.
    st_t = {}
    t = _hutchinson_trace(hvp, params, torch.Generator().manual_seed(3), n_iter=5000, status=st_t,
                          stop="rse", rse_target=0.001, min_probes=10)
    assert len(t) > len(p) and t[:len(p)] == p and st_t["pyhessian_n"] == len(p)


def test_pyhessian_rule_is_the_default_and_unchanged():
    A, params, hvp = _problem(seed=1)
    st = {}
    p = _hutchinson_trace(hvp, params, torch.Generator().manual_seed(5), n_iter=100, tol=1e-3, status=st)
    assert st["stop"] == "pyhessian" and st["n_iter"] == len(p)
    # Recompute PyHessian's rule by hand on the same samples.
    trace, stop_at = 0.0, None
    for i in range(1, len(p) + 1):
        m = float(np.mean(p[:i]))
        if abs(m - trace) / max(abs(trace), 1e-300) < 1e-3:
            stop_at = i
            break
        trace = m
    assert stop_at in (None, len(p))
