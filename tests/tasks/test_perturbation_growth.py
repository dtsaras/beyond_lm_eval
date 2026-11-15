"""dynamics_perturbation_growth: divergence-profile numerics and evaluate() behaviour."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adversarial_fixtures import on_grid_translation, scale_ladder  # noqa: E402


def _halving_trajectory(s, n_layers=6):
    """One-coordinate divergence delta_l = s * 0.5**l: exact log ratio (L-1) ln 1/2."""
    clean = [torch.zeros(4, 3, dtype=torch.float64) for _ in range(n_layers)]
    pert = []
    for l in range(n_layers):
        p = torch.zeros(4, 3, dtype=torch.float64)
        p[:, 0] = s * 0.5 ** l
        pert.append(p)
    return clean, pert


@pytest.mark.parametrize("s", scale_ladder(reduction_depth=2, n_terms=4))
def test_growth_profile_exact_at_every_scale(s):
    from blme.tasks.dynamics.perturbation_growth import _growth_profile
    clean, pert = _halving_trajectory(s)
    out = _growth_profile([c.float() for c in clean], [p.float() for p in pert])
    assert out, f"no metric at scale {s}"
    assert out["growth_log_ratio"] == pytest.approx(5 * np.log(0.5), rel=1e-12)


def test_growth_profile_ignores_a_common_translation():
    from blme.tasks.dynamics.perturbation_growth import _growth_profile
    g = np.random.default_rng(0)
    v = g.standard_normal((20, 8))
    d = 1e-2 * g.standard_normal((20, 8))
    base_c, base_p, shifted_c, shifted_p = [], [], [], []
    for l in range(5):
        bc, tc, ok_c = on_grid_translation(v * 0.7 ** l, 1e3, dtype=np.float64)
        bp, tp, ok_p = on_grid_translation((v + d) * 0.7 ** l, 1e3, dtype=np.float64)
        assert ok_c and ok_p
        base_c.append(torch.from_numpy(bc)); base_p.append(torch.from_numpy(bp))
        shifted_c.append(torch.from_numpy(tc)); shifted_p.append(torch.from_numpy(tp))
    a = _growth_profile(base_c, base_p)
    b = _growth_profile(shifted_c, shifted_p)
    assert b["growth_log_ratio"] == pytest.approx(a["growth_log_ratio"], rel=1e-12)


def test_relative_form_removes_state_norm_growth():
    """A stack that rescales the state by a per layer moves the divergence and the
    state alike: the absolute log ratio is L ln a, the relative one is 0."""
    import torch
    from blme.tasks.dynamics.perturbation_growth import _growth_profile
    g = torch.Generator().manual_seed(0)
    h0 = torch.randn(40, 8, generator=g, dtype=torch.float64)
    p0 = h0 + 1e-2 * torch.randn(40, 8, generator=g, dtype=torch.float64)
    a, L = 1.7, 5
    clean = [h0 * a ** l for l in range(L + 1)]
    pert = [p0 * a ** l for l in range(L + 1)]
    out = _growth_profile(clean, pert)
    assert out["growth_log_ratio"] == pytest.approx(L * np.log(a), rel=1e-10)
    assert out["growth_log_ratio_relative"] == pytest.approx(0.0, abs=1e-12)
