"""Fail-closed details found in the 2026-11-15 close-out.

- alpha_req: squared singular values of huge finite rows overflow; the ratios
  are scale-free, so they are recomputed after an exact power-of-two rescale.
- representation_sensitivity: a head constant along the probe ray is linear
  on it (discrepancy 0), not divided by an absolute 1e-30 floor.
- positional_decay: the output-projection input is captured when the
  projection is called with a keyword argument.
"""
import numpy as np
import torch

from blme.tasks.geometry.alpha_req import _eigenspectrum
from blme.tasks.geometry.positional_decay import _capture_input


def test_alpha_req_spectrum_survives_huge_finite_rows():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 8)) * np.linspace(1, 4, 8)
    ref = _eigenspectrum(X)
    huge = _eigenspectrum(X * 2.0 ** 600)      # s**2 ~ 1e361: overflows float64
    assert huge.size == ref.size
    np.testing.assert_allclose(huge, ref, rtol=1e-12)


def test_keyword_projection_input_is_captured():
    store = {}
    proj = torch.nn.Linear(4, 4)
    h = proj.register_forward_pre_hook(_capture_input(store, 0), with_kwargs=True)
    x = torch.randn(1, 3, 4)
    proj(input=x)
    h.remove()
    assert torch.equal(store[0], x)
    store.clear()
    h = proj.register_forward_pre_hook(_capture_input(store, 1), with_kwargs=True)
    proj(x)
    h.remove()
    assert torch.equal(store[1], x)


def test_a_head_constant_on_the_probe_ray_is_linear_not_floor_divided():
    from blme.tasks.geometry.information_geometry import RepresentationSensitivityTask
    model = type("M", (), {"config": type("C", (), {})()})()
    W = torch.zeros(11, 4)                       # f(x) = 0 for every x
    h = torch.randn(5, 4, generator=torch.Generator().manual_seed(1))
    rel, why = RepresentationSensitivityTask._head_linearity(model, None, W, h)
    assert why is None and rel == 0.0
