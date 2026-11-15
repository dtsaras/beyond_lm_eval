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


def test_s_unit_is_finite_with_a_non_affine_final_norm():
    """OLMo-1B's final LayerNorm has no gain or bias; gamma = 1 is well defined."""
    import sys
    from pathlib import Path
    import transformers
    sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))
    import _family_d as fd
    from blme.tasks.geometry.information_geometry import RepresentationSensitivityTask
    cfg = transformers.OlmoConfig(vocab_size=1000, hidden_size=32, intermediate_size=64,
                                  num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=4, max_position_embeddings=128)
    torch.manual_seed(0)
    model = transformers.OlmoForCausalLM(cfg).eval()
    assert getattr(model.model.norm, "weight", None) is None
    r = RepresentationSensitivityTask({"num_samples": 3}).evaluate(model, fd.tokenizer(), fd.dataset(3))
    assert r["_meta_final_norm_gain_form"] == "no_affine"
    assert np.isfinite(r["S_unit"]) and r["S_unit"] > 0
