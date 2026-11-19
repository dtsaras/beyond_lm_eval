"""dynamics_sharpness: the ASAM normaliser is a parameter (asam_variant).

invariant (default): T_w = |w| on every parameter, exactly invariant.
release: the released asam.py rule, T_w = |w| + eta on tensors named "weight"
and T_w = 1 on the rest. The other is recorded as _meta_asam_sharpness_<other>
(docs/validation/decisions.yaml, dynamics_sharpness; blind review D).
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from blme.tasks.dynamics.sharpness import LossSharpnessTask, _asam_sharpness  # noqa: E402

CFG = {"num_samples": 2, "n_hutchinson": 3, "n_power_iter": 3, "seed": 0}


def _release_by_hand(compute_loss, named, rho, eta, base):
    """asam.py's ascent step, written out: grads of 'weight' tensors are
    multiplied by |w| + eta once for the norm and once more for the step."""
    params = [p for _, p in named]
    g = torch.autograd.grad(compute_loss(), params)
    t = [(p.detach().abs() + eta) if "weight" in n else torch.ones_like(p) for (n, p) in named]
    tg = [ti * gi for ti, gi in zip(t, g)]
    norm = torch.sqrt(sum((x.double() ** 2).sum() for x in tg)).item()
    saved = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, ti, x in zip(params, t, tg):
            p.add_(ti * x * (rho / norm))
        val = float(compute_loss())
        for p, s in zip(params, saved):
            p.copy_(s)
    return val - base


def test_release_rule_matches_asam_py_by_hand():
    torch.manual_seed(0)
    lin = torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.LayerNorm(5), torch.nn.Linear(5, 1))
    x, y = torch.randn(32, 6), torch.randn(32, 1)
    named = list(lin.named_parameters())
    loss = lambda: ((lin(x) - y) ** 2).mean()  # noqa: E731
    base = float(loss())
    got = _asam_sharpness(loss, [p for _, p in named], 0.05, base, eta=0.01,
                          normalize=["weight" in n for n, _ in named])[0]
    assert got == pytest.approx(_release_by_hand(loss, named, 0.05, 0.01, base), rel=1e-6)
    inv = _asam_sharpness(loss, [p for _, p in named], 0.05, base)[0]
    assert inv != pytest.approx(got, rel=1e-3)


def test_default_is_invariant_and_records_the_release_value():
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    r = LossSharpnessTask(dict(CFG)).evaluate(model, tok, fd.dataset(2))
    assert r["_meta_asam_variant"] == "invariant"
    assert "_meta_asam_sharpness_release" in r
    rel = LossSharpnessTask(dict(CFG, asam_variant="release")).evaluate(model, tok, fd.dataset(2))
    assert rel["_meta_asam_variant"] == "release"
    assert rel["asam_sharpness"] == r["_meta_asam_sharpness_release"]
    assert rel["_meta_asam_sharpness_invariant"] == r["asam_sharpness"]


def test_unknown_variant_is_an_error():
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    assert "error" in LossSharpnessTask(dict(CFG, asam_variant="paper")).evaluate(model, tok, fd.dataset(2))
