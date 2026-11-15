"""geometry_representation_sensitivity: S_unit against autograd.

S_unit is E_tokens ||n||^2 ||d log P(y | n) / dn||^2, with n the unit-gain
normalised state that the final norm maps to h = gamma * n + beta. The
reference rebuilds n from the final norm's input with the norm's own formula
(LayerNorm / RMSNorm, no affine), pushes it through gamma, beta and the LM
head, and differentiates the true-next-token log-probability with autograd.
Gains and biases are perturbed so the gauge the key removes is not trivially
the identity.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "invariance"))
import _family_d as fd  # noqa: E402
from blme.tasks.geometry.information_geometry import RepresentationSensitivityTask  # noqa: E402

TEXTS = fd.TEXTS[:5]


def _gpt2():
    m = fd.tiny_gpt2()
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        m.transformer.ln_f.weight.copy_(0.5 + torch.rand(m.transformer.ln_f.weight.shape, generator=g))
        m.transformer.ln_f.bias.copy_(0.3 * torch.randn(m.transformer.ln_f.bias.shape, generator=g))
    norm = m.transformer.ln_f
    return m, norm, lambda x: F.layer_norm(x, x.shape[-1:], eps=norm.eps)


def _llama():
    m = fd.tiny_llama()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        m.model.norm.weight.copy_(0.5 + torch.rand(m.model.norm.weight.shape, generator=g))
    norm = m.model.norm
    eps = norm.variance_epsilon
    return m, norm, lambda x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _reference(model, norm, unit, tok):
    head = model.get_output_embeddings()
    gamma = norm.weight.detach()
    beta = norm.bias.detach() if getattr(norm, "bias", None) is not None else torch.zeros_like(gamma)
    vals = []
    for text in TEXTS:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
        seen = {}
        h = norm.register_forward_hook(lambda m, i, o: seen.__setitem__("x", i[0].detach()))
        try:
            with torch.no_grad():
                model(ids)
        finally:
            h.remove()
        n = unit(seen["x"][0].double())[:-1].clone().requires_grad_(True)
        logits = (gamma.double() * n + beta.double()) @ head.weight.detach().double().T
        if head.bias is not None:
            logits = logits + head.bias.detach().double()
        lp = F.log_softmax(logits, dim=-1).gather(1, ids[0, 1:, None]).sum()
        (g,) = torch.autograd.grad(lp, n)
        vals.append(float(((n.detach() ** 2).sum(-1) * (g ** 2).sum(-1)).mean()))
    return float(np.mean(vals))


@pytest.mark.parametrize("build", [_gpt2, _llama], ids=["gpt2_layernorm", "llama_rmsnorm"])
def test_s_unit_matches_autograd_on_the_unit_gain_state(build):
    model, norm, unit = build()
    tok = fd.tokenizer()
    res = RepresentationSensitivityTask({"num_samples": len(TEXTS)}).evaluate(
        model, tok, dataset=[{"text": t} for t in TEXTS])
    assert "error" not in res, res
    assert res["_meta_final_norm_gain_form"] == "weight"
    want = _reference(model, norm, unit, tok)
    assert res["S_unit"] == pytest.approx(want, rel=1e-4)
    # Not the raw-h sensitivity: the gain gauge moves that one.
    assert abs(res["representation_sensitivity"] - want) > 1e-3 * want
