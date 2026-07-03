"""Numeric-parity test: BLME interpretability_attribution vs captum.attr.InputXGradient.

The attribution task is input x gradient / saliency (Simonyan, Vedaldi & Zisserman
2014). BLME (src/blme/tasks/interpretability/attribution.py, ComponentAttributionTask):
  - hooks the input-embedding output  -> `activation` (batch, seq, dim), retain_grad()
  - scalar target = next-token CROSS-ENTROPY loss:
        shift_logits = logits[:, :-1, :]; shift_labels = input_ids[:, 1:]
        loss = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1))
  - loss.backward() -> activation.grad
  - per-token attribution = (activation.grad * activation).abs().sum(dim=-1)[:, :-1]
  - summary = _gini_nonnegative over the flattened per-token attributions

OFFICIAL reference (captum 0.9.0):
    captum.attr.InputXGradient(forward_func).attribute(inputs, target=...)
    returns elementwise  inputs * grad(forward_func(inputs), inputs).
We set forward_func = (embeddings -> CE loss scalar) so captum's "input" IS the
embedding activation and captum's "output" IS exactly BLME's scalar. Then we reduce
captum's elementwise map per-token identically to BLME (abs of product, sum over
hidden dim, drop last token). This is the canonical input x gradient method, run live.

The Gini summary is cross-checked against an independent mean-absolute-difference Gini.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

captum = pytest.importorskip("captum")
from captum.attr import InputXGradient  # noqa: E402

from blme.tasks.interpretability.attribution import (  # noqa: E402
    _gini_nonnegative,
    _input_x_gradient_per_token,
)


# ---------------------------------------------------------------------------
# Tiny deterministic LM: token ids -> embedding -> linear stack -> logits.
# The embedding tensor is the differentiation input for both paths.
# ---------------------------------------------------------------------------
class TinyLM(nn.Module):
    def __init__(self, vocab=23, dim=8, hidden=16, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(vocab, dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.head = nn.Linear(dim, vocab)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.empty_like(p).uniform_(-0.5, 0.5, generator=g))

    def embed_ids(self, input_ids):
        return self.embed(input_ids)

    def from_embeddings(self, emb):
        h = torch.tanh(self.fc1(emb))
        h = self.fc2(h)
        h = emb + h
        return self.head(h)


def _ce_loss(logits, input_ids):
    """BLME's exact scalar target (attribution.py lines 93-98)."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
    )


def _blme_pertoken(activation, grad):
    """BLME's REAL per-token reduction kernel (imported, not transcribed)."""
    return _input_x_gradient_per_token(activation, grad).detach().reshape(-1)


def _gini_independent(x):
    """Mean-absolute-difference Gini, independent of BLME's sorted-rank impl."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, None)
    n = x.size
    s = x.sum()
    if n == 0 or s <= 0:
        return 0.0
    mad = np.abs(x.reshape(-1, 1) - x.reshape(1, -1)).sum()
    return float(np.clip(mad / (2.0 * n * s), 0.0, 1.0))


def _build(dtype, seed_model, seed_data, S=7, vocab=23):
    torch.manual_seed(seed_data)
    model = TinyLM(vocab=vocab, seed=seed_model).to(dtype).eval()
    input_ids = torch.randint(0, vocab, (1, S))
    return model, input_ids


def _blme_attr(model, input_ids):
    """Run BLME's exact hook + backward + reduce path."""
    captured = {}

    def hook(_m, _i, output):
        act = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(act) and act.requires_grad:
            act.retain_grad()
        captured["activation"] = act

    h = model.embed.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        emb = model.embed_ids(input_ids)
        logits = model.from_embeddings(emb)
        loss = _ce_loss(logits, input_ids)
        loss.backward()
        act = captured["activation"]
        pertoken = _blme_pertoken(act, act.grad)
        elementwise = (act.grad * act).detach()
    finally:
        h.remove()
        model.zero_grad(set_to_none=True)
    return pertoken, elementwise


def _captum_attr(model, input_ids):
    """Run the OFFICIAL captum InputXGradient on the same embedding input."""

    def forward_func(emb_in):
        lg = model.from_embeddings(emb_in)
        return _ce_loss(lg, input_ids).reshape(1, 1)

    emb_input = model.embed_ids(input_ids).detach().clone().requires_grad_(True)
    attr = InputXGradient(forward_func).attribute(emb_input, target=0)
    pertoken = attr.abs().sum(dim=-1)[:, :-1].detach().reshape(-1)
    return pertoken, attr.detach()


CASES = [
    (torch.float64, 1e-9, 0, 1),
    (torch.float64, 1e-9, 3, 7),
    (torch.float64, 1e-9, 11, 42),
    (torch.float32, 1e-5, 0, 1),
    (torch.float32, 1e-5, 3, 7),
    (torch.float32, 1e-5, 11, 42),
]


@pytest.mark.parametrize("dtype,tol,seed_model,seed_data", CASES)
def test_inputxgrad_elementwise_parity(dtype, tol, seed_model, seed_data):
    """BLME (grad*input) == captum InputXGradient, elementwise, same model+input."""
    model, input_ids = _build(dtype, seed_model, seed_data)
    _, blme_elem = _blme_attr(model, input_ids)
    _, captum_elem = _captum_attr(model, input_ids)
    max_diff = (blme_elem.double() - captum_elem.double()).abs().max().item()
    assert max_diff < tol, f"elementwise max_diff={max_diff:.3e} (tol={tol:.0e})"


@pytest.mark.parametrize("dtype,tol,seed_model,seed_data", CASES)
def test_pertoken_attribution_parity(dtype, tol, seed_model, seed_data):
    """BLME per-token reduction == captum reduced identically."""
    model, input_ids = _build(dtype, seed_model, seed_data)
    blme_pt, _ = _blme_attr(model, input_ids)
    captum_pt, _ = _captum_attr(model, input_ids)
    blme_np = blme_pt.double().cpu().numpy()
    captum_np = captum_pt.double().cpu().numpy()
    max_diff = float(np.max(np.abs(blme_np - captum_np)))
    assert max_diff < tol, f"per-token max_diff={max_diff:.3e} (tol={tol:.0e})"


@pytest.mark.parametrize("dtype,tol,seed_model,seed_data", CASES)
def test_gini_summary_matches_independent(dtype, tol, seed_model, seed_data):
    """BLME _gini_nonnegative == independent mean-abs-difference Gini."""
    model, input_ids = _build(dtype, seed_model, seed_data)
    blme_pt, _ = _blme_attr(model, input_ids)
    vals = blme_pt.double().cpu().numpy()
    g_blme = _gini_nonnegative(vals)
    g_indep = _gini_independent(vals)
    assert abs(g_blme - g_indep) < 1e-9, f"gini blme={g_blme} indep={g_indep}"


def test_canonical_fixture_values():
    """Pin the canonical float64 case (seed_model=0, seed_data=1)."""
    model, input_ids = _build(torch.float64, 0, 1)
    blme_pt, _ = _blme_attr(model, input_ids)
    captum_pt, _ = _captum_attr(model, input_ids)
    expected = np.array(
        [0.1773958830829223, 0.17317480995671825, 0.0748825713157332,
         0.06405762541623476, 0.13931334985532717, 0.1399833927726529]
    )
    assert np.allclose(blme_pt.cpu().numpy(), expected, atol=1e-12)
    assert np.allclose(captum_pt.cpu().numpy(), expected, atol=1e-12)
    g = _gini_nonnegative(blme_pt.cpu().numpy())
    assert abs(g - 0.18692106469740177) < 1e-12


def test_gini_nonnegative_edge_cases():
    """BLME's documented guards: empty / all-zero -> 0.0; perfectly equal -> ~0."""
    assert _gini_nonnegative([]) == 0.0
    assert _gini_nonnegative([0.0, 0.0, 0.0]) == 0.0
    assert _gini_nonnegative([-1.0, -2.0]) == 0.0  # clamped to nonneg then total<=0
    assert abs(_gini_nonnegative([5.0, 5.0, 5.0, 5.0])) < 1e-12  # equal -> 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
