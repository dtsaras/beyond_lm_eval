"""Tests for interpretability_activation_sinks (new round-8 task).

Unifies three intrinsic phenomena the 2024-2025 literature shows are
"two (actually three) sides of the same coin" (Arroyo et al. 2025,
arXiv:2510.06477):

  1. **Attention sink** — fraction of attention mass consistently
     routed to the first token across heads/layers. Sinkε formula
     from Gu et al. ICLR 2025 (arXiv:2410.10781, reference impl
     https://github.com/sail-sg/Attention-Sink).
  2. **Massive activation** — fraction of residual-stream entries
     whose magnitude exceeds 100× the median norm (Sun et al. 2024,
     arXiv:2402.17762).
  3. **Compression valley** — the middle-layer dip in representation
     entropy observed in every modern LLM (Arroyo et al. 2025). We
     compute the layer index of minimum entropy and the valley depth.

These three metrics jointly characterise whether a model has
developed the emergent bias-token mechanism that underpins many of
its downstream behaviours.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))


def test_sink_epsilon_formula_matches_reference():
    """Reproduce Gu et al. 2025 Sink₁ε on a minimal attention tensor where
    ALL queries attend to key-0 (a BOS-sink pattern). Sink₁ε is the FIRST-
    token sink fraction over (layer, head): the first token's importance
    (1.0) exceeds 0.3 in every (layer, head), so Sink₁ε = 1.0.
    (Corrected 2026-06-22 from the old mean-over-all-key-positions form,
    which diluted this to 0.25.)"""
    from blme.tasks.interpretability.activation_sinks import _sink_epsilon

    # Attention (L=1, H=2, T=4, T=4) where every query attends to
    # token 0.
    attn = torch.zeros(1, 2, 4, 4)
    attn[:, :, :, 0] = 1.0
    mask = torch.tril(torch.ones(4, 4))
    attn = attn * mask
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    sink_frac = _sink_epsilon(attn, epsilon=0.3)
    # Every (layer, head) has the first token as a sink.
    assert sink_frac == pytest.approx(1.0, rel=1e-3)


def test_sink_epsilon_zero_when_uniform_attention():
    """A uniform-attention tensor should have zero tokens with the
    importance score exceeding epsilon (per the paper)."""
    from blme.tasks.interpretability.activation_sinks import _sink_epsilon

    T = 16
    # Uniform within each causal row.
    attn = torch.tril(torch.ones(2, T, T))
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    frac = _sink_epsilon(attn.unsqueeze(0), epsilon=0.3)
    assert frac == pytest.approx(0.0, abs=0.05)


def test_massive_activation_fraction_on_synthetic():
    """Fraction of residual entries >100× the median |·| — on a
    random Gaussian it should be tiny (< 1%); on a tensor with a
    handful of huge spikes it should match the spike count."""
    from blme.tasks.interpretability.activation_sinks import _massive_activation_fraction

    torch.manual_seed(0)
    # Random Gaussian: almost no massive activations.
    H = torch.randn(100, 512)
    frac = _massive_activation_fraction(H, threshold_ratio=100.0)
    assert frac < 0.01
    # Inject 5 outliers at 200× the median.
    med = H.abs().median().item()
    H_out = H.clone()
    H_out[0, 0] = 200 * med
    H_out[5, 3] = -200 * med
    H_out[10, 7] = 250 * med
    frac = _massive_activation_fraction(H_out, threshold_ratio=100.0)
    # Must pick up at least the 3 injected outliers.
    total = 100 * 512
    assert frac * total >= 3 - 1e-6


def test_compression_valley_layer_and_depth():
    """Given a synthetic entropy-per-layer profile with an obvious
    mid-layer dip, the valley_layer and valley_depth must match."""
    from blme.tasks.interpretability.activation_sinks import _compression_valley

    entropy_profile = [0.90, 0.85, 0.60, 0.40, 0.65, 0.88, 0.92]
    result = _compression_valley(entropy_profile)
    # Valley is at layer 3 (0-indexed, where entropy = 0.40).
    assert result["valley_layer"] == 3
    # Normalised to [0, 1]: 3 / (7-1) = 0.5.
    assert result["valley_layer_norm"] == pytest.approx(0.5)
    # Depth = surrounding (start/end) mean − valley.
    # Start+end mean = (0.90 + 0.92)/2 = 0.91; depth = 0.91 - 0.40 = 0.51.
    assert result["valley_depth"] == pytest.approx(0.51, rel=0.05)


def test_task_registers_and_runs_on_stub():
    """End-to-end sanity: the task registers, accepts a stub model with
    both hidden_states and attentions, and returns the expected keys."""
    import blme.tasks  # register
    from blme.registry import get_task

    cls = get_task("interpretability_activation_sinks")
    assert cls is not None

    class StubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Linear(16, 16, bias=False)

        def forward(self, input_ids=None, attention_mask=None, **kw):
            B, T = input_ids.shape
            torch.manual_seed(int(input_ids.sum().item()))
            h = [torch.randn(B, T, 16) for _ in range(4)]
            # Attention: (B, H, T, T) with BOS-heavy pattern.
            a = torch.zeros(2, 2, T, T)
            a[:, :, :, 0] = 1.0
            mask = torch.tril(torch.ones(T, T))
            a = a * mask
            a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            attentions = (a, a, a)

            class Out:
                def __init__(self, hidden_states, attentions):
                    self.hidden_states = hidden_states
                    self.attentions = attentions
            return Out(h, attentions)

        def parameters(self):
            yield torch.zeros(1, device="cpu")

    class StubTok:
        def __call__(self, text, return_tensors="pt", truncation=True, max_length=None):
            return {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)}

    task = cls({"num_samples": 2})
    dataset = [{"text": "x"}] * 2
    result = task.evaluate(model=StubModel(), tokenizer=StubTok(),
                           dataset=dataset, cache=None)
    assert "sink_epsilon_fraction" in result
    assert "massive_activation_fraction" in result
    assert "valley_layer_norm" in result
    assert "valley_depth" in result
