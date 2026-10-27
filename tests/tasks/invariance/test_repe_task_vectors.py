"""Direct mechanism tests for repe_task_vectors' consensus_energy_rel.

consensus_energy_rel = mean over layers of mean_{i != j} <d_i, d_j> /
mean ||h||^2. N: the U-statistic is unbiased for ||E d||^2 (the norm of the
mean carries + tr(Sigma)/N). Scale: numerator and denominator scale by c^2.
Width: zero-padding adds nothing. The replaced headline,
mean_vector_norm_normalized = mean ||v_l|| / sqrt(d), is the known negative.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import invariance as inv  # noqa: E402
from blme.tasks.repe.concept_separability import _FACT_PAIRS  # noqa: E402
from blme.tasks.repe.task_vectors import TaskVectorGeometryTask, _consensus_energy  # noqa: E402
from repe_width_padding import tiny_llama, zero_pad_llama_width  # noqa: E402

PAIRS = [{"text_pos": a, "text_neg": b} for a, b in _FACT_PAIRS]
KEYS = ["consensus_energy_rel", "pair_direction_cos_mean"]


def test_consensus_energy_is_unbiased_in_the_pair_count():
    """Same distribution of pair differences at N = 5 and N = 39 (4000 draws):
    the U-statistic agrees with itself; ||mean d||^2 falls with N."""
    rng = np.random.default_rng(0)
    d = 16
    mu = np.r_[0.5, np.zeros(d - 1)]
    rel, raw = {}, {}
    for n in (5, 39):
        r, q = [], []
        for _ in range(4000):
            D = torch.as_tensor(mu + rng.standard_normal((n, d)))
            H = torch.as_tensor(3.0 + rng.standard_normal((2 * n, d)))
            r.append(_consensus_energy(D, H))
            q.append(float((D.mean(0) ** 2).sum()))
        rel[n], raw[n] = np.mean(r), np.mean(q)
    assert rel[5] == pytest.approx(rel[39], rel=0.1)
    # Known negative: E||mean d||^2 = 0.25 + 16 / N.
    assert raw[5] / raw[39] > 5.0


def test_consensus_energy_invariant_under_residual_gauge_and_width_padding():
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = tiny_llama(len(tok))
    cfg = {"num_samples": len(PAIRS)}
    base = TaskVectorGeometryTask(cfg).evaluate(model, tok, PAIRS)

    ids = tok(PAIRS[0]["text_pos"], return_tensors="pt")["input_ids"]
    gauged = inv.residual_scale_gauge(model, 2.0, rescale_eps=True, verify_on=ids, atol=0.0)
    g = TaskVectorGeometryTask(cfg).evaluate(gauged, tok, PAIRS)
    inv.assert_invariant(base, g, KEYS, rtol=1e-6)
    # Known negative: every pre-norm layer's norm doubles; the layer mean moves
    # less only because the unscaled post-ln_f entry dominates it here.
    assert g["layer_task_vector_norms"][0] == pytest.approx(2 * base["layer_task_vector_norms"][0], rel=1e-6)
    inv.assert_moves(base, g, "_meta_mean_vector_norm_normalized", min_rel=0.01)

    pad = TaskVectorGeometryTask(cfg).evaluate(zero_pad_llama_width(model), tok, PAIRS)
    inv.assert_invariant(base, pad, KEYS, rtol=1e-4)
    # /sqrt(d) halves under the 4x padding.
    assert pad["_meta_mean_vector_norm_normalized"] == pytest.approx(
        base["_meta_mean_vector_norm_normalized"] / 2, rel=1e-3)
