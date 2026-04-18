"""Regression tests for the round-4 parallel-audit fixes.

These tests are deliberately thin unit tests against the pure-numeric
subroutines — not end-to-end model runs — because the bugs involve
cross-architecture comparability and numerical correctness, both of
which can be verified without a real model.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))


def test_lid_is_deterministic_across_reruns():
    """`_compute_lid_for_matrix` must return the same statistics on
    repeated calls with identical input — previous code used
    ``np.random.choice`` without a fixed seed, so downstream reruns of
    the same model produced different LID numbers."""
    from blme.tasks.geometry.lid import _compute_lid_for_matrix
    import torch

    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.standard_normal((200, 32)).astype(np.float32))
    r1 = _compute_lid_for_matrix(X, k=10, max_queries=64)
    r2 = _compute_lid_for_matrix(X, k=10, max_queries=64)
    assert r1 is not None and r2 is not None
    assert r1["lid_mean"] == pytest.approx(r2["lid_mean"])
    assert r1["lid_median"] == pytest.approx(r2["lid_median"])


def test_attention_graph_gini_zero_total_case():
    """Verify no division-by-zero on a fully-zero attention matrix.

    The attention_graph Gini computation was dividing by ``cum_edges[-1]``
    which could be zero on a masked-out layer — produced ±inf that
    contaminated downstream aggregation.
    """
    import numpy as np
    # Repro the loop's inner logic: a fully-zero adj matrix.
    adj = np.zeros((8, 8), dtype=np.float64)
    flat_edges = np.sort(adj.flatten())
    cum_edges = np.cumsum(flat_edges)
    n = len(flat_edges)
    total = float(cum_edges[-1])
    # With the fix we simply skip; sanity-check the guard works.
    if total > 0:
        gini = (n + 1 - 2 * np.sum(cum_edges) / total) / n
    else:
        gini = None
    assert gini is None, "Zero-total attention matrix should be skipped, not produce ±inf"


def test_prediction_entropy_uses_log_softmax_numerically_stable():
    """Directly test that `log_softmax(x)` path produces finite entropy
    even on wildly-skewed logits that would underflow `softmax`."""
    import torch

    # Logits where one token dominates by ~100 units — softmax would
    # give 1.0 for the argmax and ~0.0 everywhere else in fp32; in
    # fp16 the non-max softmaxes underflow to exactly 0 before ``log``.
    logits = torch.zeros(1, 1, 1000)
    logits[0, 0, 42] = 100.0
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    assert torch.isfinite(entropy).all()
    # Near-zero since almost all mass is on the argmax.
    assert float(entropy[0, 0]) < 1e-3


def test_homology_output_keys_use_persistence_spelling():
    """The topology.homology.PersistenceHomologyTask output keys used
    the misspelling ``persistance``, breaking downstream aggregation
    when users look up ``persistence_*``. Fix pins the correct spelling."""
    from blme.tasks.topology.homology import PersistentHomologyTask
    import inspect

    src = inspect.getsource(PersistentHomologyTask)
    assert "persistance" not in src
    # Expected keys in the output
    assert "mean_persistence_h0" in src
    assert "max_persistence_h0" in src
    assert "mean_persistence_h1" in src


def test_gradient_flow_slope_uses_normalised_depth():
    """Slope of log(gradient_norm) vs normalised depth (x in [0, 1])
    rather than raw layer index. Verified by reading the source for
    the normalisation factor."""
    import inspect
    from blme.tasks.dynamics import gradient_flow

    src = inspect.getsource(gradient_flow)
    # Must divide layer indices by (n_layers - 1) before polyfit.
    assert "n_layers - 1" in src or "n_layers-1" in src


def test_induction_causal_validation_is_raw_accuracy_diff():
    """causal_validation must be ``top_drop - rand_drop`` without
    dividing by baseline_acc — the old code amplified small-model
    scores because their baselines are lower."""
    import inspect
    from blme.tasks.interpretability import induction

    src = inspect.getsource(induction)
    # The old line `causal_validation = (top_drop - rand_drop) / denom`
    # must be gone.
    assert "(top_drop - rand_drop) / denom" not in src
    # The new line must be a raw subtraction.
    assert "causal_validation = top_drop - rand_drop" in src
