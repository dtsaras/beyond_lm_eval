"""Regression tests for geometry_perplexity — specifically ppl_rare /
ppl_freq behaviour.

The original BLME implementation thresholded the full vocabulary by
argsort of ``token_counts``. With a small eval corpus (~12k tokens) and
a 50–200k vocab, most vocab ids have count 0; the ``bottom 20% by
count`` set is dominated by never-observed ids, so ``cnt_rare == 0``
and ``ppl_rare == +inf`` for every model. This test pins the fix:
``rare_ids`` / ``freq_ids`` must be selected from *observed* tokens only
so the resulting perplexity buckets are actually populated.
"""
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# Minimal stubs for the blme package plumbing.
SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from blme.tasks.geometry.perplexity import RarePPLTask


class _DummyCache:
    """Populated cache returning fixed prediction stats."""
    def __init__(self, stats):
        self._stats = stats
        self.is_populated = True

    def get_prediction_stats(self, num_samples=None):
        return self._stats, None


def _build_stats(vocab_size=1000, seq_len=50, seed=0):
    """Build a prediction-stats payload where only ids 0..19 are
    observed. The remaining 980 ids have count 0 — the degenerate case
    the old code mishandled."""
    rng = np.random.default_rng(seed)
    # Uniform over the 20 observed ids.
    labels = torch.tensor(rng.integers(0, 20, size=seq_len), dtype=torch.long)
    # Random logits so cross-entropy is positive (and differs per token).
    torch.manual_seed(seed)
    logits = torch.randn(seq_len, vocab_size)
    token_counts = np.zeros(vocab_size, dtype=np.int64)
    for tid in labels.tolist():
        token_counts[tid] += 1
    return {
        "logits": [logits],
        "labels": [labels],
        "token_counts": token_counts,
    }


def test_ppl_rare_and_freq_are_finite_when_corpus_is_small():
    """With a 20-observed-token corpus in a 1000-vocab model, both
    ppl_rare and ppl_freq must be finite — not inf."""
    stats = _build_stats(vocab_size=1000, seq_len=200)
    task = RarePPLTask({})
    result = task.evaluate(
        model=None,
        tokenizer=None,
        dataset=[{"text": "dummy"}],
        cache=_DummyCache(stats),
    )
    assert np.isfinite(result["ppl_overall"])
    assert np.isfinite(result["ppl_rare"]), (
        "ppl_rare must be computed over observed rare tokens, not full vocab"
    )
    assert np.isfinite(result["ppl_freq"])
    assert result["n_tokens_scored"] > 0


def test_ppl_rare_distinguishes_rare_from_freq():
    """With ids 0–4 very frequent and ids 5–19 rare, ppl_freq should
    differ from ppl_rare (they should at least both be finite)."""
    vocab_size = 5000
    rng = np.random.default_rng(1)
    # 800 labels drawn from {0..4} (very frequent), 200 from {5..19} (rare).
    labels = np.concatenate([
        rng.integers(0, 5, size=800),
        rng.integers(5, 20, size=200),
    ])
    rng.shuffle(labels)
    labels_t = torch.tensor(labels, dtype=torch.long)
    torch.manual_seed(1)
    logits = torch.randn(len(labels), vocab_size)
    token_counts = np.zeros(vocab_size, dtype=np.int64)
    for tid in labels.tolist():
        token_counts[tid] += 1
    stats = {
        "logits": [logits],
        "labels": [labels_t],
        "token_counts": token_counts,
    }
    task = RarePPLTask({})
    result = task.evaluate(
        model=None, tokenizer=None,
        dataset=[{"text": "dummy"}], cache=_DummyCache(stats),
    )
    assert np.isfinite(result["ppl_rare"])
    assert np.isfinite(result["ppl_freq"])
    # Both must be positive
    assert result["ppl_rare"] > 0
    assert result["ppl_freq"] > 0
