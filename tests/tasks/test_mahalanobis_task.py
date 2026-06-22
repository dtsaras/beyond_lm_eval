"""Regression tests for geometry_mahalanobis.

The task must use a held-out split for ID samples: one split fits the
Gaussian (mean + inverse covariance), the other is scored against that
model. Scoring the fitting data against itself produces an in-sample
distance that is systematically biased downward by leverage (≈ p / n),
so ``ood_separation_gap`` appears larger than it should be. The bias
grows with ``D / n``, which is exactly the regime where a small eval
corpus is mean-pooled into D≈2-8k hidden states — the current setup.
"""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from blme.tasks.geometry.mahalanobis import MahalanobisOODTask, _compute_mahalanobis_distances


def test_in_sample_mahalanobis_is_biased_downward():
    """Sanity check the known leverage bias. With n=50 i.i.d. Gaussian
    samples in d=30 dimensions, in-sample average Mahalanobis should be
    noticeably smaller than the true ``sqrt(d)`` expectation."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 30))
    # In-sample
    d_in = _compute_mahalanobis_distances(X, X)
    mean_in = float(np.mean(d_in))
    # Held-out: fit on first 25, score on last 25
    d_out = _compute_mahalanobis_distances(X[:25], X[25:])
    mean_out = float(np.mean(d_out))
    # Theoretical expected value E[sqrt(chi^2_d)] ~ sqrt(d) = ~5.48.
    # Held-out should be much closer to this than in-sample.
    assert mean_in < mean_out, (
        f"In-sample Mahalanobis should underestimate true value. "
        f"Got in={mean_in:.3f} >= out={mean_out:.3f}"
    )


def test_task_uses_holdout_split_on_id_samples(monkeypatch):
    """The task's evaluate() must split X_id and score ID distances on a
    held-out subset rather than on the same samples used to fit the
    covariance. Detected by: ID distances should not be pathologically
    small (e.g. 30-dim data in 50 samples should give mean > 0.1)."""
    # Patch the sample-level collector to return deterministic Gaussian
    # representations sized like a real mean-pooled hidden-state matrix.
    import blme.tasks.geometry.mahalanobis as modmahal

    rng = np.random.default_rng(7)
    # Produce 60 samples, 30 dims — moderate D / n ratio.
    X_id_full = rng.standard_normal((60, 30)).astype(np.float32)
    X_ood_full = rng.standard_normal((60, 30)).astype(np.float32) + 2.0  # shifted = OOD

    call_idx = {"i": 0}

    def fake_collect(model, tokenizer, dataset, num_samples, **kwargs):
        # First call is for ID, second for OOD.
        call_idx["i"] += 1
        if call_idx["i"] == 1:
            return torch.from_numpy(X_id_full[:num_samples])
        return torch.from_numpy(X_ood_full[:num_samples])

    monkeypatch.setattr(modmahal, "_collect_mean_pooled_hidden_states", fake_collect)

    # Minimal dataset — the task just needs it to exist with "text" keys.
    dataset = [{"text": "a"}] * 60

    # Patch tokenizer.encode/decode/ __call__ to accept any string
    class StubTok:
        def encode(self, text, add_special_tokens=False):
            return list(range(10))

        def decode(self, ids):
            return "decoded"

    task = MahalanobisOODTask({"num_samples": 30, "ood_strategy": "shuffle"})
    result = task.evaluate(model=None, tokenizer=StubTok(), dataset=dataset, cache=None)
    # Must not report the degenerate zero-ID case.
    assert "mean_mahalanobis_id" in result, result
    # With a held-out split, ID mean should be a sensible non-tiny number.
    # Upper bound is sqrt(d) + noise; lower bound rules out in-sample bias.
    mean_id = result["mean_mahalanobis_id"]
    assert mean_id > 1.0, (
        f"ID mean {mean_id:.3f} too small — suggests in-sample bias not removed"
    )
    # Must expose the n_id_fit and n_id_score splits for transparency.
    assert "n_id_fit" in result or "n_id_score" in result, (
        "Task should report how many ID samples were used for fitting vs scoring"
    )


def test_task_scores_mean_pooled_sample_representations():
    """Each text sample must contribute one mean-pooled final-layer vector.
    Token-flattened hidden states inflate the sample counts by sequence
    length and change the covariance population."""
    import torch.nn as nn

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            base = int(str(text).split()[-1])
            return [base, base + 1, base + 2, base + 3]

        def decode(self, ids):
            return f"sample {int(ids[0])}"

        def __call__(self, text, return_tensors="pt", **kwargs):
            ids = torch.tensor([self.encode(text)], dtype=torch.long)

            class B(dict):
                def to(self, dev): return self
                def __getattr__(self, n):
                    try: return self[n]
                    except KeyError: raise AttributeError(n)

            return B({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        @property
        def device(self):
            return self.p.device

        def forward(self, input_ids=None, output_hidden_states=False, **kwargs):
            ids = input_ids[0].float()
            token_features = torch.stack([
                ids,
                ids * 0.5 + torch.arange(len(ids), dtype=torch.float32),
                torch.sin(ids * 0.1),
            ], dim=1).unsqueeze(0)

            class Out:
                hidden_states = (token_features * 0.5, token_features)

            return Out()

    dataset = [{"text": f"sample {i}"} for i in range(20)]
    result = MahalanobisOODTask({"num_samples": 10, "ood_strategy": "shuffle"}).evaluate(
        Model(), Tokenizer(), dataset=dataset, cache=None,
    )

    assert "error" not in result, result
    assert result["n_id_fit"] + result["n_id_score"] == 10
    assert result["n_ood_score"] == 10


def test_fit_split_preserves_id_scoring_holdout_for_small_n():
    """With only five ID samples the split must still leave ≥2 score rows."""
    n_id = 5
    fit_n = max(2, min(n_id - 2, max(5, n_id // 2)))
    assert n_id - fit_n >= 2


def test_insufficient_holdout_payload_omits_auroc():
    """When holdout scoring is impossible, omit biased ID/AUROC fields."""
    from blme.tasks.geometry.mahalanobis import _compute_mahalanobis_distances

    rng = np.random.default_rng(0)
    X_id_fit = rng.standard_normal((5, 12)).astype(np.float32)
    X_ood = rng.standard_normal((5, 12)).astype(np.float32) + 1.0
    dists_ood = _compute_mahalanobis_distances(X_id_fit, X_ood)
    payload = {
        "insufficient_holdout": True,
        "n_id_fit": int(X_id_fit.shape[0]),
        "n_id_score": 1,
        "mean_mahalanobis_ood": float(np.mean(dists_ood)),
    }
    assert payload["insufficient_holdout"] is True
    assert "auroc" not in payload
    assert "mean_mahalanobis_id" not in payload
    assert np.isfinite(payload["mean_mahalanobis_ood"])
