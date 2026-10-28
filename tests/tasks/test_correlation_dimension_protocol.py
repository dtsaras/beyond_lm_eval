"""geometry_correlation_dimension: the fixed fit window and one point per sequence.

The task runs its own forward pass, so a stub model serves a FIXED cloud: the
text "doc i" yields hidden states whose every row is point i of the cloud. The
mean-pooled and last-token readouts of document i are then the same point, so
these tests isolate the estimator.
"""
import types

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from blme.tasks.geometry.correlation_dimension import CorrelationDimensionTask  # noqa: E402

TASK_CFG = {"num_samples": 200}


class _Tok:
    eos_token_id = None
    sep_token_id = None

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=None, **kw):
        i = int(str(text).split()[-1])
        ids = torch.full((1, 3), i, dtype=torch.long)
        return transformers.BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})


class _CloudModel(torch.nn.Module):
    """Every row of document i is cloud[i] (float64)."""

    def __init__(self, cloud):
        super().__init__()
        self.cloud = torch.as_tensor(cloud, dtype=torch.float64)
        self._anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=True, **kw):
        i = int(input_ids[0, 0])
        T = input_ids.shape[1]
        h = self.cloud[i].reshape(1, 1, -1).expand(1, T, -1).clone()
        return types.SimpleNamespace(hidden_states=(0.5 * h, h))


def _run(cloud, cfg=TASK_CFG):
    data = [{"text": f"doc {i}"} for i in range(cloud.shape[0])]
    return CorrelationDimensionTask(dict(cfg)).evaluate(_CloudModel(cloud), _Tok(), data)


def _gauss(seed=0, n=200, d=32):
    return np.random.default_rng(seed).standard_normal((n, d))


def test_headline_is_continuous_under_tiny_jitter():
    """1e-3 relative jitter of the points moves the fixed-window D2 by < 1%
    over 8 draws; the plateau search it replaced (now _meta_) jumped between
    windows (dossier: gpt2 6.18-8.21 at 1e-3)."""
    X = _gauss()
    outs = [_run(X * (1 + 1e-3 * np.random.default_rng(100 + s).standard_normal(X.shape)))
            for s in range(8)]
    vals = np.asarray([o["correlation_dimension"] for o in outs])
    assert np.all(np.isfinite(vals))
    assert (vals.max() - vals.min()) / vals.mean() < 0.01, vals
    assert outs[0]["_meta_correlation_dimension_fit_window"] == "du_tanaka_ishii_2025_appA"
    plateau = np.asarray([o["_meta_correlation_dimension_plateau"] for o in outs])
    assert (np.nanmax(plateau) - np.nanmin(plateau)) / np.nanmean(plateau) > 0.02, plateau


def test_window_is_the_published_rank_range():
    """C in [20 / (N (N - 1)), 1 / N]: ranks 10 .. (N - 1) / 2 of the sorted
    pair distances, on every positive distance there."""
    from scipy.spatial.distance import pdist
    from blme.tasks.geometry.correlation_dimension import _fixed_window_dimension
    X = _gauss(1, n=200, d=8)
    d = pdist(X)
    slope, r2, n_fit, status, _ = _fixed_window_dimension(d, 200)
    assert status == "ok" and n_fit == 99 - 10 + 1
    ds = np.sort(d)
    k = np.arange(1, ds.size + 1)
    sel = (k >= 10) & (k <= 99)
    ref = np.polyfit(np.log(ds[sel]), np.log(k[sel] / ds.size), 1)[0]
    assert slope == pytest.approx(ref, rel=1e-12)
