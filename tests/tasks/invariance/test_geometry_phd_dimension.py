"""Direct mechanism tests for geometry_phd_dimension through the real
evaluate(). The task runs its own per-text forward pass, so a stub model
serves a fixed cloud per text and the transform is applied to that cloud."""
import types

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from blme.tasks.geometry.phd_dimension import PHDimensionTask  # noqa: E402

T_TOK, D, N_TEXTS = 140, 24, 5
CFG = {"documents": "dataset", "num_samples": N_TEXTS, "max_length": 512, "n_reruns": 1}
KEYS = ["phd_dimension", "phd_dimension_std"]


class _Tok:
    eos_token_id = None
    sep_token_id = None

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=None, **kw):
        i = int(str(text).split()[-1])
        T = T_TOK if max_length is None else min(T_TOK, int(max_length))
        ids = torch.full((1, T), i, dtype=torch.long)
        return transformers.BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})


class _CloudModel(torch.nn.Module):
    """Text i -> a fixed curved cloud, mapped by ``transform`` (an (T, D) ->
    (T, D') function standing in for a function-preserving change of the
    representation)."""

    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform or (lambda h: h)
        self._anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=True, **kw):
        i, T = int(input_ids[0, 0]), input_ids.shape[1]
        g = torch.Generator().manual_seed(11 + i)
        z = torch.rand(T, 4, generator=g, dtype=torch.float64)
        h = torch.cat([z, torch.cos(2 * z), torch.zeros(T, D - 8, dtype=torch.float64)], 1)
        h = torch.as_tensor(self.transform(h.numpy()), dtype=torch.float64).unsqueeze(0)
        return types.SimpleNamespace(hidden_states=(0.5 * h, h))


def _run(transform=None, cfg=CFG):
    data = [{"text": f"doc {i}"} for i in range(N_TEXTS)]
    return PHDimensionTask(dict(cfg)).evaluate(_CloudModel(transform), _Tok(), data, cache=None)


def test_per_text_phd_invariant():
    """Per-text PHD depends only on distances: zero-padding is bit-identical,
    a random isometric embedding into 2D agrees to rounding, and a global
    rescale x10 leaves it unchanged (MST lengths scale, the log-log slope
    does not)."""
    base = _run()
    assert np.isfinite(base["phd_dimension"])
    inv_pad = _run(lambda h: np.concatenate([h, np.zeros_like(h)], 1))
    q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((2 * D, 2 * D)))
    inv_emb = _run(lambda h: h @ q[:D])
    inv_big = _run(lambda h: 10.0 * h)
    for other, tol in ((inv_pad, 0.0), (inv_emb, 1e-9), (inv_big, 1e-9)):
        for k in KEYS:
            assert abs(other[k] - base[k]) <= tol * abs(base[k]), (k, base[k], other[k])


def test_text_length_moves_the_estimate():
    """Known negative: at fixed geometry the per-text estimate moves with the
    text length (the regime short passages put the study in), which is why
    texts are >= 300-token articles truncated at one length for every model."""
    long_, short = _run(), _run(cfg={**CFG, "max_length": 70})
    assert abs(short["phd_dimension"] - long_["phd_dimension"]) > 1e-3 * abs(long_["phd_dimension"])
