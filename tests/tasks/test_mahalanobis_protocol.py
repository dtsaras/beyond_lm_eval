"""geometry_mahalanobis: the sink policy in the mean-pool and the fit size."""
import types

import numpy as np
import pytest
import torch

from blme.tasks.geometry.mahalanobis import MahalanobisOODTask

T, D, L = 9, 6, 4
N_DOCS = 24


class _Tok:
    """Text "sample i" -> ids i*11 .. i*11 + T - 1; ids divisible by 5 are the
    stub's massive tokens."""
    all_special_ids = []
    eos_token_id = None
    sep_token_id = None

    def __call__(self, text, return_tensors=None, truncation=True, max_length=None, **kw):
        i = int(str(text).split()[-1])
        ids = [i * 11 + t for t in range(T)]
        if return_tensors == "pt":
            ids = torch.tensor([ids], dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        return {"input_ids": ids}

    def decode(self, ids):
        return "decoded"


class _Model(torch.nn.Module):
    """Final state of token (id, position) is a smooth function of both; at
    the middle blocks tokens with id % 5 == 0 are x1000 (massive). ``scale``
    multiplies the FINAL state of position 0 and of the massive tokens."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=True, **kw):
        ids = input_ids[0].double()
        pos = torch.arange(ids.numel(), dtype=torch.float64)
        f = torch.stack([torch.sin(0.37 * ids + 0.5 * k) + 0.1 * torch.cos(pos + k)
                         for k in range(D)], 1)
        sink = (input_ids[0] % 5 == 0) | (pos == 0)
        block = f.clone()
        block[input_ids[0] % 5 == 0] *= 1000.0
        final = f.clone()
        final[sink] *= self.scale
        hs = tuple([0.5 * f.unsqueeze(0)] + [block.unsqueeze(0)] * (L - 1) + [final.unsqueeze(0)])
        return types.SimpleNamespace(hidden_states=hs)


def _run(scale, cfg=None):
    cfg = {"num_samples": N_DOCS, "ood_strategy": "shuffle", "n_splits": 2,
           "id_corpus": "dataset", **(cfg or {})}
    data = [{"text": f"sample {i}"} for i in range(2 * N_DOCS)]
    return MahalanobisOODTask(cfg).evaluate(_Model(scale), _Tok(), data, cache=None)


def test_sinks_are_left_out_of_the_mean_pool():
    """Rescaling the final state of position 0 and of the massive tokens x50
    leaves every score unchanged: those rows are not in any mean."""
    base, moved = _run(1.0), _run(50.0)
    keys = ["auroc", "mean_mahalanobis_id", "mean_mahalanobis_ood",
            "mahalanobis_sq_per_dim_id", "ood_separation_ratio"]
    assert all(np.isfinite(base[k]) for k in keys)
    for k in keys:
        assert base[k] == moved[k], (k, base[k], moved[k])
    assert base["_meta_sink_n_massive_dropped"] > 0
