"""geometry_hubness: rows without a direction, the unembedding, emittable rows
and the pinned row count."""
import numpy as np
import pytest
import torch

from blme.tasks.geometry.hubness import GlobalHubnessTask

K = {"k_values": [10, 50]}


class _Tok:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class _Tables(torch.nn.Module):
    """A model stand-in exposing an input and an output embedding table."""

    def __init__(self, W_in, W_out=None):
        super().__init__()
        self.emb = torch.nn.Embedding.from_pretrained(torch.as_tensor(W_in))
        W_out = W_in if W_out is None else W_out
        self.head = torch.nn.Linear(W_out.shape[1], W_out.shape[0], bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.as_tensor(W_out))

    def get_input_embeddings(self):
        return self.emb

    def get_output_embeddings(self):
        return self.head


def _table(n=400, d=16, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


def _feature(out):
    return {k: v for k, v in out.items() if k.startswith("hubness_")}


def test_zero_rows_are_not_ranked():
    """Eight exactly-zero rows (no direction) leave every hubness key equal to
    the table without them; ranked, they became near-universal neighbours
    (dossier: pythia-160m k10_max 38 -> 8000)."""
    W = _table()
    Wz = np.concatenate([W, np.zeros((8, W.shape[1]), np.float32)])
    clean = GlobalHubnessTask(dict(K)).evaluate(_Tables(W), _Tok(len(W)), None)
    zero = GlobalHubnessTask(dict(K)).evaluate(_Tables(Wz), _Tok(len(Wz)), None)
    assert _feature(zero) == _feature(clean)
    assert zero["_meta_n_zero_rows"] == 8 and zero["_meta_n_rows_ranked"] == len(W)
