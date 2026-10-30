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


def test_headline_is_the_unembedding():
    """Nielsen et al. 2025 measure hubness on the unembedding rows: with
    untied tables the headline equals the unembedding's own hubness and the
    input-embedding table is reported under _meta_input_embedding_*."""
    W_in, W_out = _table(seed=1), _table(seed=2) ** 3
    untied = GlobalHubnessTask(dict(K)).evaluate(_Tables(W_in, W_out), _Tok(len(W_in)), None)
    on_out = GlobalHubnessTask(dict(K)).evaluate(_Tables(W_out), _Tok(len(W_in)), None)
    on_in = GlobalHubnessTask(dict(K)).evaluate(_Tables(W_in), _Tok(len(W_in)), None)
    assert _feature(untied) == _feature(on_out) != _feature(on_in)
    assert untied["_meta_input_embedding_hubness_k10_skew"] == on_in["hubness_k10_skew"]
    assert untied["_meta_hubness_matrix"] == "unembedding" and not untied["_meta_tied_embeddings"]
    assert on_in["_meta_tied_embeddings"]


def test_unused_rows_are_excluded():
    """Rows past len(tokenizer) are never emitted (untrained padding): 40
    init-like rows appended to a 400-row table leave every key unchanged."""
    W = _table()
    pad = (0.02 * np.random.default_rng(9).standard_normal((40, W.shape[1]))).astype(np.float32)
    base = GlobalHubnessTask(dict(K)).evaluate(_Tables(W), _Tok(len(W)), None)
    padded = GlobalHubnessTask(dict(K)).evaluate(_Tables(np.concatenate([W, pad])), _Tok(len(W)), None)
    assert _feature(padded) == _feature(base)
    assert padded["_meta_n_unused_rows"] == 40 and padded["_meta_n_emittable_rows"] == len(W)


def test_row_count_is_pinned():
    """S_Nk grows with the number of rows, so a fixed seeded draw of n_rows
    emittable ids is ranked: a 900-row table at n_rows=400 equals the 400 drawn
    rows scored on their own."""
    W = _table(n=900, seed=4)
    cfg = {**K, "n_rows": 400}
    out = GlobalHubnessTask(cfg).evaluate(_Tables(W), _Tok(len(W)), None)
    idx = np.sort(np.random.default_rng(0).choice(900, 400, replace=False))
    ref = GlobalHubnessTask(cfg).evaluate(_Tables(W[idx]), _Tok(400), None)
    assert out["_meta_n_rows_ranked"] == 400 and out["_meta_rows_subsampled"]
    assert _feature(out) == _feature(ref)
