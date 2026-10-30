"""Direct mechanism tests for geometry_hubness through the real evaluate() on
stub unembedding tables (the task reads weights only)."""
import numpy as np
import torch

from blme.tasks.geometry.hubness import GlobalHubnessTask

N, D = 600, 16
CFG = {"k_values": [10, 50, 100], "n_rows": 500}


class _Tok:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class _Tables(torch.nn.Module):
    def __init__(self, W):
        super().__init__()
        W = torch.as_tensor(W)
        self.emb = torch.nn.Embedding.from_pretrained(W)
        self.head = torch.nn.Linear(W.shape[1], W.shape[0], bias=False)
        with torch.no_grad():
            self.head.weight.copy_(W)

    def get_input_embeddings(self):
        return self.emb

    def get_output_embeddings(self):
        return self.head


def _run(W, n_tok=N, cfg=CFG):
    out = GlobalHubnessTask(dict(cfg)).evaluate(_Tables(W), _Tok(n_tok), None)
    return {k: v for k, v in out.items() if k.startswith("hubness_")}


def _table(seed=0):
    # Anisotropic, so hubs exist.
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((N, D)) * np.linspace(3, 0.2, D) + 0.5).astype(np.float64)


def test_width_scale_and_vocabulary_invariant():
    """Zero-padding, a power-of-two rescale and 64 appended unused rows (past
    len(tokenizer)) leave every key bit-identical; a random isometric
    embedding leaves the k-NN ranking, hence every key, unchanged."""
    W = _table()
    base = _run(W)
    assert _run(np.concatenate([W, np.zeros_like(W)], 1)) == base
    assert _run(2.0 * W) == base
    pad = 0.02 * np.random.default_rng(5).standard_normal((64, D))
    assert _run(np.concatenate([W, pad]), n_tok=N) == base
    q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((2 * D, 2 * D)))
    emb = _run(W @ q[:D])
    for k in base:
        assert abs(emb[k] - base[k]) <= 1e-9 * max(1.0, abs(base[k])), (k, base[k], emb[k])


def test_unpinned_row_count_moves_the_value():
    """Known negative: with the full vocabulary ranked, the same distribution
    at more rows reads differently -- why the row count is pinned."""
    W = _table()
    unpinned_small = _run(W[:300], n_tok=300, cfg={**CFG, "n_rows": 0})
    unpinned_full = _run(W, cfg={**CFG, "n_rows": 0})
    assert unpinned_small["hubness_k10_max"] != unpinned_full["hubness_k10_max"]
