"""Direct mechanism tests for geometry_mahalanobis through the real
evaluate(): a stub model maps each token to a fixed state, and the transform
is applied to every hidden state it returns (the task runs its own forward)."""
import sys
import types

import numpy as np
import torch

from blme.tasks.geometry.mahalanobis import MahalanobisOODTask

T, D, L, N_DOCS = 9, 256, 4, 200
PER_DIM = ["auroc", "ood_separation_ratio", "mahalanobis_sq_per_dim_id",
           "mahalanobis_sq_per_dim_ood", "ood_separation_gap_per_dim"]


class _Tok:
    all_special_ids = []
    eos_token_id = None
    sep_token_id = None

    def __call__(self, text, return_tensors=None, truncation=True, max_length=None, **kw):
        i = int(str(text).split()[-1])
        ids = [i * 13 + t for t in range(T)]
        if return_tensors == "pt":
            ids = torch.tensor([ids], dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        return {"input_ids": ids}

    def decode(self, ids):
        return "decoded"


class _Model(torch.nn.Module):
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform or (lambda h: h)
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=True, **kw):
        ids = input_ids[0].double()
        prev = torch.cat([ids[:1], ids[:-1]])
        # Context-dependent: each state mixes its token with the previous one,
        # so shuffling the tokens moves the pooled vector (word-order signal).
        f = torch.stack([torch.sin(0.37 * ids * (1 + 0.05 * k) + 0.3 * k)
                         + torch.cos(0.11 * (ids - prev) * (k + 1)) for k in range(D)], 1)
        f = torch.as_tensor(self.transform(f.numpy()))
        return types.SimpleNamespace(hidden_states=tuple([f.unsqueeze(0)] * (L + 1)))


def _run(transform=None):
    cfg = {"num_samples": N_DOCS, "n_splits": 3, "id_corpus": "dataset"}
    data = [{"text": f"sample {i}"} for i in range(N_DOCS)]
    return MahalanobisOODTask(cfg).evaluate(_Model(transform), _Tok(), data, cache=None)


def _rel(a, b):
    return abs(a - b) / max(abs(a), 1e-12)


def test_per_dim_forms_invariant():
    """n_fit = 100 << d = 256, the regime the dist^2/d forms are for: under
    zero-padding to 2d and an isometric embedding they move by < 5% (Ledoit-
    Wolf's isotropic target moves with the width), and a global rescale leaves
    them unchanged (the shrunk precision scales as 1/s^2). As n_fit nears d
    the width dependence returns (see the next test)."""
    base = _run()
    pad = _run(lambda h: np.concatenate([h, np.zeros_like(h)], 1))
    q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((2 * D, 2 * D)))
    emb = _run(lambda h: h @ q[:D])
    big = _run(lambda h: 4.0 * h)
    for k in PER_DIM:
        assert _rel(base[k], pad[k]) < 0.05, (k, base[k], pad[k])
        assert _rel(base[k], emb[k]) < 0.05, (k, base[k], emb[k])
        assert _rel(base[k], big[k]) < 1e-9, (k, base[k], big[k])


def test_raw_distances_grow_with_width():
    """Known negative: the raw distances the per-dim forms replace grow by
    about sqrt(2) when the cloud is zero-padded to twice the width."""
    base = _run()
    pad = _run(lambda h: np.concatenate([h, np.zeros_like(h)], 1))
    assert pad["mean_mahalanobis_id"] / base["mean_mahalanobis_id"] > 1.2


def test_per_dim_width_invariance_needs_n_fit_below_d(monkeypatch):
    """The limit of the claim, pinned: with 300 fit points in d = 256 the
    isotropic Ledoit-Wolf target dominates differently at 2d and the per-dim
    ID distance moves by > 10% under zero-padding. The study's n_fit = 1000
    is below d for d >= 2048 models and above it for d = 768-1024."""
    monkeypatch.setattr(sys.modules[__name__], "N_DOCS", 600)
    base = _run()
    pad = _run(lambda h: np.concatenate([h, np.zeros_like(h)], 1))
    assert _rel(base["mahalanobis_sq_per_dim_id"], pad["mahalanobis_sq_per_dim_id"]) > 0.1
