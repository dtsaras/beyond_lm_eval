"""Point-cloud transforms for direct mechanism tests (numpy and torch).

Exact transforms keep the input dtype: ``zero_pad``, ``subsample_rows``.
Transforms that round (``embed_isometric``, ``rescale``) compute and return
float64, so the transformed cloud carries no storage error of its own. The
injected cache stores float32 by default, as the real cache does; pass
``store_dtype=torch.float64`` to ``run_task_on_states`` to keep it.

Every ``*_docs`` variant applies ONE transform to every array it is given (all
documents, all layers, the final norm and the embedding), so distances within
and across layers are preserved together. It accepts an array, a list of
arrays, a ``{key: [arrays]}`` dict or a ``States``.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
import torch

from .states import States


def _is_torch(x) -> bool:
    return isinstance(x, torch.Tensor)


@lru_cache(maxsize=32)
def _isometry_np(d: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((k * d, d))
    Q, R = np.linalg.qr(A)                 # (kd, d), orthonormal columns
    Q = Q * np.sign(np.diag(R))            # Haar-distributed
    M = np.ascontiguousarray(Q.T)          # (d, kd), orthonormal rows
    M.setflags(write=False)
    return M


def random_isometry(d: int, k: int, seed: int = 0) -> np.ndarray:
    """(d, k*d) float64 matrix with orthonormal rows: the first d rows of a
    Haar-random orthogonal matrix of size k*d. ``H @ M`` is an isometry."""
    if int(k) != k or k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")
    return _isometry_np(int(d), int(k), int(seed))


def embed_isometric(H, k: int, seed: int = 0):
    """H @ Q[:d] for a random orthogonal Q of size k*d. Returns float64."""
    M = random_isometry(H.shape[-1], k, seed)
    if _is_torch(H):
        return H.detach().to(torch.float64) @ torch.tensor(M, device=H.device)
    return np.asarray(H, dtype=np.float64) @ M


def zero_pad(H, k: int):
    """[H, 0, ..., 0] at width k*d. Exact; dtype preserved."""
    if int(k) != k or k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")
    k = int(k)
    if _is_torch(H):
        pad = H.new_zeros(H.shape[:-1] + ((k - 1) * H.shape[-1],))
        return torch.cat([H, pad], dim=-1)
    H = np.asarray(H)
    pad = np.zeros(H.shape[:-1] + ((k - 1) * H.shape[-1],), dtype=H.dtype)
    return np.concatenate([H, pad], axis=-1)


def _row_subset(n_total: int, n: int, seed: int) -> np.ndarray:
    if not 0 < n <= n_total:
        raise ValueError(f"cannot take {n} of {n_total} rows")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n, replace=False))


def subsample_rows(H, n: int, seed: int = 0):
    """n rows drawn without replacement, kept in their original order."""
    idx = _row_subset(H.shape[0], int(n), seed)
    if _is_torch(H):
        return H[torch.from_numpy(idx).to(H.device)]
    return np.asarray(H)[idx]


def rescale(H, c: float):
    """c * H in float64."""
    if _is_torch(H):
        return H.detach().to(torch.float64) * float(c)
    return np.asarray(H, dtype=np.float64) * float(c)


# --------------------------------------------------------------------------
# Per-document versions
# --------------------------------------------------------------------------

def _first_array(obj):
    if isinstance(obj, States):
        return obj.blocks[0][0]
    if isinstance(obj, dict):
        return next(iter(obj.values()))[0]
    if isinstance(obj, (list, tuple)):
        return obj[0]
    return obj


def _map_arrays(obj, fn):
    if isinstance(obj, States):
        return obj.map(fn)
    if isinstance(obj, dict):
        return {k: _map_arrays(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [fn(x) for x in obj]
    return fn(obj)


def embed_isometric_docs(docs, k: int, seed: int = 0):
    """``embed_isometric`` with one Q shared by every array."""
    d = _first_array(docs).shape[-1]
    random_isometry(d, k, seed)            # build once
    return _map_arrays(docs, lambda x: embed_isometric(x, k, seed))


def zero_pad_docs(docs, k: int):
    return _map_arrays(docs, lambda x: zero_pad(x, k))


def rescale_docs(docs, c: float):
    return _map_arrays(docs, lambda x: rescale(x, c))


def subsample_docs(docs, n: int, seed: int = 0):
    """Keep n documents (the same ones in every layer), in original order."""
    if isinstance(docs, States):
        return docs.select_docs(_row_subset(docs.n_docs, int(n), seed))
    if isinstance(docs, dict):
        n_total = len(next(iter(docs.values())))
        idx = _row_subset(n_total, int(n), seed)
        return {k: [v[i] for i in idx] for k, v in docs.items()}
    idx = _row_subset(len(docs), int(n), seed)
    return [docs[i] for i in idx]


def pool_docs(docs: Sequence):
    """Concatenate a list of (T_i, d) arrays into one (sum T_i, d) cloud."""
    if _is_torch(docs[0]):
        return torch.cat(list(docs), dim=0)
    return np.concatenate([np.asarray(x) for x in docs], axis=0)
