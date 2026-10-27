"""Shared helpers for the layer-similarity invariance tests (cka, hsic, rsa,
cknna, procrustes): real-model states, sink injection, identity insertion and
row-permutation nulls, all built on ``tests/invariance``."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_TESTS = Path(__file__).resolve().parents[2]
_SRC = _TESTS.parent / "src"
for _p in (str(_SRC), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import invariance as inv  # noqa: E402

TEXTS = [
    "The cat sat on the mat and looked out of the window at the rain.",
    "Stock markets fell sharply on Tuesday after the central bank raised rates.",
    "In 1905 Einstein published four papers that changed physics forever.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "She opened the old wooden door and stepped into the dusty library.",
    "The recipe calls for two cups of flour, one egg and a pinch of salt.",
    "Volcanoes form where tectonic plates diverge or where one plate subducts.",
    "The orchestra tuned their instruments while the audience took their seats.",
    "A prime number has exactly two distinct positive divisors.",
    "Rivers carry sediment downstream and deposit it in deltas near the sea.",
    "The committee will vote on the proposal at its meeting next Thursday.",
    "Bees communicate the location of flowers through a waggle dance.",
]


def load(name):
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float32).eval()
    except OSError as e:
        pytest.skip(f"{name} not in the local HF cache: {e}")
    return model, tok


def scale_position0(S: "inv.States", c: float) -> "inv.States":
    """Multiply every document's position-0 row by ``c`` in every entry
    (blocks, final norm, embedding): a synthetic attention sink."""
    def f(x):
        y = np.array(x, dtype=np.float64, copy=True)
        y[0] *= c
        return y
    return S.map(f)


def permute_layer_rows(S: "inv.States", layer: int, seed: int = 0) -> "inv.States":
    """Replace block ``layer`` by its own rows permuted across the whole cloud
    and re-split by document: same marginal, pairing with every other layer
    broken (a row-permutation null)."""
    lengths = S.lengths
    X = np.concatenate([np.asarray(x, dtype=np.float64) for x in S.blocks[layer]], 0)
    perm = np.random.default_rng(seed).permutation(X.shape[0])
    X = X[perm]
    cuts = np.cumsum(lengths)[:-1]
    blocks = dict(S.blocks)
    blocks[layer] = list(np.split(X, cuts, axis=0))
    final = S.final_norm
    if final is not None and layer == S.n_layers - 1:
        final = blocks[layer]
    return inv.States(blocks=blocks, final_norm=final, embedding=S.embedding)


def keep_layers(S: "inv.States", layers) -> "inv.States":
    """A States holding only ``layers`` (renumbered 0..k-1); the final norm is
    dropped unless the original last block is kept last."""
    layers = list(layers)
    blocks = {i: S.blocks[l] for i, l in enumerate(layers)}
    final = S.final_norm if layers[-1] == S.n_layers - 1 else None
    return inv.States(blocks=blocks, final_norm=final, embedding=S.embedding)


def feature_keys(result):
    """Headline (non-_meta_) numeric scalar keys of a task result."""
    return sorted(k for k, v in result.items() if not k.startswith("_meta_")
                  and isinstance(v, (int, float)) and not isinstance(v, bool))
