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


WIDTH_TRANSFORMS = {
    "zero_pad": lambda S: inv.zero_pad_docs(S, 2),
    "isometry": lambda S: inv.embed_isometric_docs(S, 2, 3),
    "rescale": lambda S: inv.rescale_docs(S, 2.0),
}


def check_size_invariant_entry(task, task_cls, key, states, identity_states,
                               null_states=None, negatives=None, rtol_width=1e-5,
                               null_bound=0.03, null_drift=0.1):
    """Run the direct test for every transform ``task_cls.SIZE_INVARIANT[key]``
    declares, and its known negative.

    * ``identity_blocks`` -- exact identity blocks inserted (``identity_states``):
      ``key`` bit-stable (rtol 1e-9); ``negatives["identity_blocks"]`` moves.
    * ``zero_pad`` / ``isometry`` / ``rescale`` -- ``key`` within ``rtol_width``.
    * ``N`` -- on a row-permutation null (population value 0), ``key`` stays
      within ``null_bound`` of 0 at 3 documents (two draws) and at all of them,
      while ``negatives["N"]`` (the form it replaced) drifts by > ``null_drift``.
    """
    transforms, node = task_cls.SIZE_INVARIANT[key]
    assert node.endswith(f"[{key}]"), node
    negatives = negatives or {}
    base = inv.run_task_on_states(task, states)
    for t in transforms:
        if t == "identity_blocks":
            ins = inv.run_task_on_states(task, identity_states)
            inv.assert_invariant(base, ins, [key], rtol=1e-9)
            if t in negatives:
                inv.assert_moves(base, ins, negatives[t], 1e-3)
        elif t in WIDTH_TRANSFORMS:
            other = inv.run_task_on_states(task, WIDTH_TRANSFORMS[t](states))
            inv.assert_invariant(base, other, [key], rtol=rtol_width)
        elif t == "N":
            small = [inv.run_task_on_states(task, inv.subsample_docs(null_states, 3, seed=s))
                     for s in (0, 1)]
            full = inv.run_task_on_states(task, null_states)
            for r in small + [full]:
                assert abs(r[key]) < null_bound, (key, r[key])
            neg = negatives["N"]
            drift = np.mean([r[neg] for r in small]) - full[neg]
            assert drift > null_drift, (neg, drift)
        else:
            raise AssertionError(f"no direct test for transform {t!r}")


def feature_keys(result):
    """Headline (non-_meta_) numeric scalar keys of a task result."""
    return sorted(k for k, v in result.items() if not k.startswith("_meta_")
                  and isinstance(v, (int, float)) and not isinstance(v, bool))
