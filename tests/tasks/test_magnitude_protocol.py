"""geometry_magnitude: one point per sequence (Q1) and the magnipy protocol."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
TASK = "geometry_magnitude"


@pytest.fixture(scope="module")
def tiny():
    name = "hf-internal-testing/tiny-random-gpt2"
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name)
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"{name} not cached: {e}")
    return model, tok


@pytest.fixture(scope="module")
def states(tiny):
    return inv.states_from_model(*tiny, TEXTS)


def _replace_all_but_last(S, seed=0):
    """Every row of every document except its last, replaced by noise."""
    rng = np.random.default_rng(seed)

    def edit(x):
        y = np.array(x, dtype=np.float64, copy=True)
        y[:-1] = rng.standard_normal(y[:-1].shape) * np.abs(y).max()
        return y
    return inv.States(blocks={k: [edit(x) for x in v] for k, v in S.blocks.items()},
                      final_norm=None if S.final_norm is None else [edit(x) for x in S.final_norm])


def _feature(out):
    return [k for k, v in out.items() if not k.startswith("_meta_")
            and isinstance(v, float) and np.isfinite(v)]


def test_headline_is_one_point_per_sequence(states):
    """Q1: magnitude of the last content token of each document at the output
    representation. Rewriting every other row leaves every feature key
    bit-identical (the token cloud moves); the point count is the document
    count."""
    base = inv.run_task_on_states(TASK, states)
    edited = inv.run_task_on_states(TASK, _replace_all_but_last(states))
    keys = _feature(base)
    assert len(keys) >= 2
    inv.assert_invariant(base, edited, keys, rtol=0.0)
    assert base["_meta_n_points"] == base["_meta_magnitude_n_points"] == len(TEXTS)
    moved = [k for k in base if k.startswith("_meta_tokens_magnitude")
             and inv.rel_change(base[k], edited[k]) > 1e-3]
    assert moved
