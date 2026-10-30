"""geometry_lid: one point per sequence (Q1) and the demoted lid_min."""
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
TASK = "geometry_lid"
MODES = [{"k": 5}, {"k": 5, "layerwise": True}]


@pytest.fixture(scope="module")
def states():
    name = "hf-internal-testing/tiny-random-gpt2"
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name)
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"{name} not cached: {e}")
    return inv.states_from_model(model, tok, TEXTS)


def _replace_all_but_last(S, seed=0):
    """Every row of every document except its last, replaced by noise."""
    rng = np.random.default_rng(seed)

    def edit(x):
        y = np.array(x, dtype=np.float64, copy=True)
        y[:-1] = rng.standard_normal(y[:-1].shape) * np.abs(y).max()
        return y
    return inv.States(blocks={k: [edit(x) for x in v] for k, v in S.blocks.items()},
                      final_norm=None if S.final_norm is None else [edit(x) for x in S.final_norm])


@pytest.mark.parametrize("mode", MODES, ids=["single", "layerwise"])
def test_headline_is_one_point_per_sequence(states, mode):
    """Q1: LID of the last content token of each document, neighbours from the
    other documents. Rewriting every other row leaves it bit-identical (the
    token cloud moves), and the point count is the document count."""
    base = inv.run_task_on_states(TASK, states, mode)
    edited = inv.run_task_on_states(TASK, _replace_all_but_last(states), mode)
    keys = [k for k in ("lid_mean", "lid_std", "lid_max", "lid_median", "lid_first_layer",
                        "lid_mid_layer", "layer_lid_mean") if k in base]
    assert len(keys) >= 4
    inv.assert_invariant(base, edited, keys, rtol=0.0)
    assert base["_meta_n_points"] == len(TEXTS)
    inv.assert_moves(base, edited, "_meta_tokens_lid_mean", min_rel=1e-3)


@pytest.mark.parametrize("mode", MODES, ids=["single", "layerwise"])
def test_lid_min_is_meta(states, mode):
    """P7: one duplicated document makes the minimum exactly 0 (a duplicate
    query scores -k/(-inf) = 0, as the released mle_batch does), so lid_min
    reports duplicate presence, not geometry, and is _meta_."""
    S = states.select_docs(list(range(len(TEXTS))) + [0])
    out = inv.run_task_on_states(TASK, S, mode)
    assert "lid_min" not in out
    assert out["_meta_lid_min"] == 0.0
    assert out["_meta_num_zero_estimates"] >= 1
