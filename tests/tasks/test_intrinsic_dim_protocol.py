"""geometry_intrinsic_dim: cache gate and the one-point-per-sequence protocol."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import invariance as inv  # noqa: E402
from blme.cache import ModelOutputCache  # noqa: E402
from blme.tasks.geometry.intrinsic_dim import IntrinsicDimensionTask  # noqa: E402

TEXTS = [
    "The cat sat on the mat and looked out of the window at the rain.",
    "Stock markets fell sharply on Tuesday after the central bank raised rates.",
    "In 1905 Einstein published four papers that changed physics forever.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "She opened the old wooden door and stepped into the dusty library.",
    "The recipe calls for two cups of flour, one egg and a pinch of salt.",
    "Volcanoes form where tectonic plates diverge or where one plate subducts.",
    "The orchestra tuned their instruments while the audience took their seats.",
]


@pytest.fixture(scope="module")
def tiny():
    name = "hf-internal-testing/tiny-random-gpt2"
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name)
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"{name} not cached: {e}")
    return model, tok


def test_cache_without_hidden_states(tiny):
    """A cache populated for a logits-only consumer returns no hidden states;
    the layerwise route self-collects instead of raising AttributeError."""
    model, tok = tiny
    data = [{"text": t} for t in TEXTS]
    cache = ModelOutputCache(model, tok, dataset=data, num_samples=len(data))
    cache.populate(need_hidden=False, need_logits=True)
    assert cache.is_populated and cache.get_hidden_states(layer_idx="all") is None
    out = IntrinsicDimensionTask({"layerwise": True, "num_samples": len(data)}).evaluate(
        model, tok, data, cache=cache)
    assert "error" not in out and "lid_first_layer" in out
    assert out["_meta_points_source"] == "self_collected"
    assert out["_meta_cache_rejected"] == "no_hidden_states"


def _replace_all_but_last(S, seed=0):
    """Every row of every document except its last, replaced by noise."""
    rng = np.random.default_rng(seed)

    def edit(x):
        y = np.array(x, dtype=np.float64, copy=True)
        y[:-1] = rng.standard_normal(y[:-1].shape) * np.abs(y).max()
        return y
    return inv.States(blocks={k: [edit(x) for x in v] for k, v in S.blocks.items()},
                      final_norm=None if S.final_norm is None else [edit(x) for x in S.final_norm])


def test_headline_is_one_point_per_sequence(tiny):
    """Q1: the profile is TwoNN on the last content token of each document.
    Rewriting every other row leaves it bit-identical (the token cloud moves),
    and the point count is the document count."""
    model, tok = tiny
    S = inv.states_from_model(model, tok, TEXTS)
    base = inv.run_task_on_states("geometry_intrinsic_dim", S)
    edited = inv.run_task_on_states("geometry_intrinsic_dim", _replace_all_but_last(S))
    keys = ["lid_first_layer", "lid_mid_layer", "lid_last_layer", "lid_trend"]
    inv.assert_invariant(base, edited, keys, rtol=0.0)
    assert base["_meta_n_points"] == len(TEXTS)
    assert base["_meta_points_readout"] == "last_content_token"
    inv.assert_moves(base, edited, "_meta_tokens_lid_trend", min_rel=1e-3)


def test_depth_slope_is_a_meta_profile_summary(tiny):
    """P4: an identity block (exact) after blocks 0 and 1 moves the OLS slope
    over layers, so it is _meta_; the endpoints do not move."""
    model, tok = tiny
    base = inv.run_task_on_states("geometry_intrinsic_dim", inv.states_from_model(model, tok, TEXTS))
    ins = inv.run_task_on_states("geometry_intrinsic_dim", inv.states_from_model(
        inv.insert_identity_blocks(model, [0, 1]), tok, TEXTS))
    assert "lid_depth_slope" not in base
    inv.assert_moves(base, ins, "_meta_lid_depth_slope", min_rel=1e-3)
    inv.assert_invariant(base, ins, ["lid_first_layer", "lid_last_layer"], rtol=0.0)
