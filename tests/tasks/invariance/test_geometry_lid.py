"""Direct mechanism tests for geometry_lid, through the real evaluate() on
injected states (tests/invariance driver)."""
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
# k below the 12 per-sequence points of the fixture (the default is 20 of 1000).
SINGLE = {"k": 5}
LAYERWISE = {"k": 5, "layerwise": True}
KEYS = ["lid_mean", "lid_std", "lid_max", "lid_median"]


@pytest.fixture(scope="module")
def tiny():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-gpt2", dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"tiny-random-gpt2 not cached: {e}")
    return model, tok


@pytest.fixture(scope="module")
def states(tiny):
    return inv.states_from_model(*tiny, TEXTS)


@pytest.mark.parametrize("mode", [SINGLE, LAYERWISE], ids=["single", "layerwise"])
def test_width_scale_and_depth_invariant(tiny, states, mode):
    """LID depends only on distance ratios: zero-padding and a power-of-two
    rescale are bit-identical, a random isometric embedding agrees to the
    cache's float32 rounding. Identity blocks after blocks 0 and 1 (exact)
    leave the output representation, the deepest block and the first block
    unchanged."""
    model, tok = tiny
    keys = KEYS + (["lid_first_layer"] if mode is LAYERWISE else [])
    base = inv.run_task_on_states(TASK, states, mode)
    inv.assert_invariant(base, inv.run_task_on_states(TASK, inv.zero_pad_docs(states, 2), mode),
                         keys, rtol=0.0)
    inv.assert_invariant(base, inv.run_task_on_states(TASK, inv.rescale_docs(states, 2.0), mode),
                         keys, rtol=0.0)
    inv.assert_invariant(base, inv.run_task_on_states(
        TASK, inv.embed_isometric_docs(states, 2, 3), mode), keys, rtol=1e-6)
    ins = inv.run_task_on_states(TASK, inv.states_from_model(
        inv.insert_identity_blocks(model, [0, 1]), tok, TEXTS), mode)
    inv.assert_invariant(base, ins, keys, rtol=0.0)


def test_mid_layer_moves_under_identity_insertion(tiny, states):
    """Known negative: the index-selected middle layer, and the uniform mean of
    the layer profile, move when the same function is cut into more blocks."""
    model, tok = tiny
    base = inv.run_task_on_states(TASK, states, LAYERWISE)
    ins = inv.run_task_on_states(TASK, inv.states_from_model(
        inv.insert_identity_blocks(model, [0, 1]), tok, TEXTS), LAYERWISE)
    inv.assert_moves(base, ins, "lid_mid_layer", min_rel=1e-3)
    assert inv.rel_change(inv.uniform_layer_mean(base["layer_lid_mean"]),
                          inv.uniform_layer_mean(ins["layer_lid_mean"])) > 1e-3


def test_sink_diagnostics_are_emitted(states):
    out = inv.run_task_on_states(TASK, states, SINGLE)
    assert 0.0 <= out["_meta_position0_share"] <= 1.0
    assert 0.0 <= out["_meta_massive_token_share"] <= 1.0
    assert out["_meta_n_points"] == len(TEXTS) and out["_meta_token_cloud_n_docs"] == len(TEXTS)
