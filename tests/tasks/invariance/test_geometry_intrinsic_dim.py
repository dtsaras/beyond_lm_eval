"""Direct mechanism tests for geometry_intrinsic_dim (layerwise), through the
real evaluate() on injected states (tests/invariance driver)."""
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
TASK = "geometry_intrinsic_dim"
ENDPOINTS = ["lid_first_layer", "lid_last_layer"]


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


def test_endpoints_invariant(tiny, states):
    """The first and last block's per-sequence TwoNN is a property of the
    cloud: zero-padding and a power-of-two rescale leave it bit-identical, a
    random isometric embedding to rounding, and identity blocks inserted after
    blocks 0 and 1 (exact) leave the endpoints bit-identical. The embedding
    is stored in float32 by the cache, so it agrees to float32 rounding."""
    model, tok = tiny
    base = inv.run_task_on_states(TASK, states)
    inv.assert_invariant(base, inv.run_task_on_states(TASK, inv.zero_pad_docs(states, 2)),
                         ENDPOINTS, rtol=0.0)
    inv.assert_invariant(base, inv.run_task_on_states(TASK, inv.rescale_docs(states, 2.0)),
                         ENDPOINTS, rtol=0.0)
    inv.assert_invariant(base, inv.run_task_on_states(TASK, inv.embed_isometric_docs(states, 2, 3)),
                         ENDPOINTS, rtol=1e-6)
    ins = inv.run_task_on_states(TASK, inv.states_from_model(
        inv.insert_identity_blocks(model, [0, 1]), tok, TEXTS))
    inv.assert_invariant(base, ins, ENDPOINTS, rtol=0.0)


def test_replaced_depth_summaries_move_under_identity_insertion(tiny, states):
    """Known negative: the layer slope (now _meta_) and the index-selected
    middle layer move when the same function is cut into more blocks."""
    model, tok = tiny
    base = inv.run_task_on_states(TASK, states)
    ins = inv.run_task_on_states(TASK, inv.states_from_model(
        inv.insert_identity_blocks(model, [0, 1]), tok, TEXTS))
    inv.assert_moves(base, ins, "_meta_lid_depth_slope", min_rel=1e-3)
    inv.assert_moves(base, ins, "lid_mid_layer", min_rel=1e-3)


def test_sink_diagnostics_are_emitted(states):
    """The token cloud follows the sink policy, and the shares of the centred
    sum of squares in position 0 / massive rows are reported."""
    base = inv.run_task_on_states(TASK, states)
    assert 0.0 <= base["_meta_position0_share"] <= 1.0
    assert 0.0 <= base["_meta_massive_token_share"] <= 1.0
    assert base["_meta_sink_n_depths"] >= 1 and base["_meta_token_cloud_n_docs"] == len(TEXTS)
