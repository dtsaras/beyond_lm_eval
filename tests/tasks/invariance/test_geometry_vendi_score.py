"""Direct mechanism tests for geometry_vendi_score, through the real
evaluate() on injected states (tests/invariance driver)."""
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
TASK = "geometry_vendi_score"
KEYS = ["vendi_mean_first_layer", "vendi_mean_last_layer", "vendi_max"]


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


def test_endpoints_and_max_invariant(tiny, states):
    """The cosine kernel depends only on angles: zero-padding and a
    power-of-two rescale are bit-identical, an isometric embedding agrees to
    the cache's float32 rounding; identity blocks after blocks 0 and 1 (exact)
    leave the first, last and maximum layer values bit-identical."""
    model, tok = tiny
    base = inv.run_task_on_states(TASK, states)
    inv.assert_invariant(base, inv.run_task_on_states(TASK, inv.zero_pad_docs(states, 2)),
                         KEYS, rtol=0.0)
    inv.assert_invariant(base, inv.run_task_on_states(TASK, inv.rescale_docs(states, 2.0)),
                         KEYS, rtol=0.0)
    inv.assert_invariant(base, inv.run_task_on_states(TASK, inv.embed_isometric_docs(states, 2, 3)),
                         KEYS, rtol=1e-6)
    ins = inv.run_task_on_states(TASK, inv.states_from_model(
        inv.insert_identity_blocks(model, [0, 1]), tok, TEXTS))
    inv.assert_invariant(base, ins, KEYS, rtol=0.0)


def test_replaced_aggregates_move_under_identity_insertion(tiny, states):
    """Known negative: the uniform layer mean, slope and the index-selected
    middle layer move under the same insertion."""
    model, tok = tiny
    base = inv.run_task_on_states(TASK, states)
    ins = inv.run_task_on_states(TASK, inv.states_from_model(
        inv.insert_identity_blocks(model, [0, 1]), tok, TEXTS))
    for key in ("_meta_vendi_overall_mean", "_meta_vendi_slope", "vendi_mean_mid_layer"):
        inv.assert_moves(base, ins, key, min_rel=1e-3)


def test_sink_diagnostics_are_emitted(states):
    out = inv.run_task_on_states(TASK, states)
    assert 0.0 <= out["_meta_position0_share"] <= 1.0
    assert 0.0 <= out["_meta_massive_token_share"] <= 1.0
    assert out["_meta_sink_n_depths"] >= 1
