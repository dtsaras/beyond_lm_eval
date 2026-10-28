"""Direct mechanism tests for geometry_magnitude, through the real evaluate()
on injected states (tests/invariance driver)."""
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
TASK = "geometry_magnitude"
KEYS = ["magnitude_area_tconv", "magnitude_dimension_tconv"]


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


def test_headline_invariant(tiny, states):
    """Cosine distances on [0, t_conv]: zero-padding and rescaling (x2 and x10)
    leave MagArea/t_conv and the magnitude dimension unchanged, an isometric
    embedding to the cache's float32 rounding; identity blocks after blocks 0
    and 1 (exact) leave the output representation, so the headline, unchanged."""
    model, tok = tiny
    base = inv.run_task_on_states(TASK, states)
    inv.assert_invariant(base, inv.run_task_on_states(TASK, inv.zero_pad_docs(states, 2)),
                         KEYS, rtol=1e-12)
    inv.assert_invariant(base, inv.run_task_on_states(TASK, inv.rescale_docs(states, 2.0)),
                         KEYS, rtol=1e-12)
    inv.assert_invariant(base, inv.run_task_on_states(TASK, inv.rescale_docs(states, 10.0)),
                         KEYS, rtol=1e-6)
    inv.assert_invariant(base, inv.run_task_on_states(TASK, inv.embed_isometric_docs(states, 2, 3)),
                         KEYS, rtol=1e-6)
    ins = inv.run_task_on_states(TASK, inv.states_from_model(
        inv.insert_identity_blocks(model, [0, 1]), tok, TEXTS))
    inv.assert_invariant(base, ins, KEYS, rtol=0.0)


def test_replaced_ladder_moves_under_rescale(states):
    """Known negative: the median-heuristic ladder keys it replaced (now
    _meta_) are not scale-free -- the median scale goes as 1/sqrt(s) and the
    median-window area as 1/s under a x10 rescale."""
    base = inv.run_task_on_states(TASK, states)
    big = inv.run_task_on_states(TASK, inv.rescale_docs(states, 10.0))
    for key in ("_meta_magnitude_median_scale", "_meta_magnitude_area",
                "_meta_magnitude_low_scale"):
        inv.assert_moves(base, big, key, min_rel=0.1)


def test_sink_diagnostics_are_emitted(states):
    out = inv.run_task_on_states(TASK, states)
    assert 0.0 <= out["_meta_position0_share"] <= 1.0
    assert 0.0 <= out["_meta_massive_token_share"] <= 1.0
    assert out["_meta_n_points"] == len(TEXTS)
