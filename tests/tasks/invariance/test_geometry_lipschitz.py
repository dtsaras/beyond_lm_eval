"""Direct mechanism tests for geometry_lipschitz, through the real evaluate()
on injected states (tests/invariance driver)."""
import sys
from pathlib import Path

import numpy as np
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
]
TASK = "geometry_lipschitz"


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


def _with_position0(S, fn):
    """Apply ``fn`` to the position-0 row of every document at every block."""
    def edit(x):
        y = np.array(x, copy=True)
        y[0] = fn(y[0])
        return y
    return S.with_blocks({k: [edit(x) for x in v] for k, v in S.blocks.items()})


def test_position0_is_excluded(states):
    """A position-0 sink (x40 at every block) leaves every scored key alone;
    with position 0 in the rows it moved them."""
    base = inv.run_task_on_states(TASK, states)
    sunk = inv.run_task_on_states(TASK, _with_position0(states, lambda r: 40.0 * r))
    keys = [k for k in ("relative_change_mean", "_meta_relative_change_mean",
                        "relative_change_max", "e2e_log_norm_growth", "total_angular_path",
                        "embedding_relative_change") if k in base]
    assert len(keys) >= 2
    inv.assert_invariant(base, sunk, keys, rtol=0.0)
    assert base["_meta_rows_policy"] == "exclude_pos0"
    inv.assert_moves(base, sunk, "_meta_relative_change_mean_incl_pos0", min_rel=1e-3)
