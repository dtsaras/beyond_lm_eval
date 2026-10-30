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


def _with_position0(S, fn, block=1):
    """Apply ``fn`` to the position-0 row of every document at one block."""
    def edit(x):
        y = np.array(x, copy=True)
        y[0] = fn(y[0])
        return y
    blocks = dict(S.blocks)
    blocks[block] = [edit(x) for x in S.blocks[block]]
    return S.with_blocks(blocks)


def test_position0_is_excluded(states):
    """A position-0 sink (x40 at one block) leaves every scored key alone;
    with position 0 in the rows it moved them."""
    base = inv.run_task_on_states(TASK, states)
    sunk = inv.run_task_on_states(TASK, _with_position0(states, lambda r: 40.0 * r))
    keys = [k for k in ("relative_change_mean", "_meta_relative_change_mean",
                        "relative_change_max", "e2e_log_norm_growth", "total_angular_path",
                        "embedding_relative_change") if k in base]
    assert len(keys) >= 2
    inv.assert_invariant(base, sunk, keys, rtol=0.0)
    assert base["_meta_rows_policy"] == "exclude_sinks"
    inv.assert_moves(base, sunk, "_meta_relative_change_mean_incl_pos0", min_rel=1e-3)


@pytest.fixture(scope="module")
def gpt2_states():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "gpt2", dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"gpt2 not cached: {e}")
    return inv.states_from_model(model, tok, TEXTS)


def test_massive_row_at_one_middle_depth_is_dropped_at_every_layer(gpt2_states, monkeypatch):
    """A first-period-style sink (row 3 of every document x1000 at one middle
    block only, the one the shares are read on) dominates two per-update means
    and moves exclude_pos0's relative_change_max; exclude_sinks drops that
    token at every layer, embedding included, giving the row-deleted corpus's
    value bit for bit, and with no massive row it equals exclude_pos0 exactly."""
    from _sinks import check_state_task
    from blme.tasks.geometry.lipschitz import LipschitzContinuityTask
    # Entry n // 2 of [embedding, block 0, ..., block L-1] is block (L+1)//2 - 1.
    block = (gpt2_states.n_layers + 1) // 2 - 1
    check_state_task(TASK, LipschitzContinuityTask, gpt2_states, monkeypatch,
                     "relative_change_max", block=block,
                     keys=list(LipschitzContinuityTask.HEADLINE_KEYS))


def test_embedding_transition_is_reported_separately(states):
    """Rescaling the embedding output (an embedding-scale convention, e.g.
    Gemma's sqrt(d)) moves only embedding_relative_change; the block keys,
    whose max was the embedding update on every model measured, stay."""
    import dataclasses
    base = inv.run_task_on_states(TASK, states)
    emb = dataclasses.replace(states, embedding=[30.0 * np.asarray(x) for x in states.embedding])
    big = inv.run_task_on_states(TASK, emb)
    keys = [k for k in ("relative_change_max", "relative_change_mean", "_meta_relative_change_mean",
                        "e2e_log_norm_growth", "total_angular_path") if k in base]
    inv.assert_invariant(base, big, keys, rtol=0.0)
    inv.assert_moves(base, big, "embedding_relative_change", min_rel=1e-2)
    assert base["_meta_embedding_transition_available"] is True


def test_depth_additive_keys_invariant_under_identity_insertion(tiny, states):
    """Identity blocks (exact) after blocks 0, 1 and 3: e2e_log_norm_growth and
    total_angular_path are bit-identical, while the uniform layer mean of the
    relative change falls as (L-1)/(L-1+3) (the known negative)."""
    model, tok = tiny
    S_new = inv.states_from_model(inv.insert_identity_blocks(model, [0, 1, 3]), tok, TEXTS)
    base = inv.run_task_on_states(TASK, states)
    ins = inv.run_task_on_states(TASK, S_new)
    inv.assert_invariant(base, ins, ["e2e_log_norm_growth", "total_angular_path",
                                     "relative_change_max", "embedding_relative_change"], rtol=0.0)
    mean_key = "relative_change_mean" if "relative_change_mean" in base else "_meta_relative_change_mean"
    L1 = states.n_layers - 1
    assert ins[mean_key] / base[mean_key] == pytest.approx(L1 / (L1 + 3), rel=1e-12)


def test_e2e_log_norm_growth_invariant_under_redecomposition(tiny, states):
    """Splitting block 2 into attention-only and MLP-only blocks keeps the
    end-to-end map: e2e_log_norm_growth does not move. The angular path, which
    now passes through the mid-block state, may (it is not claimed here)."""
    model, tok = tiny
    S_new = inv.states_from_model(inv.redecompose(model, 2), tok, TEXTS)
    base = inv.run_task_on_states(TASK, states)
    split = inv.run_task_on_states(TASK, S_new)
    inv.assert_invariant(base, split, ["e2e_log_norm_growth"], rtol=1e-12)
