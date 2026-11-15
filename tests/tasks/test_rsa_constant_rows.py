"""geometry_rsa: a row exactly constant at one layer is dropped from every
correlation RDM instead of turning the layer pairs that touch it into NaN."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from _family_d import inv  # noqa: E402

TASK = "geometry_rsa"
HEADLINE = ("rsa_adjacent_min_corrdist", "rsa_early_late_corrdist")
CORR_KEYS = HEADLINE + ("_meta_rsa_adjacent_mean_corrdist", "_meta_rsa_first_middle_corrdist")
DOC, ROW = 1, 4  # a mid-sentence token: neither position 0 nor a massive row


@pytest.fixture(scope="module")
def states():
    return inv.states_from_model(fd.tiny_gpt2(), fd.tokenizer(), fd.TEXTS)


def _with_constant_row(S, layer):
    blocks = {li: [np.array(x, dtype=np.float32, copy=True) for x in docs]
              for li, docs in S.blocks.items()}
    blocks[layer][DOC][ROW] = blocks[layer][DOC][ROW].mean()
    return S.with_blocks(blocks)


def _without_row(S):
    drop = (lambda docs: [np.delete(x, ROW, axis=0) if i == DOC else x
                          for i, x in enumerate(docs)])
    return inv.States(blocks={li: drop(d) for li, d in S.blocks.items()},
                      final_norm=None if S.final_norm is None else drop(S.final_norm),
                      embedding=None if S.embedding is None else drop(S.embedding))


def test_constant_row_at_first_layer_is_dropped_not_propagated(states):
    """The constant row sits in the first RDM, so before the fix the first-last
    pair and the first adjacent pair were NaN. Now every correlation key is
    finite and equals the run on the corpus with that token deleted, bit for
    bit (every row is selected, so both runs use the same conditions)."""
    cfg = {"max_tokens": 10_000}
    hit = inv.run_task_on_states(TASK, _with_constant_row(states, 0), config=cfg)
    ref = inv.run_task_on_states(TASK, _without_row(states), config=cfg)
    for k in HEADLINE:
        assert math.isfinite(hit[k]), (k, hit[k])
    assert hit["_meta_rsa_n_undefined_pairs_corrdist"] == 0
    assert hit["_meta_n_constant_rows_dropped_corrdist"] == 1
    assert hit["_meta_rsa_corrdist_nan_reason"] is None
    for k in CORR_KEYS:
        assert hit[k] == ref[k], (k, hit[k], ref[k])
    # Euclidean distance is defined on a constant row, so it keeps it.
    assert hit["_meta_n_tokens_used"] == ref["_meta_n_tokens_used"] + 1


def test_no_constant_row_changes_nothing(states):
    base = inv.run_task_on_states(TASK, states)
    assert base["_meta_n_constant_rows_dropped_corrdist"] == 0
    assert all(math.isfinite(base[k]) for k in HEADLINE)
