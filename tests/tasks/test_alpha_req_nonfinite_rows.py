"""geometry_alpha_req: a non-finite row is masked and counted instead of
making the whole spectrum non-finite."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from _family_d import inv  # noqa: E402

TASK = "geometry_alpha_req"
DOC = 3


@pytest.fixture(scope="module")
def states():
    return inv.states_from_model(fd.tiny_gpt2(), fd.tokenizer(), fd.TEXTS)


def _poison_last_token(S, value):
    fn = [np.array(x, dtype=np.float32, copy=True) for x in S.final_norm]
    fn[DOC][-1, 0] = value
    return inv.States(blocks=S.blocks, final_norm=fn, embedding=S.embedding)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_row_is_masked_and_counted(states, value):
    """The read-out cloud is one last-token row per document; one non-finite
    coordinate used to make the centred cloud non-finite (the SVD raised).
    Masked, the result equals the run without that document, bit for bit."""
    hit = inv.run_task_on_states(TASK, _poison_last_token(states, value))
    ref = inv.run_task_on_states(TASK, states.select_docs(
        [i for i in range(states.n_docs) if i != DOC]))
    assert math.isfinite(hit["rankme"])
    assert hit["rankme"] == ref["rankme"]
    assert hit["_meta_n_nonfinite_rows_masked"] == 1
    assert ref["_meta_n_nonfinite_rows_masked"] == 0
    assert hit["_meta_n_eigenvalues"] == ref["_meta_n_eigenvalues"]
    # The rank < 100 reason is kept: 15 points give 15 eigenvalues.
    assert "the fit needs ranks 11-100" in hit["_meta_alpha_req_skipped_reason"]
