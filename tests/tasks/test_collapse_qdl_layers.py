"""geometry_collapse: qdl_entropy_min is read over every block ("all", the
default) or over blocks 1..L-2 ("interior"), and the other choice is always
recorded. The blind re-implementation found pythia-160m's minimum at the final
pre-norm block (0.266) while its mid-depth valley is 0.80; gpt2's minimum is
mid-depth, so the "all" key mixes the paper's valley with a final-layer effect.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from invariance.driver import run_task_on_states  # noqa: E402


def _states():
    rng = np.random.default_rng(0)
    T, d = 70, 16
    out = {}
    for l in range(5):
        docs = []
        for _ in range(4):
            X = rng.standard_normal((T, d))
            if l == 4:      # a dominant direction at the last block only
                X[:, 0] *= 40.0
            docs.append(X.astype(np.float32))
        out[l] = docs
    return out


def test_interior_excludes_the_end_blocks_and_both_are_recorded():
    st = _states()
    a = run_task_on_states("geometry_collapse", st, {"num_samples": 4})
    b = run_task_on_states("geometry_collapse", st, {"num_samples": 4, "qdl_layers": "interior"})
    prof = np.asarray(a["_meta_qdl_entropy_per_layer"], dtype=float)
    assert a["_meta_qdl_layers"] == "all" and b["_meta_qdl_layers"] == "interior"
    assert a["qdl_entropy_min"] == float(np.nanmin(prof))
    assert b["qdl_entropy_min"] == float(np.nanmin(prof[1:-1]))
    assert b["qdl_entropy_min"] > a["qdl_entropy_min"]          # the last-block dip is left out
    assert a["_meta_qdl_entropy_min_interior"] == b["qdl_entropy_min"]
    assert b["_meta_qdl_entropy_min_all"] == a["qdl_entropy_min"]
