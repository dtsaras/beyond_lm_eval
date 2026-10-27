"""geometry_cknna: the platonic-rep feature protocol (per-text masked mean,
joint outlier clamp, embedding layer, 1024 texts), driven through the real
``evaluate()``."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from _family_c import TEXTS, inv, load

TASK = "geometry_cknna"


@pytest.fixture(scope="module")
def tiny():
    return load("hf-internal-testing/tiny-random-gpt2")


@pytest.fixture(scope="module")
def tiny_states(tiny):
    return inv.states_from_model(*tiny, TEXTS)


def _doc_means(docs):
    return torch.stack([torch.as_tensor(np.asarray(d, dtype=np.float64)).mean(0)
                        for d in docs]).float()


def test_points_are_per_text_means(tiny_states):
    """Huh et al. score one vector per text (extract_features.py pool='avg'):
    early_late_cknna equals the reference pipeline on per-document means of
    the first and last HF entries (the last is post-norm, as in the tuple)."""
    from blme.tasks.geometry.cknna import _cknna, _l2_normalize, _remove_outliers
    r = inv.run_task_on_states(TASK, tiny_states, {"topk": 5})
    S = tiny_states
    first = _doc_means(S.blocks[0])
    last = _doc_means(S.final_norm)
    prep = lambda F: _l2_normalize(_remove_outliers(F))   # noqa: E731
    want = _cknna(prep(first), prep(last), topk=5, unbiased=True)
    assert r["_meta_n_points"] == S.n_docs
    assert r["_meta_pooling"] == "mean"
    assert r["early_late_cknna"] == pytest.approx(want, rel=1e-6)


def test_pooled_forward_matches_the_cache_route(tiny):
    """The self-collected route pools per text on the fly and reproduces the
    shared-cache route bit for bit."""
    model, tok = tiny
    a = inv.run_task_on_model(TASK, model, tok, TEXTS, {"topk": 5}, use_cache=True)
    b = inv.run_task_on_model(TASK, model, tok, TEXTS, {"topk": 5}, use_cache=False)
    assert a["_meta_features_source"] == "shared_cache"
    assert b["_meta_features_source"] == "self_collected"
    keys = [k for k in a if isinstance(a[k], float)]
    inv.assert_invariant(a, b, keys, rtol=0.0)
