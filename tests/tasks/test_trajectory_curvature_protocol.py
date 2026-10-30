"""geometry_trajectory_curvature follows Hosseini & Fedorenko 2023 /
eghbalhosseini/sent_sampling@32ac271 (curvature_utils.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))
import _family_d as fd  # noqa: E402

from blme.cache import ModelOutputCache  # noqa: E402
from blme.tasks.geometry.trajectory_curvature import (  # noqa: E402
    TrajectoryCurvatureTask, _pooled_layer_curvature)

N = len(fd.TEXTS)


@pytest.fixture(scope="module")
def tiny():
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    cache = ModelOutputCache(model, tok, dataset=fd.dataset(), num_samples=N)
    cache.populate(need_hidden=True)
    return model, tok, cache


def test_last_layer_is_hfs_post_norm_state(tiny):
    """compute_model_activations keeps outputs['hidden_states'] as returned
    (curvature_utils.py:22-23): the last entry is ln_f(block L-1). BLME read
    the raw last block instead."""
    model, tok, cache = tiny
    res = TrajectoryCurvatureTask({"num_samples": N}).evaluate(model, tok, fd.dataset(), cache)
    post = cache.get_hidden_states(layer_idx="all", num_samples=N, per_sample=True,
                                   replace_final_norm=False)
    raw = cache.get_hidden_states(layer_idx="all", num_samples=N, per_sample=True,
                                  replace_final_norm=True)
    skip = res.get("_meta_skip_first_tokens", 1)
    want = _pooled_layer_curvature(post[max(post)], skip_first=skip)
    assert res["curvature_mean_last_layer"] == want
    assert want != _pooled_layer_curvature(raw[max(raw)], skip_first=skip)
    assert res["_meta_last_entry_is_block_output"] is False
