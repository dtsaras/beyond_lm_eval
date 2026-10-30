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


def test_no_real_token_is_dropped_on_a_bos_less_tokenizer(tiny):
    """The reference keeps every token (no special tokens added). The task
    dropped the first row of every document, a real word on GPT-2/Pythia/Qwen."""
    model, tok, cache = tiny
    assert tok("a")["input_ids"][0] != tok.bos_token_id or tok.bos_token_id is None
    res = TrajectoryCurvatureTask({"num_samples": N}).evaluate(model, tok, fd.dataset(), cache)
    assert res["_meta_skip_first_tokens"] == 0 and res["_meta_tokenizer_adds_bos"] is False
    hs = cache.get_hidden_states(layer_idx="all", num_samples=N, per_sample=True,
                                 replace_final_norm=False)
    assert res["curvature_mean_first_layer"] == _pooled_layer_curvature(hs[0], skip_first=0)
    assert res["curvature_mean_first_layer"] != _pooled_layer_curvature(hs[0], skip_first=1)


def test_a_tokenizer_added_bos_is_skipped():
    from blme.tasks.geometry.trajectory_curvature import _tokenizer_adds_bos

    class _BosTok:
        bos_token_id = 1

        def __call__(self, text):
            return {"input_ids": [1, 7]}
    assert _tokenizer_adds_bos(_BosTok()) is True
    assert _tokenizer_adds_bos(fd.tokenizer()) is False
