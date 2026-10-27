"""geometry_categories: vocabulary-size independence (draws and searches over
the rows the tokenizer can emit) and key hygiene, on real gpt2 weights."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from _family_c import inv, load

TASK = "geometry_categories"


@pytest.fixture(scope="module")
def gpt2():
    return load("gpt2")


def _run(model, tok):
    from blme.core import _register_all_tasks
    from blme.registry import get_task
    _register_all_tasks()
    return get_task(TASK)(config={}).evaluate(model, tok, dataset=None)


def _headline(r):
    return sorted(k for k, v in r.items() if not k.startswith("_meta_")
                  and isinstance(v, (int, float)) and not isinstance(v, bool))


@pytest.fixture(scope="module")
def base(gpt2):
    return _run(*gpt2)


@pytest.fixture(scope="module")
def zero_padded(gpt2):
    """64 unused zero rows appended (gpt2's tokenizer still emits 50257 ids)."""
    model, tok = gpt2
    return _run(inv.pad_vocabulary(model, 64, kind="zero"), tok)


def test_comparison_draw_does_not_depend_on_the_vocabulary_size(base, zero_padded):
    """Appending unused rows used to redraw the 10 comparison tokens (28 of 87
    keys moved, up to 17.6% on gpt2): the draw is now over emittable ids with a
    local generator, so no headline key moves."""
    assert zero_padded["_meta_n_unused_rows"] == 64
    assert base["_meta_n_unused_rows"] == 0
    keys = _headline(base)
    assert any(k.endswith("_inter") for k in keys)
    inv.assert_invariant(base, zero_padded, keys, rtol=1e-6)


def test_unused_rows_never_enter_a_neighbour_search(gpt2, base):
    """Unused rows (ids the tokenizer cannot emit) holding copies of category
    vectors would be every category token's nearest neighbours; the searches
    run over emittable rows, so purity, hubness and consistency do not move."""
    from blme.tasks.geometry.categories import CategoryGeometryTask
    model, tok = gpt2
    import json, os, blme
    cats = json.load(open(os.path.join(os.path.dirname(blme.__file__), "assets",
                                       "categories.json")))
    cat_tokens, _ = CategoryGeometryTask()._get_category_tokens(tok, cats)
    ids = sorted({t for v in cat_tokens.values() for t in v})[:128]
    padded = inv.pad_vocabulary(model, len(ids), kind="zero")
    W = padded.get_input_embeddings().weight
    V = model.get_input_embeddings().weight.shape[0]
    with torch.no_grad():
        W[V:] = W[torch.tensor(ids)]
    r = _run(padded, tok)
    keys = [k for k in _headline(base) if k.endswith(("_purity", "_hub_skew", "_hub_max",
                                                      "_consistency"))]
    assert keys
    inv.assert_invariant(base, r, keys, rtol=1e-6)



def test_token_count_is_not_a_headline(base):
    """n_category_tokens_mapped counts single-token category words (704 gpt2,
    711 pythia): a tokenizer property, emitted as _meta_ only."""
    assert "n_category_tokens_mapped" not in base
    assert base["_meta_n_category_tokens_mapped"] > 0
