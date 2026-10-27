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
