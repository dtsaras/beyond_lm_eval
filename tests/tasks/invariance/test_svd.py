"""geometry_svd: route-independent schema, token policies, key hygiene and size
invariance, driven through the real ``evaluate()``."""
from __future__ import annotations

import numpy as np
import pytest

from _family_c import TEXTS, inv, load

TASK = "geometry_svd"


@pytest.fixture(scope="module")
def tiny():
    return load("hf-internal-testing/tiny-random-gpt2")


@pytest.fixture(scope="module")
def tiny_states(tiny):
    return inv.states_from_model(*tiny, TEXTS)


def _run_model(model, tok, **cfg):
    from blme.tasks.geometry.isotropy import SVDIsotropyTask
    return SVDIsotropyTask({"num_samples": len(TEXTS), **cfg}).evaluate(
        model, tok, dataset=[{"text": t} for t in TEXTS], cache=None)


def test_layerwise_without_a_cache_collects_every_layer(tiny, tiny_states):
    """layerwise: true with no usable cache used to fall through to the
    single-layer branch, so the per-layer keys vanished on that route. It now
    collects every layer and reproduces the cache route."""
    model, tok = tiny
    r = _run_model(model, tok, layerwise=True, use_cache=False)
    assert "layer_effective_rank" in r
    assert r["_meta_svd_mode"] == "layerwise"
    assert r["_meta_svd_source"] == "self_collected"
    cached = inv.run_task_on_states(TASK, tiny_states, {"layerwise": True})
    assert cached["_meta_svd_mode"] == "layerwise"
    assert cached["_meta_svd_source"] == "shared_cache"
    lw = [k for k in cached if "layer" in k and not k.startswith("_meta_svd")]
    assert lw and set(lw) <= set(r)
    single = _run_model(model, tok, layerwise=False, use_cache=False)
    assert single["_meta_svd_mode"] == "single_layer"
