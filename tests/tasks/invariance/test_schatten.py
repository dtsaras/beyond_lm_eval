"""geometry_schatten: Yusupov et al.'s Schatten norm (MLP-internal activations,
averaged over layers, full text) with the residual-stream version as _meta_."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from _family_c import TEXTS, inv, load

TASK = "geometry_schatten"


@pytest.fixture(scope="module")
def tiny():
    return load("hf-internal-testing/tiny-random-gpt2")


def _run(model, tok, texts, **cfg):
    from blme.tasks.geometry.schatten import SchattenNormTask
    return SchattenNormTask({"num_samples": len(texts), **cfg}).evaluate(
        model, tok, dataset=[{"text": t} for t in texts], cache=None)


def _yusupov_reference(model, tok, texts, p):
    """Written from Sec. 4.1 / Eq. 5: per text, the Schatten-p norm of the raw
    (T, d_ff) activation entering each block's MLP down-projection, averaged
    over the L layers; then averaged over texts."""
    acts = {}
    hooks = [blk.mlp.c_proj.register_forward_pre_hook(
        lambda m, a, i=i: acts.__setitem__(i, a[0].detach()))
        for i, blk in enumerate(model.transformer.h)]
    per_text = []
    try:
        with torch.no_grad():
            for t in texts:
                acts.clear()
                model(**tok(t, return_tensors="pt"))
                vals = []
                for i in sorted(acts):
                    s = np.linalg.svd(acts[i][0].double().numpy(), compute_uv=False)
                    vals.append(float(np.sum(s ** p) ** (1.0 / p)))
                per_text.append(np.mean(vals))
    finally:
        for h in hooks:
            h.remove()
    return float(np.mean(per_text))


def test_schatten_is_yusupov_mlp_layer_mean(tiny):
    """schatten_1 / schatten_4 are Yusupov et al.'s quantity (MLP-internal
    activations, mean over layers); the residual last-layer values that used
    to carry those names are _meta_."""
    model, tok = tiny
    r = _run(model, tok, TEXTS)
    for p in (1, 4):
        assert r[f"schatten_{p}"] == pytest.approx(
            _yusupov_reference(model, tok, TEXTS, p), rel=1e-6)
    assert r["_meta_schatten_mlp_layers_found"] == len(model.transformer.h)
    assert "schatten_1_last" not in r and "_meta_schatten_1_last" in r
