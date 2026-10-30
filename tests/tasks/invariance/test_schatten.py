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


def test_full_text_not_a_128_token_window(tiny):
    """MNN.py scores every token of the text: a ~330-token document is read in
    full (up to the model's context), not cut at 128."""
    model, tok = tiny
    long_text = " ".join(TEXTS)
    assert tok(long_text, return_tensors="pt")["input_ids"].shape[-1] > 128
    full = _run(model, tok, [long_text])
    cut = _run(model, tok, [long_text], max_length=128)
    assert full["_meta_schatten_n_docs_truncated"] == 0
    assert cut["_meta_schatten_n_docs_truncated"] == 1
    assert full["row_normalized_matrix_nuclear_norm_last"] != cut["row_normalized_matrix_nuclear_norm_last"]
