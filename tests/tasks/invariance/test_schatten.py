"""geometry_schatten: Yusupov et al.'s Schatten norm (MLP-internal activations,
averaged over layers, the first 256 tokens of each text) with the
residual-stream version as _meta_."""
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
    cfg = {"num_samples": len(texts), "documents": "dataset", **cfg}
    return SchattenNormTask(cfg).evaluate(
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
    """_meta_schatten_{1,4}_mlp are Yusupov et al.'s quantity (MLP-internal
    activations, mean over layers); the residual last-layer values are the
    _meta_schatten_{1,4}_last keys."""
    model, tok = tiny
    r = _run(model, tok, TEXTS)
    for p in (1, 4):
        assert r[f"_meta_schatten_{p}_mlp"] == pytest.approx(
            _yusupov_reference(model, tok, TEXTS, p), rel=1e-6)
    assert r["_meta_schatten_mlp_layers_found"] == len(model.transformer.h)
    assert "schatten_1_last" not in r and "_meta_schatten_1_last" in r


def test_raw_mlp_norms_are_meta_only(tiny):
    """The raw MLP-activation norms are in the activation's units and grow
    with d_ff: they are emitted as _meta_ only, and the headline is the two
    RankMe keys and the two row-normalised keys."""
    from blme.tasks.geometry.schatten import SchattenNormTask as cls
    r = _run(*tiny, TEXTS[:3])
    assert cls.HEADLINE_KEYS == ("rankme_last", "row_normalized_rankme_last",
                                 "row_normalized_matrix_nuclear_norm_last",
                                 "row_normalized_schatten_inf_last")
    for p in (1, 4):
        assert f"schatten_{p}" not in r
        assert np.isfinite(r[f"_meta_schatten_{p}_mlp"]) and r[f"_meta_schatten_{p}_mlp"] > 0
    scalars = sorted(k for k, v in r.items() if not k.startswith("_meta_")
                     and isinstance(v, float))
    assert scalars == sorted(cls.HEADLINE_KEYS)


def test_a_long_text_is_read_to_the_window(tiny):
    """A ~330-token document is read to its first 256 tokens, exactly the
    256-token prefix scored on its own; seq_tokens moves the window."""
    model, tok = tiny
    long_text = " ".join(TEXTS)
    ids = tok(long_text, return_tensors="pt")["input_ids"][0]
    assert ids.shape[-1] > 256
    win = _run(model, tok, [long_text])
    prefix = _run(model, tok, [tok.decode(ids[:256])])
    assert win["_meta_schatten_seq_tokens"] == 256
    assert win["_meta_schatten_n_docs_truncated"] == 1
    assert prefix["_meta_schatten_n_docs_truncated"] == 0
    from blme.tasks.geometry.schatten import SchattenNormTask
    for k in SchattenNormTask.HEADLINE_KEYS:
        assert win[k] == prefix[k], k
    cut = _run(model, tok, [long_text], seq_tokens=128)
    assert win["row_normalized_matrix_nuclear_norm_last"] != cut["row_normalized_matrix_nuclear_norm_last"]


def _article(i, tail_seed):
    rng = np.random.default_rng(1000 * tail_seed + i)
    head = " ".join(TEXTS)                         # > 256 tokens, shared
    tail = " ".join(rng.choice(head.split(), size=200))
    return f"\n = Article {i} = \n\n {head} {tail} \n"


def test_text_past_the_window_is_not_read(tiny, monkeypatch):
    """By default every model scores the first 256 tokens of the long WikiText
    articles geometry_positional_decay reads: rewriting each article past its
    first 256 tokens leaves every headline bit-identical."""
    datasets = pytest.importorskip("datasets")
    from blme.tasks.geometry.schatten import SchattenNormTask
    model, tok = tiny
    outs = []
    for tail_seed in (1, 2):
        pages = [{"page": _article(i, tail_seed)} for i in range(4)]
        monkeypatch.setattr(datasets, "load_dataset", lambda *a, _p=pages, **k: _p)
        data = [{"text": p["page"].strip()} for p in pages[:3]]
        outs.append(SchattenNormTask({"num_samples": 3}).evaluate(model, tok, data, cache=None))
    a, b = outs
    for k in SchattenNormTask.HEADLINE_KEYS:
        assert a[k] == b[k], (k, a[k], b[k])
    assert a["_meta_schatten_seq_tokens"] == 256 and a["_meta_schatten_documents"] == "long"
    assert a["_meta_schatten_n_docs_truncated"] == 3



def test_declarations(tiny):
    from blme.tasks.geometry.schatten import SchattenNormTask as cls
    assert cls.TOKEN_POLICY.startswith("paper_protocol:")
    assert {"paper", "code", "llm_precedent"} <= set(cls.SOURCES)
    assert cls.SIZE_INVARIANT == {}
    r = _run(*tiny, TEXTS[:3])
    assert set(cls.HEADLINE_KEYS) <= set(r)
