"""geometry_neural_collapse: Wu & Papyan's next-token protocol (token
features, next-token classes, CDNV), on a real model."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from _family_c import TEXTS, inv, load

TASK = "geometry_neural_collapse"


@pytest.fixture(scope="module")
def tiny():
    return load("hf-internal-testing/tiny-random-gpt2")


def _run(model, tok, **cfg):
    from blme.core import _register_all_tasks
    from blme.registry import get_task
    _register_all_tasks()
    config = {"num_samples": len(TEXTS), "min_per_class": 2, **cfg}
    with inv.limit_threads(1):
        return get_task(TASK)(config=config).evaluate(
            model, tok, dataset=[{"text": t} for t in TEXTS])


def _reference_cdnv(model, tok, min_per_class, dist_exp):
    """Independent recomputation from the definitions: features = post-norm
    last hidden state at t < T-1, label = token t+1 (lib/collection.py);
    var_c = mean ||x - mu_c||^2 (VarNormAccumulator); mean over c < c' of
    ((var_c + var_c')/2) / (||mu_c - mu_c'||^2)^dist_exp (kernels.py)."""
    X, Y = [], []
    with torch.no_grad():
        for t in TEXTS:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=512)
            h = model(**ids, output_hidden_states=True).hidden_states[-1][0]
            X.append(h[:-1].double().numpy())
            Y.extend(ids["input_ids"][0, 1:].tolist())
    X = np.concatenate(X)
    Y = np.asarray(Y)
    classes = [c for c in np.unique(Y) if (Y == c).sum() >= min_per_class]
    mu = np.stack([X[Y == c].mean(0) for c in classes])
    var = np.array([((X[Y == c] - m) ** 2).sum(1).mean() for c, m in zip(classes, mu)])
    vals = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            d2 = ((mu[i] - mu[j]) ** 2).sum()
            vals.append(0.5 * (var[i] + var[j]) / d2 ** dist_exp)
    return float(np.mean(vals)), len(classes)


def test_next_token_classes_and_cdnv(tiny):
    """Wu & Papyan 2024: NC1 as CDNV over next-token classes of token-level
    post-norm features (Eq. 4, dist_exp 2), with Galanti's CDNV beside it."""
    model, tok = tiny
    r = _run(model, tok)
    want, n_cls = _reference_cdnv(model, tok, 2, 2.0)
    want_g, _ = _reference_cdnv(model, tok, 2, 1.0)
    assert r["_meta_nc_n_classes"] == n_cls
    assert r["nc1_cdnv"] == pytest.approx(want, rel=1e-6)
    assert r["nc1_cdnv_galanti"] == pytest.approx(want_g, rel=1e-6)
    # The topic proxy survives as _meta_ only.
    assert "nc1_within_class_collapse" not in r
    assert np.isfinite(r["_meta_topic_nc1_within_class_collapse"])


def test_galanti_twin_is_free_of_the_feature_scale(tiny):
    """Function-preserving final-norm gauge (gamma x2, W_U / 2): every logit is
    unchanged and every feature doubles. Galanti's CDNV is a ratio of squared
    lengths and stays put; the paper's Eq. 4 divides by one more squared
    distance and moves by exactly 1/4 (known negative)."""
    model, tok = tiny
    base = _run(model, tok)
    gauged = _run(inv.final_norm_unembedding_gauge(model, D=2.0), tok)
    inv.assert_invariant(base, gauged, ["nc1_cdnv_galanti"], rtol=1e-12)
    assert gauged["nc1_cdnv"] == pytest.approx(base["nc1_cdnv"] / 4.0, rel=1e-9)


def test_declarations():
    from blme.tasks.geometry.neural_collapse import NeuralCollapseTask as N
    assert N.TOKEN_POLICY.startswith("paper_protocol:")
    assert set(N.SIZE_INVARIANT) <= set(N.HEADLINE_KEYS)
    assert {"paper", "code", "llm_precedent"} <= set(N.SOURCES)
    assert N.ADAPTATION_NOTE.count(". ") <= 8
