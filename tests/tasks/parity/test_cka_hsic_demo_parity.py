"""geometry_cka / geometry_hsic vs Kornblith et al.'s released Demo.ipynb.

The reference functions (gram_linear, center_gram, cka,
_debiased_dot_product_similarity_helper, feature_space_linear_cka) are taken
from google-research representation_similarity/Demo.ipynb with ``ast``. The
feature-space U-statistic ``_debiased_hsic_features`` is checked against the
Demo's helper, and both tasks' ``evaluate()`` run on injected states, biased
and debiased keys alike, against the Demo on the same rows (position 0
dropped; the clouds hold no massive row). Skips without the clone (fails
under BLME_REQUIRE_REFS=1).
"""
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_TESTS = Path(__file__).resolve().parents[2]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import invariance as inv  # noqa: E402
from blme.tasks.geometry.cka import _debiased_hsic_features, _linear_cka_debiased  # noqa: E402

_NB = Path(os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))) \
    / "kornblith_repsim" / "Demo.ipynb"
_FUNCS = ("gram_linear", "center_gram", "cka", "_debiased_dot_product_similarity_helper",
          "feature_space_linear_cka")


def _missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


@pytest.fixture(scope="module")
def demo():
    if not _NB.is_file():
        _missing(_NB)
    nodes = []
    for cell in json.loads(_NB.read_text())["cells"]:
        if cell["cell_type"] == "code":
            body = ast.parse("".join(cell["source"])).body
            nodes += [n for n in body if isinstance(n, ast.FunctionDef) and n.name in _FUNCS]
    assert sorted(n.name for n in nodes) == sorted(_FUNCS)
    ns = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_NB), "exec"), ns)
    return ns


def _pair(seed, n=60, d=7):
    g = np.random.default_rng(seed)
    X = g.standard_normal((n, d)) + 2.0
    Y = X @ g.standard_normal((d, 5)) + 0.7 * g.standard_normal((n, 5)) - 1.0
    return X, Y


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_feature_space_u_statistic_is_the_demo_helper(demo, seed):
    """HSIC_u = helper / (n (n - 3)) on column-centred features."""
    X, Y = _pair(seed)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    n = X.shape[0]
    rx, ry = np.einsum("ij,ij->i", Xc, Xc), np.einsum("ij,ij->i", Yc, Yc)
    want = demo["_debiased_dot_product_similarity_helper"](
        np.linalg.norm(Xc.T @ Yc) ** 2, rx, ry, rx.sum(), ry.sum(), n) / (n * (n - 3))
    got = _debiased_hsic_features(torch.from_numpy(Xc), torch.from_numpy(Yc))
    assert got == pytest.approx(want, rel=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_debiased_cka_is_the_demo_in_feature_and_gram_space(demo, seed):
    X, Y = _pair(seed)
    got = _linear_cka_debiased(torch.from_numpy(X), torch.from_numpy(Y))
    want_f = demo["feature_space_linear_cka"](X, Y, debiased=True)
    want_g = demo["cka"](demo["gram_linear"](X - X.mean(0)), demo["gram_linear"](Y - Y.mean(0)),
                         debiased=True)
    assert want_f == pytest.approx(want_g, rel=1e-9)
    # _linear_cka_debiased centres in float32 before the float64 U-statistic.
    assert got == pytest.approx(want_f, rel=1e-5)


L, N_DOCS, T, D = 5, 6, 14, 8


@pytest.fixture(scope="module")
def walk():
    """Injected block outputs: a random walk across depth, so layer pairs
    differ; Gaussian rows hold no massive activation."""
    g = np.random.default_rng(7)
    docs = [g.standard_normal((T, D)) for _ in range(N_DOCS)]
    blocks = {}
    for li in range(L):
        blocks[li] = [x.astype(np.float32) for x in docs]
        docs = [x @ (np.eye(D) + 0.6 * g.standard_normal((D, D)) / np.sqrt(D))
                + 0.5 * g.standard_normal((T, D)) for x in docs]
    rows = {li: np.concatenate([np.asarray(x, np.float64)[1:] for x in blocks[li]])
            for li in range(L)}
    return blocks, rows


def _summaries(M):
    iu = np.triu_indices(L, k=1)
    return {"early_late": M[0, -1], "min_offdiag": M[iu].min(),
            "min_adjacent": min(M[i, i + 1] for i in range(L - 1))}


def test_cka_evaluate_matches_the_demo(demo, walk):
    blocks, rows = walk
    res = inv.run_task_on_states("geometry_cka", blocks)
    assert res["_meta_n_layers"] == L and res["_meta_sink_n_massive_dropped"] == 0
    for debiased, sfx in ((False, ""), (True, "_debiased")):
        M = np.array([[demo["feature_space_linear_cka"](rows[i], rows[j], debiased=debiased)
                       for j in range(L)] for i in range(L)])
        for name, v in _summaries(M).items():
            assert res[f"{name}_cka{sfx}"] == pytest.approx(v, rel=2e-5), (name, sfx)


def test_hsic_evaluate_matches_the_demo(demo, walk):
    """Normalised linear HSIC is linear CKA (Kornblith Sec. 3): the anchor is
    the first block, compression ratio = CKA(first, last)."""
    blocks, rows = walk
    res = inv.run_task_on_states("geometry_hsic", blocks)
    assert res["_meta_n_tokens_used"] == N_DOCS * (T - 1)
    for debiased, sfx in ((False, ""), (True, "_debiased")):
        def c(i, j):
            return demo["cka"](demo["gram_linear"](rows[i] - rows[i].mean(0)),
                               demo["gram_linear"](rows[j] - rows[j].mean(0)), debiased=debiased)
        assert res[f"hsic_compression_ratio{sfx}"] == pytest.approx(c(0, L - 1), rel=2e-5)
        assert res[f"min_adjacent_hsic{sfx}"] == pytest.approx(
            min(c(i, i + 1) for i in range(L - 1)), rel=2e-5)
