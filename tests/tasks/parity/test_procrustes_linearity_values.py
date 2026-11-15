"""geometry_procrustes_linearity: every headline key through evaluate().

The reference pair score is LLM-Microscope's released ``procrustes_similarity``
(LLM_microscope.ipynb, taken with ``ast``), computed in float64 as the task
does. evaluate() runs on injected states (embedding + 5 blocks, every row kept,
as the paper's protocol), and the 17 keys are the profile summaries of the
reference scores: mean / min / max / first / last, the OLS slope on depth
l / (n_pairs - 1), the linear-interpolation quartiles, and the adjusted
twins 1 - (1 - sim)(N - 1)/(N - 1 - r) (Wherry 1931) with r = d, the rank
of every centred cloud here. Skips without the clone (fails under
BLME_REQUIRE_REFS=1).
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

_NB = Path(os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))) \
    / "LLM-Microscope" / "LLM_microscope.ipynb"
N_DOCS, T, D, L = 6, 10, 6, 5


def _missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


@pytest.fixture(scope="module")
def release():
    if not _NB.is_file():
        _missing(_NB)
    want = ("get_est_svd", "procrustes_similarity")
    nodes = []
    for cell in json.loads(_NB.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        try:
            body = ast.parse("".join(cell["source"])).body
        except SyntaxError:
            continue
        nodes += [n for n in body if isinstance(n, ast.FunctionDef) and n.name in want]
    assert sorted(n.name for n in nodes) == sorted(want)
    ns = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_NB), "exec"), ns)
    return lambda X, Y: ns["procrustes_similarity"](torch.from_numpy(X), torch.from_numpy(Y))


@pytest.fixture(scope="module")
def walk():
    """A nonlinear residual walk, so the pair scores differ and sit below 1;
    float32 values, as the cache stores them."""
    g = np.random.default_rng(3)
    docs = [g.standard_normal((T, D)) for _ in range(N_DOCS)]
    layers = []
    for _ in range(L + 1):
        layers.append([x.astype(np.float32) for x in docs])
        W = g.standard_normal((D, D)) / np.sqrt(D)
        docs = [x + np.tanh(1.5 * x @ W) + 0.3 * g.standard_normal((T, D)) for x in docs]
    return inv.States(blocks={li: layers[li + 1] for li in range(L)}, embedding=layers[0]), layers


def _summary(s, prefix, slope=True):
    depth = np.arange(s.size) / (s.size - 1)
    out = {"mean": s.mean(), "min": s.min(), "max": s.max(), "first": s[0], "last": s[-1],
           "q25": np.percentile(s, 25), "q50": np.percentile(s, 50), "q75": np.percentile(s, 75)}
    if slope:
        out["slope"] = np.polyfit(depth, s, 1)[0]
    return {f"{prefix}_{k}": v for k, v in out.items()}


def test_every_headline_key_matches_the_release(release, walk):
    states, layers = walk
    res = inv.run_task_on_states("geometry_procrustes_linearity", states)
    assert res["_meta_n_pairs"] == L and res["_meta_n_tokens"] == N_DOCS * T
    clouds = [np.concatenate(docs).astype(np.float64) for docs in layers]
    n = clouds[0].shape[0]
    sims = np.array([release(a, b) for a, b in zip(clouds[:-1], clouds[1:])])
    assert np.ptp(sims) > 1e-2 and sims.max() < 1.0
    adj = 1.0 - (1.0 - sims) * (n - 1) / (n - 1 - D)
    want = {**_summary(sims, "procrustes_linearity"),
            **_summary(adj, "procrustes_linearity_adjusted", slope=False)}
    assert len(want) == 17
    for key, v in want.items():
        assert res[key] == pytest.approx(v, rel=1e-9, abs=1e-12), key
    assert "procrustes_linearity_adjusted_slope" not in res
