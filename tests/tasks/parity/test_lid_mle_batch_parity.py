"""geometry_lid vs Ma et al.'s released ``mle_batch``, through evaluate().

``mle_batch`` is taken with ``ast`` from
lid_adversarial_subspace_detection/util.py (the module imports Keras). The
task runs in the study's layerwise mode on injected states: one point per
sequence (its last token) per block, every point a query when N <= 500, and
k = 20 as packaged. The six headline keys are then plain summaries of the
reference estimates: mean / std / max / median at the deepest block, and the
mean at the first and middle blocks. mle_batch works in float32, hence
rel 1e-5. The default single-layer mode reads HF's post-norm last entry and
emits the four deepest-layer keys from the same estimator. Skips without the
clone (fails under BLME_REQUIRE_REFS=1).
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist

_TESTS = Path(__file__).resolve().parents[2]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import invariance as inv  # noqa: E402
from blme.tasks.geometry import lid as LID  # noqa: E402

_UTIL = Path(os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))) \
    / "lid_adversarial_subspace_detection" / "util.py"
N_DOCS, D, L = 80, 6, 5


def _missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


@pytest.fixture(scope="module")
def mle_batch():
    if not _UTIL.is_file():
        _missing(_UTIL)
    body = ast.parse(_UTIL.read_text()).body
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name == "mle_batch"]
    assert len(nodes) == 1
    ns = {"np": np, "cdist": cdist}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_UTIL), "exec"), ns)
    return ns["mle_batch"]


@pytest.fixture(scope="module")
def clouds():
    """Per-block last-token clouds on a curved low-dimensional manifold, so the
    LID differs across blocks; float32 as the cache stores them."""
    g = np.random.default_rng(5)
    t = g.uniform(-1.0, 1.0, (N_DOCS, 2))
    out = []
    for li in range(L):
        W = g.standard_normal((4, D))
        F = np.concatenate([t, np.sin((li + 1) * t)], axis=1) @ W
        out.append((F + (0.02 + 0.05 * li) * g.standard_normal((N_DOCS, D))).astype(np.float32))
    return out


def test_layerwise_lid_matches_mle_batch(mle_batch, clouds):
    g = np.random.default_rng(6)
    blocks = {li: [np.vstack([g.standard_normal((3, D)), X[i:i + 1]]) for i in range(N_DOCS)]
              for li, X in enumerate(clouds)}
    res = inv.run_task_on_states("geometry_lid", blocks, {"layerwise": True})
    assert res["_meta_lid_mode"] == "layerwise"
    k = 20
    est = [mle_batch(X, X, k) for X in clouds]
    deepest = est[-1]
    want = {"lid_mean": deepest.mean(), "lid_std": deepest.std(), "lid_max": deepest.max(),
            "lid_median": np.median(deepest), "lid_first_layer": est[0].mean(),
            "lid_mid_layer": est[L // 2].mean()}
    assert np.ptp([e.mean() for e in est]) > 0.1
    for key, v in want.items():
        assert res[key] == pytest.approx(v, rel=1e-5), key


def test_sources_pin_the_cloned_commit():
    repo = _UTIL.parent
    if not repo.is_dir():
        _missing(repo)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert LID.LocalIntrinsicDimensionalityTask.SOURCES["code"].endswith("@" + head)


def test_single_layer_lid_matches_mle_batch(mle_batch, clouds):
    """The packaged default (layerwise: false): one last-token point per
    sequence from the final (post-norm) entry, every point a query."""
    g = np.random.default_rng(7)
    final = [np.vstack([g.standard_normal((2, D)), clouds[-1][i:i + 1]]).astype(np.float32)
             for i in range(N_DOCS)]
    blocks = {li: [g.standard_normal((3, D)).astype(np.float32) for _ in range(N_DOCS)]
              for li in range(L)}
    res = inv.run_task_on_states("geometry_lid", blocks, final_norm=final)
    assert res["_meta_lid_mode"] == "single_layer"
    est = mle_batch(clouds[-1], clouds[-1], 20)
    want = {"lid_mean": est.mean(), "lid_std": est.std(), "lid_max": est.max(),
            "lid_median": np.median(est)}
    for key, v in want.items():
        assert res[key] == pytest.approx(v, rel=1e-5), key
