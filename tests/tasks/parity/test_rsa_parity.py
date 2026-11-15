"""geometry_rsa vs rsatoolbox on the 200-row subsample evaluate() takes.

evaluate() runs on injected states holding more non-sink rows than the
``max_tokens: 200`` default, so the evenly spaced selection is the one that
ships. Every key is rsatoolbox's own pipeline on those rows:

    rsatoolbox.rdm.calc_rdm(Dataset(X), method=...)   one RDM per layer
    rsatoolbox.rdm.compare(a, b, method="spearman")   between layers

Headline (Kriegeskorte et al. 2008 correlation distance): method="correlation".
``_meta_`` Euclidean twins: method="euclidean", which rsatoolbox defines as
squared distance / n_features; Spearman is unchanged by that monotone map.

test_rsa_corrdist_parity.py pins the headline below the cap (every row used).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[2]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import invariance as inv  # noqa: E402

rsatoolbox = pytest.importorskip("rsatoolbox")
from rsatoolbox.data import Dataset  # noqa: E402
from rsatoolbox.rdm import calc_rdm, compare  # noqa: E402

TASK = "geometry_rsa"
L, N_DOCS, T, D = 5, 30, 12, 10
CAP = 200


@pytest.fixture(scope="module")
def blocks():
    g = np.random.default_rng(20240624)
    docs = [g.standard_normal((T, D)) for _ in range(N_DOCS)]
    out = {}
    for li in range(L):
        out[li] = [x.astype(np.float32) for x in docs]
        W = np.eye(D) + 0.5 * g.standard_normal((D, D)) / np.sqrt(D)
        docs = [np.tanh(x @ W) + 0.4 * g.standard_normal((T, D)) for x in docs]
    return out


@pytest.fixture(scope="module")
def result(blocks):
    return inv.run_task_on_states(TASK, blocks)


@pytest.fixture(scope="module")
def rows(blocks, result):
    """The rows evaluate() scores: position 0 dropped, then 200 evenly spaced."""
    pool = {li: np.concatenate([np.asarray(x, np.float64)[1:] for x in blocks[li]])
            for li in range(L)}
    n = pool[0].shape[0]
    assert n == N_DOCS * (T - 1) > CAP
    assert result["_meta_sink_n_massive_dropped"] == 0
    assert result["_meta_n_tokens_used"] == CAP
    assert result["_meta_n_constant_rows_dropped_corrdist"] == 0
    sel = np.linspace(0, n - 1, CAP).astype(int)
    return {li: X[sel] for li, X in pool.items()}


def _reference(rows, method):
    rdms = {li: calc_rdm(Dataset(X), method=method) for li, X in rows.items()}

    def rsa(i, j):
        return float(compare(rdms[i], rdms[j], method="spearman")[0, 0])

    adjacent = [rsa(i, i + 1) for i in range(L - 1)]
    assert np.ptp(adjacent) > 1e-3
    return {"adjacent": adjacent, "early_late": rsa(0, L - 1),
            "first_middle": rsa(0, L // 2)}


def test_headline_corrdist_keys_match_rsatoolbox_on_the_subsample(result, rows):
    ref = _reference(rows, "correlation")
    assert result["rsa_adjacent_min_corrdist"] == pytest.approx(min(ref["adjacent"]), rel=1e-6)
    assert result["rsa_early_late_corrdist"] == pytest.approx(ref["early_late"], rel=1e-6)
    assert result["_meta_rsa_adjacent_mean_corrdist"] == pytest.approx(
        float(np.mean(ref["adjacent"])), rel=1e-6)
    assert result["_meta_rsa_first_middle_corrdist"] == pytest.approx(
        ref["first_middle"], rel=1e-6)
    assert result["_meta_rsa_min_continuity_layer_corrdist"] == int(np.argmin(ref["adjacent"]))


def test_headline_is_not_the_leading_slice_nor_the_euclidean_twin(blocks, result, rows):
    """The two protocols the headline replaced give other numbers on this
    fixture, so the test above cannot pass on either."""
    lead = {li: np.concatenate([np.asarray(x, np.float64)[1:] for x in blocks[li]])[:CAP]
            for li in range(L)}
    for other in (_reference(lead, "correlation"), _reference(rows, "euclidean")):
        assert abs(min(other["adjacent"]) - result["rsa_adjacent_min_corrdist"]) > 1e-4
        assert abs(other["early_late"] - result["rsa_early_late_corrdist"]) > 1e-4


def test_euclidean_meta_twins_match_rsatoolbox_on_the_subsample(result, rows):
    ref = _reference(rows, "euclidean")
    assert result["_meta_rsa_adjacent_min_euclidean"] == pytest.approx(
        min(ref["adjacent"]), rel=1e-6)
    assert result["_meta_rsa_adjacent_mean_euclidean"] == pytest.approx(
        float(np.mean(ref["adjacent"])), rel=1e-6)
    assert result["_meta_rsa_early_late_euclidean"] == pytest.approx(ref["early_late"], rel=1e-6)
    assert result["_meta_rsa_first_middle_euclidean"] == pytest.approx(
        ref["first_middle"], rel=1e-6)
