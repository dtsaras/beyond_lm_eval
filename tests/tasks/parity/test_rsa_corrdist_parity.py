"""geometry_rsa: the correlation-distance headline keys vs rsatoolbox.

The euclidean twins (now _meta_) are pinned against rsatoolbox in
test_comprehensive_parity; the headline uses correlation-distance RDMs. Here
evaluate() runs on injected states (fewer than the 200-token cap, so every
non-sink row is used) and each key is rsatoolbox's own pipeline on the same
rows: calc_rdm(method="correlation") per layer, compare(method="spearman")
between layers; adjacent minimum and first-vs-last.
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

L, N_DOCS, T, D = 5, 8, 9, 10


def test_corrdist_keys_match_rsatoolbox():
    g = np.random.default_rng(4)
    docs = [g.standard_normal((T, D)) for _ in range(N_DOCS)]
    blocks = {}
    for li in range(L):
        blocks[li] = [x.astype(np.float32) for x in docs]
        W = np.eye(D) + 0.5 * g.standard_normal((D, D)) / np.sqrt(D)
        docs = [np.tanh(x @ W) + 0.4 * g.standard_normal((T, D)) for x in docs]
    res = inv.run_task_on_states("geometry_rsa", blocks)
    rows = {li: np.concatenate([np.asarray(x, np.float64)[1:] for x in blocks[li]])
            for li in range(L)}
    assert rows[0].shape[0] < 200
    rdms = {li: calc_rdm(Dataset(X), method="correlation") for li, X in rows.items()}

    def rsa(i, j):
        return float(compare(rdms[i], rdms[j], method="spearman")[0, 0])

    adjacent = [rsa(i, i + 1) for i in range(L - 1)]
    assert np.ptp(adjacent) > 1e-3
    assert res["rsa_adjacent_min_corrdist"] == pytest.approx(min(adjacent), rel=1e-6)
    assert res["rsa_early_late_corrdist"] == pytest.approx(rsa(0, L - 1), rel=1e-6)
