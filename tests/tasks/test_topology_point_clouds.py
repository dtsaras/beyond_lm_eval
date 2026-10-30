"""The Vietoris-Rips / kNN topology tasks read one point per sequence (Q1).

Fay et al. 2025 (arXiv:2505.20435 Sec. 2.2) and Gardinazzi et al. 2025 read
the last token of each prompt; the tasks mean-pooled every token (position 0
included) of 20-50 documents. N = 1000 sequences for every model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))
import _family_d as fd  # noqa: E402

pytest.importorskip("ripser")

from blme.cache import ModelOutputCache  # noqa: E402
from blme.tasks._tokens import select_rows  # noqa: E402
from blme.tasks.topology._points import last_token_points, role_layers  # noqa: E402

TASKS = {
    "topology_homology": ("blme.tasks.topology.homology", "PersistentHomologyTask"),
    "topology_persistence_entropy": ("blme.tasks.topology.persistence_entropy",
                                     "PersistenceEntropyTask"),
    "topology_persistence_landscape": ("blme.tasks.topology.persistence_landscape",
                                       "PersistenceLandscapeTask"),
    "topology_betti_curve": ("blme.tasks.topology.betti_curve", "BettiCurveTask"),
}


def _cls(name):
    import importlib
    mod, cls = TASKS[name]
    return getattr(importlib.import_module(mod), cls)


@pytest.fixture(scope="module")
def tiny():
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    cache = ModelOutputCache(model, tok, dataset=fd.dataset(), num_samples=len(fd.TEXTS))
    cache.populate(need_hidden=True)
    return model, tok, cache


def test_points_are_the_last_row_of_each_cached_document(tiny):
    model, tok, cache = tiny
    n = len(fd.TEXTS)
    pts, info = last_token_points(model, tok, fd.dataset(), cache, num_samples=n,
                                  layers=role_layers)
    states = cache.get_hidden_states(layer_idx="all", num_samples=n, per_sample=True,
                                     replace_final_norm=True)
    for l, X in pts.items():
        want, meta = select_rows(states[l], "last")
        np.testing.assert_array_equal(X, want.double().numpy())
        assert X.shape[0] == n
    assert info["source"] == "shared_cache" and info["last_is_block"] is True
    lengths = [c.shape[0] for c in states[0]]
    assert info["positions"] == [T - 1 for T in lengths]


def test_private_route_reproduces_the_cache_route(tiny):
    """With fewer cached documents than requested the task collects its own
    last-token rows; both routes must give the same points."""
    model, tok, cache = tiny
    n = len(fd.TEXTS)
    a, ia = last_token_points(model, tok, fd.dataset(), cache, num_samples=n, layers=role_layers)
    b, ib = last_token_points(model, tok, fd.dataset(), None, num_samples=n, layers=role_layers)
    assert ia["source"] == "shared_cache" and ib["source"] == "self_collected"
    for l in a:
        np.testing.assert_array_equal(a[l], b[l])
    assert ia["positions"] == ib["positions"] and ib["last_is_block"] is True


@pytest.mark.parametrize("task", sorted(TASKS))
def test_task_scores_one_point_per_sequence(task, tiny):
    model, tok, cache = tiny
    cls = _cls(task)
    assert cls.TOKEN_POLICY == "last_per_sequence"
    res = cls({"num_samples": len(fd.TEXTS)}).evaluate(model, tok, fd.dataset(), cache=cache)
    assert "error" not in res, res
    assert res["_meta_n_points"] == len(fd.TEXTS)
    assert res["_meta_rows_policy"] == "last"
    assert res["_meta_points_source"] == "shared_cache"
    own = cls({"num_samples": len(fd.TEXTS)}).evaluate(model, tok, fd.dataset(), cache=None)
    assert own["_meta_points_source"] == "self_collected"
    for k, v in res.items():
        if not k.startswith("_meta_points"):
            assert own[k] == v or (isinstance(v, float) and np.isnan(v) and np.isnan(own[k])), k


@pytest.mark.parametrize("task", sorted(TASKS))
def test_default_is_1000_sequences(task):
    from blme.tasks.config_loader import resolve_task_config
    assert resolve_task_config(task)["num_samples"] == 1000


def _two_far_clusters():
    """Two unit-scale clusters 1e6 apart. Centring cannot remove that spread,
    and ripser's point-cloud path (sklearn's ||x||^2 + ||y||^2 - 2 x.y in
    float64) then loses ~1e-4 of every within-cluster distance."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from adversarial_fixtures import on_grid_translation
    rng = np.random.default_rng(0)
    A = rng.standard_normal((40, 3))
    B0, B, ok = on_grid_translation(rng.standard_normal((40, 3)), 1e6, dtype=np.float64)
    assert ok
    return np.vstack([A, B]), A, B0


def _h1_lifetimes(X):
    from ripser import ripser
    from scipy.spatial.distance import pdist, squareform
    d = ripser(squareform(pdist(X)), distance_matrix=True, maxdim=1)["dgms"][1]
    return list(d[:, 1] - d[:, 0])


def test_rips_reads_float64_distances():
    """The H1 bars of two far-apart clusters are those of each cluster; read
    through float64 pdist they match to ripser's float32 rounding (5e-8),
    while the point-cloud path was off by ~1e-4."""
    X, A, B0 = _two_far_clusters()
    want = _h1_lifetimes(A) + _h1_lifetimes(B0)
    assert len(want) >= 3
    S = fd.inv.States(blocks={l: [X[i:i + 1] for i in range(len(X))] for l in range(3)})
    res = fd.inv.run_task_on_states("topology_homology", S, {"num_samples": len(X)},
                                    force=True, store_dtype=torch.float64)
    assert res["first_layer_num_loops_h1"] == len(want)
    assert res["first_layer_mean_persistence_h1"] == pytest.approx(np.mean(want), rel=1e-6)


def test_a_smaller_cache_is_not_silently_accepted(tiny):
    """min_cached_docs: a cache holding fewer documents than requested is
    rejected and the task collects the full N itself."""
    model, tok, _ = tiny
    small = ModelOutputCache(model, tok, dataset=fd.dataset(), num_samples=5)
    small.populate(need_hidden=True)
    pts, info = last_token_points(model, tok, fd.dataset(), small,
                                  num_samples=len(fd.TEXTS), layers=role_layers)
    assert info["source"] == "self_collected" and info["cache_rejected"] == "too_few_docs"
    assert all(len(X) == len(fd.TEXTS) for X in pts.values())


@pytest.mark.parametrize("task", sorted(TASKS))
def test_a_short_dataset_is_recorded_against_the_requested_n(task, tiny, caplog):
    """A caller dataset smaller than N caps the cloud; the result and the log
    say so (the study once passed a 500-passage corpus to N = 1000 tasks)."""
    model, tok, _ = tiny
    n = len(fd.TEXTS)
    with caplog.at_level("WARNING", logger="blme"):
        res = _cls(task)({"num_samples": 1000}).evaluate(model, tok, fd.dataset(), cache=None)
    assert res["_meta_n_points"] == n and res["_meta_n_points_requested"] == 1000
    assert f"{n} of the 1000 requested sequences" in caplog.text
