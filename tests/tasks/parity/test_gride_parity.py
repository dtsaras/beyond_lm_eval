"""Parity: geometry_gride and geometry_information_imbalance vs the RELEASED
DADApy (sissa-data-science/DADApy@d729b88), called the way Cheng et al.'s
scripts (chengemily1/id-llm-abstraction@4a170a2) call it:

* ``dadapy_id.py``: ``Data(reps)``, ``remove_identical_points()``,
  ``return_id_scaling_gride(range_max=2**13)``;
* ``precompute_distances.py`` + ``information_imbalance.py``:
  ``Data(reps).compute_distances(maxk=N-1)``, then
  ``_return_imbalance(indices_a, indices_b, rng, k=1)``.

DADApy is not a BLME dependency: it is installed with ``uv pip install
--no-deps --target ~/.cache/blme-sideinstalls/dadapy-d729b88`` from the clone
(BLME_SIDEINSTALLS overrides the root). Skips when absent (fails under
BLME_REQUIRE_REFS=1).
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import invariance as inv  # noqa: E402
from blme.tasks.geometry import gride as G  # noqa: E402

_ROOT = os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))
_SIDE = os.path.join(os.environ.get("BLME_SIDEINSTALLS", os.path.expanduser("~/.cache/blme-sideinstalls")),
                     "dadapy-d729b88")


def _missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


@pytest.fixture(scope="module")
def dadapy():
    if not os.path.isdir(os.path.join(_SIDE, "dadapy")):
        _missing(_SIDE)
    sys.path.insert(0, _SIDE)
    try:
        import dadapy
        from dadapy._utils.metric_comparisons import _return_imbalance
    finally:
        sys.path.remove(_SIDE)
    return dadapy, _return_imbalance


@pytest.mark.parametrize("repo,task", [("DADApy", G.GrideProfileTask),
                                        ("id-llm-abstraction", G.InformationImbalanceTask),
                                        ("DADApy", G.NeighborhoodOverlapTask)])
def test_sources_pin_the_cloned_commits(repo, task):
    path = os.path.join(_ROOT, repo)
    if not os.path.isdir(path):
        _missing(path)
    head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert ("@" + head) in " ".join(task.SOURCES.values())


def _manifold(n, intrinsic, ambient, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, intrinsic))
    A = rng.standard_normal((intrinsic, ambient)) / np.sqrt(intrinsic)
    return scale * np.tanh(Z @ A) + 1e-3 * rng.standard_normal((n, ambient))


@pytest.mark.parametrize("n,intrinsic,ambient", [(600, 4, 40), (1000, 10, 64), (300, 2, 16)])
def test_gride_scaling_matches_dadapy(dadapy, n, intrinsic, ambient):
    dp, _ = dadapy
    X = _manifold(n, intrinsic, ambient, seed=n)
    data = dp.Data(X.copy(), verbose=False)
    data.remove_identical_points()
    ids_ref, _, ranks_ref = data.return_id_scaling_gride(range_max=2 ** 13, return_ranks=True)
    ids, ranks = G._gride_scaling(X, 2 ** 13)
    np.testing.assert_array_equal(ranks, ranks_ref)
    np.testing.assert_allclose(ids, ids_ref, rtol=1e-6, atol=1e-6)


def test_gride_recovers_a_known_dimension():
    """Known positive: a 5-D Gaussian in 30 dimensions reads about 5 at small scale."""
    rng = np.random.default_rng(3)
    X = np.c_[rng.standard_normal((1000, 5)), np.zeros((1000, 25))]
    ids, ranks = G._gride_scaling(X, 2 ** 13)
    assert ids[list(ranks).index(4)] == pytest.approx(5.0, rel=0.1)


@pytest.mark.parametrize("seed", [0, 1])
def test_information_imbalance_matches_dadapy(dadapy, seed):
    dp, ret_imb = dadapy
    A = _manifold(400, 3, 24, seed)
    B = np.tanh(A @ np.random.default_rng(seed + 7).standard_normal((24, 24)))
    ia, ib = G._full_rank_indices(A), G._full_rank_indices(B)
    da, db = dp.Data(A.copy(), verbose=False), dp.Data(B.copy(), verbose=False)
    da.compute_distances(maxk=A.shape[0] - 1)
    db.compute_distances(maxk=B.shape[0] - 1)
    np.testing.assert_array_equal(ia, da.dist_indices)
    rng = np.random.default_rng(0)
    assert G._information_imbalance(ia, ib) == ret_imb(da.dist_indices, db.dist_indices, rng, k=1)
    assert G._information_imbalance(ib, ia) == ret_imb(db.dist_indices, da.dist_indices, rng, k=1)
    assert G._information_imbalance(ia, ia) == pytest.approx(2.0 / A.shape[0])   # rank 1 everywhere


def _layered_states(n=300, d=48, layers=4, seed=5):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(2, 5, size=n)
    base = [rng.standard_normal((int(t), d)) for t in lengths]
    mats = [np.linalg.qr(rng.standard_normal((d, d)))[0] * (1.0 + 0.3 * l) for l in range(layers)]
    return {l: [torch.tensor(np.tanh(x @ mats[l]), dtype=torch.float32) for x in base]
            for l in range(layers)}


def test_evaluate_matches_dadapy_per_layer(dadapy):
    """The real evaluate() on injected states against DADApy on the clouds it
    scores (the last row of each document, stored as float32 by the cache)."""
    dp, ret_imb = dadapy
    docs = _layered_states()
    g = inv.run_task_on_states("geometry_gride", docs)
    ii = inv.run_task_on_states("geometry_information_imbalance", docs)
    clouds = [torch.stack([doc[-1] for doc in docs[l]]).double().numpy() for l in sorted(docs)]
    for l, X in enumerate(clouds):
        data = dp.Data(X.copy(), verbose=False)
        data.remove_identical_points()
        ids_ref, _, ranks = data.return_id_scaling_gride(range_max=2 ** 13, return_ranks=True)
        assert g["gride_id_profile"][l] == pytest.approx(ids_ref[list(ranks).index(4)], rel=1e-6)
    ref_idx = []
    for X in clouds:
        dt = dp.Data(X.copy(), verbose=False)
        dt.compute_distances(maxk=X.shape[0] - 1)
        ref_idx.append(dt.dist_indices)
    rng = np.random.default_rng(0)
    assert ii["ii_first_to_last"] == ret_imb(ref_idx[0], ref_idx[-1], rng, k=1)
    assert ii["ii_last_to_first"] == ret_imb(ref_idx[-1], ref_idx[0], rng, k=1)


@pytest.mark.parametrize("seed,k", [(0, 30), (1, 10)])
def test_neighbourhood_overlap_matches_dadapy(dadapy, seed, k):
    dp, _ = dadapy
    from dadapy import NeighborhoodOverlap
    A = _manifold(400, 3, 24, seed)
    B = np.tanh(A @ np.random.default_rng(seed + 7).standard_normal((24, 24)))
    ref = NeighborhoodOverlap(A.copy(), verbose=False).return_data_overlap(B.copy(), k=k)
    got = G._neighbourhood_overlap(G._full_rank_indices(A), G._full_rank_indices(B), k)
    assert got == pytest.approx(ref, abs=1e-15)
    assert G._neighbourhood_overlap(G._full_rank_indices(A), G._full_rank_indices(A), k) == 1.0


def test_neighbourhood_overlap_evaluate_matches_dadapy(dadapy):
    from dadapy import NeighborhoodOverlap
    docs = _layered_states()
    r = inv.run_task_on_states("geometry_neighborhood_overlap", docs, config={"k": 30})
    clouds = [torch.stack([doc[-1] for doc in docs[l]]).double().numpy() for l in sorted(docs)]
    ref = [NeighborhoodOverlap(X.copy(), verbose=False).return_data_overlap(clouds[-1].copy(), k=30)
           for X in clouds]
    assert r["nn_overlap_layer_to_last"] == pytest.approx(ref, abs=1e-15)
    adj = [NeighborhoodOverlap(a.copy(), verbose=False).return_data_overlap(b.copy(), k=30)
           for a, b in zip(clouds, clouds[1:])]
    assert r["nn_overlap_adjacent_min"] == pytest.approx(min(adj), abs=1e-15)
