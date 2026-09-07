"""Numeric-parity test: BLME topology_zigzag_persistence vs the paper's zigzag.

TASK: topology_zigzag_persistence
BLME: src/blme/tasks/topology/zigzag.py
      ZigzagPersistenceTask.evaluate() -> _zigzag_summary(_layer_topology,
      _contiguous_bands).

Paper: Gardinazzi, Viswanathan, Panerai, Ansuini, Cazzaniga & Biagetti,
       "Persistent Topological Features in Large Language Models",
       ICML 2025 (poster), arXiv:2410.11042.
Repo:  RitAreaSciencePark/ZigZagLLMs @ bcfe0a6 (src/zigzag/zigzag_DL.py).

VERDICT: FAITHFUL PROXY. BLME cannot depend on the paper's zigzag engine
(dionysus / FastZigZag `fzz` — C++ builds, not repo deps, and the system's
Boost 1.74 breaks the dionysus build). The self-contained helper detects
per-layer significant H0/H1 (graph-geodesic kNN + ripser, the same
construction as topology_betti_curve) and summarises how features PERSIST
across the layer axis as contiguous layer-bands.

The bar this test pins:

  (A) GROUND-TRUTH ANCHORS — synthetic layer-sequences with KNOWN feature
      lifetimes (a circle placed only in the middle layers -> a loop present
      exactly in those layers). BLME must recover the exact lifetime.

  (B) DIONYSUS ANCHOR — the reference dionysus zigzag barcode for these same
      constructions was computed in an isolated venv (see
      $SCRATCH/newtasks/dump_reference.py) and recorded in the fixture. We
      assert BLME's band count / lifetime matches the dionysus H1 bars, and
      that the recorded dionysus numbers are the KNOWN-by-construction ones
      (one bar spanning the loop layers; zero bars when there is no loop).

  (C) INDEPENDENT REIMPLEMENTATION — an independent, from-scratch
      reimplementation of the summary (different code path, same math) must
      match BLME bit-for-bit (< 1e-12) on every scenario.

src/blme is NOT modified. dionysus is NOT imported here (the reference numbers
live in the fixture); only ripser/scipy (BLME deps) are used.
"""

import json
from pathlib import Path

import numpy as np
import pytest

ripser_mod = pytest.importorskip("ripser")

from blme.tasks.topology.zigzag import (  # noqa: E402
    _zigzag_summary,
    _layer_topology,
    _contiguous_bands,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures/reference_parity/parity/zigzag.json"
)


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Ground-truth layer-sequences (MUST match dump_reference.py / zigzag_verify.py)
# ---------------------------------------------------------------------------
def _make_clouds():
    np.random.seed(0)
    n = 20
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1).astype(np.float64)
    line = np.stack([np.linspace(-1, 1, n), np.zeros(n)], axis=1).astype(np.float64)
    return {
        "circle_middle": [line, line, circle, circle, line, line],
        "circle_all": [circle, circle, circle, circle, circle],
        "no_loop": [line, line, line, line, line],
        "circle_single_mid": [line, line, circle, line, line],
    }


def _make_nondegenerate_clouds():
    """Scenarios where the 'redundant-looking' metrics genuinely differ:
    a layer with TWO disjoint circles separates h1_loop_layer_sum from
    h1_total_layer_persistence, and two-cluster layers exercise the H0
    metrics away from the constant 1.0 of the original scenarios.
    (Not dionysus-anchored: ground truth is known by construction and
    cross-checked against the independent reimplementation.)"""
    base = _make_clouds()
    circle = base["circle_all"][0]
    line = base["no_loop"][0]
    two_circles = np.concatenate([circle, circle + np.array([10.0, 0.0])])
    two_lines = np.concatenate([line, line + np.array([0.0, 10.0])])
    return {
        "double_loop_mid": [circle, two_circles, circle],
        "two_clusters_mid": [line, two_lines, two_lines, line],
    }


# ---------------------------------------------------------------------------
# Independent from-scratch reimplementation of the summary (different code path)
# ---------------------------------------------------------------------------
def _independent_summary(layer_clouds, n_neighbors=5, persistence_frac=0.3):
    import warnings

    from ripser import ripser
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, shortest_path
    from scipy.spatial import cKDTree

    def topo(X):
        X = np.asarray(X, float)
        X = X[np.all(np.isfinite(X), axis=1)]
        n = len(X)
        if n < 3:
            return max(n, 0), 0
        k = int(min(n_neighbors, n - 1))
        tr = cKDTree(X)
        dd, ii = tr.query(X, k=k + 1)
        dd = np.atleast_2d(dd)
        ii = np.atleast_2d(ii)
        r = np.repeat(np.arange(n), k)
        c = ii[:, 1:].ravel()
        v = dd[:, 1:].ravel()
        g = csr_matrix((v, (r, c)), shape=(n, n))
        g = g.maximum(g.T)
        nc, _ = connected_components(g, directed=False)
        geo = shortest_path(g, method="D", directed=False)
        fin = geo[np.isfinite(geo)]
        if fin.size == 0:
            return int(nc), 0
        sc = float(fin.max())
        if sc <= 0:
            return int(nc), 0
        geo = np.where(np.isfinite(geo), geo, sc * 2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dg = ripser(geo, distance_matrix=True, maxdim=1)["dgms"]
        h1 = dg[1] if len(dg) > 1 else []
        nb1 = sum(
            1 for b, de in h1
            if np.isfinite(de) and (de - b) > persistence_frac * sc
        )
        return int(nc), int(nb1)

    comps, loops = [], []
    for X in layer_clouds:
        c, h = topo(X)
        comps.append(c)
        loops.append(h)
    comps = np.array(comps, float)
    loops = np.array(loops, int)
    L = len(layer_clouds)

    # bands via explicit scan (deliberately NOT _contiguous_bands)
    bands, run = [], None
    for i in range(L):
        if loops[i] >= 1:
            run = [i, i] if run is None else [run[0], i]
        elif run is not None:
            bands.append(tuple(run))
            run = None
    if run is not None:
        bands.append(tuple(run))

    if bands:
        lengths = [e - s + 1 for s, e in bands]
        longest = max(bands, key=lambda be: be[1] - be[0])
        denom = max(1, L - 1)
        first, last = longest[0] / denom, longest[1] / denom
        maxlen = float(max(lengths))
    else:
        first = last = maxlen = 0.0

    return {
        "h1_total_layer_persistence": float((loops >= 1).sum()),
        "h1_loop_layer_sum": float(loops.sum()),
        "h1_num_bands": float(len(bands)),
        "h1_max_band_length": maxlen,
        "h1_first_layer": float(first),
        "h1_last_layer": float(last),
        "betti1_fraction_layers": float((loops >= 1).sum()) / L,
        "h0_mean_components": float(comps.mean()),
        "h0_persistence_range": float(comps.max() - comps.min()),
        "h0_first_layer_components": float(comps[0]),
        "h0_last_layer_components": float(comps[-1]),
    }


# ===========================================================================
# (A) Ground-truth anchors: KNOWN feature lifetimes recovered exactly.
# ===========================================================================
def test_anchor_circle_middle_loop_lifetime():
    """A circle placed only in layers 2,3 -> loop present exactly there,
    one contiguous band, everything else loop-free."""
    clouds = _make_clouds()["circle_middle"]
    loops = [h1 for _, h1 in (_layer_topology(X, 5, 0.3) for X in clouds)]
    assert [i for i, v in enumerate(loops) if v >= 1] == [2, 3]

    s = _zigzag_summary(clouds, n_neighbors=5, persistence_frac=0.3)
    assert s["h1_num_bands"] == 1.0
    assert s["h1_max_band_length"] == 2.0
    assert s["h1_total_layer_persistence"] == 2.0
    # normalized band extent over 6 layers: layers 2,3 -> 2/5, 3/5
    assert s["h1_first_layer"] == pytest.approx(0.4, abs=1e-12)
    assert s["h1_last_layer"] == pytest.approx(0.6, abs=1e-12)


def test_anchor_no_loop_is_empty():
    clouds = _make_clouds()["no_loop"]
    s = _zigzag_summary(clouds, n_neighbors=5, persistence_frac=0.3)
    assert s["h1_num_bands"] == 0.0
    assert s["h1_total_layer_persistence"] == 0.0
    assert s["h1_max_band_length"] == 0.0
    assert s["betti1_fraction_layers"] == 0.0


def test_anchor_circle_all_single_full_depth_band():
    clouds = _make_clouds()["circle_all"]
    s = _zigzag_summary(clouds, n_neighbors=5, persistence_frac=0.3)
    assert s["h1_num_bands"] == 1.0
    assert s["h1_max_band_length"] == float(len(clouds))
    assert s["h1_first_layer"] == 0.0
    assert s["h1_last_layer"] == 1.0
    assert s["betti1_fraction_layers"] == 1.0


def test_anchor_single_mid_layer_loop():
    clouds = _make_clouds()["circle_single_mid"]
    loops = [h1 for _, h1 in (_layer_topology(X, 5, 0.3) for X in clouds)]
    assert [i for i, v in enumerate(loops) if v >= 1] == [2]

    s = _zigzag_summary(clouds, n_neighbors=5, persistence_frac=0.3)
    assert s["h1_num_bands"] == 1.0
    assert s["h1_max_band_length"] == 1.0
    assert s["h1_first_layer"] == pytest.approx(0.5, abs=1e-12)
    assert s["h1_last_layer"] == pytest.approx(0.5, abs=1e-12)


# ===========================================================================
# (B) Dionysus anchor: BLME band count matches recorded dionysus H1 bars, and
#     the recorded dionysus numbers ARE the known-by-construction lifetimes.
# ===========================================================================
@pytest.mark.parametrize("name", ["circle_middle", "circle_all", "no_loop", "circle_single_mid"])
def test_dionysus_anchor_band_count_matches(name):
    clouds = _make_clouds()[name]
    s = _zigzag_summary(clouds, n_neighbors=5, persistence_frac=0.3)
    fx = _fixture()["scenarios"][name]

    n_ref = fx["dionysus_n_H1_bars"]
    assert int(s["h1_num_bands"]) == int(n_ref)

    # Anchor the ONSET LAYER. dionysus reports zigzag births/deaths in
    # 2L-1 sequence-index units; the loop is "born" at the intersection node
    # *after* the layer where it first appears, so the layer-mapped birth is
    # offset by +0.5 (documented in the fixture's index_convention_note).
    # The exact, engine-consistent relationship, verified on all scenarios, is
    #   floor(dionysus_birth_layer) == BLME band-onset layer.
    if n_ref >= 1:
        ref_layer_bars = fx["dionysus_H1_bars_layer"]
        assert len(ref_layer_bars) == n_ref
        L = len(clouds)
        band_first_layer = int(round(s["h1_first_layer"] * (L - 1)))
        dionysus_births = [int(np.floor(b)) for b, _ in ref_layer_bars]
        assert band_first_layer in dionysus_births


def test_fixture_dionysus_numbers_are_ground_truth():
    """The dionysus reference recorded in the fixture must itself equal the
    known-by-construction answer (one bar over the loop layers; none when
    loop-free). This guards against a stale/incorrect reference dump."""
    fx = _fixture()["scenarios"]
    assert fx["circle_middle"]["dionysus_H1_bars_zigzag_idx"] == [[5.0, 8.0]]
    assert fx["circle_all"]["dionysus_H1_bars_zigzag_idx"] == [[1.0, 10.0]]
    assert fx["no_loop"]["dionysus_H1_bars_zigzag_idx"] == []
    assert fx["circle_single_mid"]["dionysus_H1_bars_zigzag_idx"] == [[5.0, 6.0]]


# ===========================================================================
# (C) BLME == independent reimplementation, bit-exact, all scenarios.
# ===========================================================================
@pytest.mark.parametrize("name", ["circle_middle", "circle_all", "no_loop", "circle_single_mid"])
def test_blme_matches_independent_reimplementation(name):
    clouds = _make_clouds()[name]
    blme = _zigzag_summary(clouds, n_neighbors=5, persistence_frac=0.3)
    indep = _independent_summary(clouds, n_neighbors=5, persistence_frac=0.3)

    # Compare on the keys the independent reimplementation defines. BLME also
    # emits the identity-aware `zz_h1_*` family (GF(2) induced-rank matching
    # across layers), which this reimplementation does not model; those have
    # their own ground-truth tests. Assert coverage in that direction so a
    # legacy key silently disappearing still fails here.
    keys = [k for k in indep if not k.startswith("_meta_")]
    missing = [k for k in keys if k not in blme]
    assert not missing, f"BLME dropped reference keys: {missing}"
    max_diff = max(abs(float(blme[k]) - float(indep[k])) for k in keys)
    assert max_diff < 1e-12

    # ...and matches the fixture-recorded BLME summary.
    fx = _fixture()["scenarios"][name]["blme_summary"]
    for k, v in fx.items():
        assert float(blme[k]) == pytest.approx(v, abs=1e-9), k


# ===========================================================================
# (A') Non-degenerate ground-truth anchors: metrics that the original four
#      scenarios could not distinguish (loop multiplicity, H0 structure).
# ===========================================================================
def test_anchor_double_loop_separates_sum_from_persistence():
    """Layer 1 holds TWO disjoint circles (2 loops), layers 0/2 one circle.
    loops-per-layer = [1, 2, 1], so the loop-count SUM (4) must differ from
    the #layers-with-a-loop count (3). A regression that conflates
    loops.sum() with mask.sum() fails here and nowhere else."""
    clouds = _make_nondegenerate_clouds()["double_loop_mid"]
    loops = [h1 for _, h1 in (_layer_topology(X, 5, 0.3) for X in clouds)]
    assert loops == [1, 2, 1]

    s = _zigzag_summary(clouds, n_neighbors=5, persistence_frac=0.3)
    assert s["h1_loop_layer_sum"] == 4.0
    assert s["h1_total_layer_persistence"] == 3.0
    assert s["h1_num_bands"] == 1.0
    assert s["h1_max_band_length"] == 3.0
    assert s["betti1_fraction_layers"] == 1.0
    # H0 is non-constant too: components = [1, 2, 1].
    assert s["h0_mean_components"] == pytest.approx(4.0 / 3.0, abs=1e-12)
    assert s["h0_persistence_range"] == 1.0


def test_anchor_two_clusters_h0_metrics():
    """Middle layers split into two far-apart clusters: components per layer
    = [1, 2, 2, 1]. Pins the H0 summary metrics away from the constant-1.0
    degenerate regime of the original scenarios; no loops anywhere."""
    clouds = _make_nondegenerate_clouds()["two_clusters_mid"]
    s = _zigzag_summary(clouds, n_neighbors=5, persistence_frac=0.3)
    assert s["h0_mean_components"] == 1.5
    assert s["h0_persistence_range"] == 1.0
    assert s["h0_first_layer_components"] == 1.0
    assert s["h0_last_layer_components"] == 1.0
    assert s["h1_num_bands"] == 0.0
    assert s["h1_loop_layer_sum"] == 0.0
    assert s["betti1_fraction_layers"] == 0.0


@pytest.mark.parametrize("name", ["double_loop_mid", "two_clusters_mid"])
def test_blme_matches_independent_reimpl_nondegenerate(name):
    """(C) bit-exact independent-reimplementation parity on the
    non-degenerate scenarios (no fixture entry: ground truth is analytic)."""
    clouds = _make_nondegenerate_clouds()[name]
    blme = _zigzag_summary(clouds, n_neighbors=5, persistence_frac=0.3)
    indep = _independent_summary(clouds, n_neighbors=5, persistence_frac=0.3)
    # Compare on the keys the independent reimplementation defines. BLME also
    # emits the identity-aware `zz_h1_*` family (GF(2) induced-rank matching
    # across layers), which this reimplementation does not model; those have
    # their own ground-truth tests. Assert coverage in that direction so a
    # legacy key silently disappearing still fails here.
    keys = [k for k in indep if not k.startswith("_meta_")]
    missing = [k for k in keys if k not in blme]
    assert not missing, f"BLME dropped reference keys: {missing}"
    max_diff = max(abs(float(blme[k]) - float(indep[k])) for k in keys)
    assert max_diff < 1e-12


# ===========================================================================
# Helper-level unit checks.
# ===========================================================================
def test_contiguous_bands_helper():
    assert _contiguous_bands(np.array([False, False, True, True, False, False])) == [(2, 3)]
    assert _contiguous_bands(np.array([True, False, True, False, True])) == [(0, 0), (2, 2), (4, 4)]
    assert _contiguous_bands(np.array([False, False, False])) == []
    assert _contiguous_bands(np.array([True, True, True])) == [(0, 2)]


def test_degenerate_inputs_do_not_crash():
    # too-few points per layer
    tiny = [np.zeros((2, 3)), np.ones((2, 3))]
    s = _zigzag_summary(tiny, n_neighbors=5, persistence_frac=0.3)
    assert s["h1_num_bands"] == 0.0
    # empty layer list
    assert _zigzag_summary([], 5, 0.3) == {"_meta_n_layers": 0}


def test_fixture_records_parity_verdict():
    fx = _fixture()
    assert fx["task"] == "topology_zigzag_persistence"
    assert fx["verdict"] == "FAITHFUL-PROXY"
    assert fx["independent_reimpl_pass"] is True
    assert fx["dionysus_anchor_pass"] is True
    assert fx["anchors"]["all_anchors_pass"] is True


# ---------------------------------------------------------------------------
# Identity-aware cross-layer H1 matching (GF(2) induced ranks).
#
# The band summary cannot distinguish "one loop persists across N layers" from
# "N different loops, one per layer" -- both give a single band of length N.
# These tests pin the distinction the `zz_h1_*` family exists to make.
# ---------------------------------------------------------------------------

def _circle(n, seed=None, jitter=0.01):
    import numpy as _np
    th = _np.linspace(0, 2 * _np.pi, n, endpoint=False)
    X = _np.c_[_np.cos(th), _np.sin(th)]
    if jitter:
        X = X + jitter * _np.random.default_rng(seed).standard_normal((n, 2))
    return X


def test_same_loop_persists_and_shuffled_loop_does_not():
    import numpy as np
    from blme.tasks.topology.zigzag import _zigzag_summary

    n, n_layers = 40, 6
    circ = _circle(n, jitter=0.0)
    rng = np.random.default_rng(0)

    same = [circ + 0.01 * rng.standard_normal((n, 2)) for _ in range(n_layers)]
    shuffled = [circ[rng.permutation(n)] + 0.01 * rng.standard_normal((n, 2))
                for _ in range(n_layers)]

    r_same = _zigzag_summary(same, n_neighbors=5)
    r_shuf = _zigzag_summary(shuffled, n_neighbors=5)

    # The legacy band summary is BLIND here: identical on both.
    assert r_same["h1_num_bands"] == r_shuf["h1_num_bands"]
    assert r_same["h1_max_band_length"] == r_shuf["h1_max_band_length"]

    # The identity-aware family separates them.
    assert r_same["_meta_zz_h1_survival_rate"] == pytest.approx(1.0)
    assert r_same["zz_h1_turnover_rate"] == pytest.approx(0.0)
    assert r_shuf["_meta_zz_h1_survival_rate"] == pytest.approx(0.0)
    assert r_shuf["zz_h1_turnover_rate"] > 0.0


def test_induced_rank_matches_hand_computed_homology():
    """Direct check of the GF(2) machinery on a hand-checkable case."""
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    from blme.tasks.topology.zigzag import (
        _f2_rank, _fundamental_cycles, _graph_edges, _h1_persisting_across,
    )

    n = 24
    X = _circle(n, jitter=0.0)
    D = squareform(pdist(X))
    eps = 0.30
    edges = _graph_edges(D, eps)
    edge_index = {e: i for i, e in enumerate(edges)}

    # A single cycle graph on n vertices: |E| - |V| + c = n - n + 1 = 1.
    cycles = _fundamental_cycles(edges, n, edge_index)
    assert len(cycles) == len(edges) - n + 1

    # Every returned mask must be an actual cycle: each vertex touched an
    # even number of times (kernel of the GF(2) boundary map d1).
    for mask in cycles:
        deg = [0] * n
        for e, i in edge_index.items():
            if (mask >> i) & 1:
                deg[e[0]] += 1
                deg[e[1]] += 1
        assert all(d % 2 == 0 for d in deg)

    assert _f2_rank(cycles) == len(cycles)

    # Same circle in both layers -> the class survives.
    rf, rg, shared = _h1_persisting_across(D, D, eps)
    assert (rf, rg, shared) == (1, 1, 1)

    # Reviewer counterexample (2026-09-03) that the UNION construction fails
    # and the paper's INTERSECTION construction gets right: a hexagon whose
    # H1 class is nonzero in both layers, but whose union flag complex fills
    # it in. Union reports 0 persisting; intersection reports 1.
    from blme.tasks.topology.zigzag import _h1_persisting_edges
    hexagon = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    layer_a = hexagon + [(0, 6), (1, 6), (2, 6)]
    layer_b = hexagon + [(3, 6), (4, 6), (5, 6)]
    assert _h1_persisting_edges(layer_a, layer_b, 7) == (1, 1, 1)

    # A different cycle on the SAME vertices -> nothing survives, even though
    # both layers individually contain exactly one loop.
    perm = np.random.default_rng(0).permutation(n)
    D2 = squareform(pdist(_circle(n, jitter=0.0)[perm]))
    rf2, rg2, shared2 = _h1_persisting_across(D, D2, eps)
    assert rf2 == 1 and rg2 == 1
    assert shared2 == 0


def test_survival_rate_is_scale_stable_while_raw_counts_are_not():
    """The raw H1 dimension of a kNN flag complex is cycle rank and grows with
    the point count, so only the RATIO is comparable across models."""
    import numpy as np
    from blme.tasks.topology.zigzag import _zigzag_summary

    rng = np.random.default_rng(0)
    rates = []
    for n in (40, 80, 160):
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        circ = np.c_[np.cos(th), np.sin(th)]
        layers = [circ + 0.01 * rng.standard_normal((n, 2)) for _ in range(6)]
        r = _zigzag_summary(layers, n_neighbors=5)
        rates.append(r["_meta_zz_h1_survival_rate"])
    assert all(x == pytest.approx(1.0) for x in rates)

    # And the raw count must NOT be exported as a feature.
    r = _zigzag_summary(layers, n_neighbors=5)
    assert "zz_h1_persisting_mean" not in r
    assert "_meta_zz_h1_persisting_mean" in r


def test_f2_intersect_is_a_correct_zassenhaus():
    """GF(2) subspace intersection, against the dimension formula and by
    explicit membership. This code had no coverage when it shipped."""
    import numpy as np
    from blme.tasks.topology.zigzag import _f2_intersect, _f2_rank

    rng = np.random.default_rng(0)
    n = 12
    for _ in range(500):
        U = [int(rng.integers(0, 1 << n))
             for _ in range(int(rng.integers(1, 6)))]
        W = [int(rng.integers(0, 1 << n))
             for _ in range(int(rng.integers(1, 6)))]
        inter = _f2_intersect(U, W, n)
        # dim(U n W) = dim U + dim W - dim(U + W)
        expected = _f2_rank(U) + _f2_rank(W) - _f2_rank(list(U) + list(W))
        assert _f2_rank(inter) == expected
        # and every returned vector really lies in both spans
        for v in inter:
            assert _f2_rank(list(U) + [v]) == _f2_rank(U)
            assert _f2_rank(list(W) + [v]) == _f2_rank(W)


def test_turnover_rate_is_bounded_and_scale_free():
    """`zz_h1_turnover_rate` is a Jaccard distance on the two layers' H1
    classes, so it is bounded in [0, 1] and does not grow with the point count.

    Regression: dividing by `max(1, dim_a)` turned steps with no H1 class into
    unnormalized COUNTS, and the mean went 12 -> 385 as N went 40 -> 400.
    """
    import numpy as np
    from blme.tasks.topology.zigzag import _zigzag_summary

    rng = np.random.default_rng(0)
    for n in (40, 120, 400):
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        circ = np.c_[np.cos(th), np.sin(th)]
        # Alternate loop / no-loop layers: this is what produces dim_a == 0
        # steps, the case that leaked raw counts.
        layers = [circ + 0.01 * rng.standard_normal((n, 2)) if i % 2
                  else rng.standard_normal((n, 2)) * 0.4
                  for i in range(8)]
        r = _zigzag_summary(layers, n_neighbors=5)
        assert 0.0 <= r["zz_h1_turnover_rate"] <= 1.0
        assert 0.0 <= r["_meta_zz_h1_survival_rate"] <= 1.0
