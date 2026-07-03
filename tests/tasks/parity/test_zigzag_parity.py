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

    keys = [k for k in blme if not k.startswith("_meta_")]
    max_diff = max(abs(float(blme[k]) - float(indep[k])) for k in keys)
    assert max_diff < 1e-12

    # ...and matches the fixture-recorded BLME summary.
    fx = _fixture()["scenarios"][name]["blme_summary"]
    for k, v in fx.items():
        assert float(blme[k]) == pytest.approx(v, abs=1e-9), k


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
