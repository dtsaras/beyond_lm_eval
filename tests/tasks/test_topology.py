"""
Tests for all 3 topology tasks.
Each test is parameterized over GPT2, Llama, and BERT via conftest.py.

All topology tasks require the `ripser` library for persistent homology.
Tests are skipped if ripser is not installed.
"""
import pytest
import torch
import numpy as np


@pytest.mark.parametrize("mod_path,cls_name", [
    ("blme.tasks.topology.homology", "PersistentHomologyTask"),
    ("blme.tasks.topology.betti_curve", "BettiCurveTask"),
    ("blme.tasks.topology.persistence_entropy", "PersistenceEntropyTask"),
])
def test_topology_falls_back_on_dataset_network_error(mod_path, cls_name, monkeypatch):
    """Campaign-3 release blocker: the dataset=None path caught only ImportError
    around load_dataset('EleutherAI/lambada_openai'), so a network/offline OSError
    crashed the task instead of falling back to the bundled examples."""
    pytest.importorskip("ripser")
    import importlib
    import datasets
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def _boom(*a, **k):
        raise OSError("Couldn't reach the Hugging Face Hub (offline)")

    monkeypatch.setattr(datasets, "load_dataset", _boom)
    cls = getattr(importlib.import_module(mod_path), cls_name)
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    # Must NOT raise OSError — falls back to bundled examples and returns a dict.
    r = cls(config={"num_samples": 5}).evaluate(model, tok, dataset=None)
    assert isinstance(r, dict)


def test_count_betti_raises_clear_error_without_ripser(monkeypatch):
    """Campaign-3 release blocker: _count_betti called ripser() even when the
    optional dep is missing, crashing with a bare NameError. It must raise a
    clear ImportError with an install hint."""
    import blme.tasks.topology.betti_curve as bc

    monkeypatch.setattr(bc, "HAS_RIPSER", False)
    with pytest.raises(ImportError, match="ripser"):
        bc._count_betti(np.random.default_rng(0).standard_normal((5, 2)))


def test_betti_zero_distance_edges_do_not_inflate_beta0():
    """Campaign-3 BUG: zero-weight kNN edges (between duplicate points) were
    stored as explicit zeros, which scipy's csgraph treats as "no edge", so a
    cluster of duplicates disconnected and inflated beta_0. Duplicate points
    must stay one connected component."""
    pytest.importorskip("ripser")
    from blme.tasks.topology.betti_curve import _count_betti

    dup = np.zeros((8, 2))  # 8 EXACT duplicates -> all kNN distances are 0
    far = np.array([[100.0, 0.0], [100.1, 0.0], [100.0, 0.1],
                    [100.1, 0.1], [100.2, 0.0]])
    X = np.vstack([dup, far])
    beta_0, _ = _count_betti(X, maxdim=1, n_neighbors=4)
    assert beta_0 == 2, (
        f"beta_0={beta_0}: the duplicate cluster (1 component) + far blob (1) "
        f"should give 2, not inflate when zero-distance edges are dropped"
    )


def test_persistent_homology(mock_model, mock_tokenizer):
    """Persistent homology via Vietoris-Rips complex."""
    from blme.tasks.topology.homology import PersistentHomologyTask

    task = PersistentHomologyTask(config={"num_samples": 5})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # May error if get_layers returns None for this architecture
    if "error" not in results:
        assert any("persistence_h0" in k for k in results.keys())


def test_betti_curve(mock_model, mock_tokenizer):
    """Betti number trajectory across layers, with the derived summary
    statistics numerically pinned to their definitions (recomputed here
    from the returned curves) — a source-grep test alone lets the polyfit
    degree/coefficient silently change."""
    import numpy as np
    from blme.tasks.topology.betti_curve import BettiCurveTask

    task = BettiCurveTask(config={"num_samples": 5})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # May error if get_layers returns None for this architecture
    if "error" not in results:
        curve0 = results["betti_0_curve"]
        curve1 = results["betti_1_curve"]
        L = len(curve0)
        assert results["betti_0_first"] == curve0[0]
        assert results["betti_0_last"] == curve0[-1]
        # simplification_ratio = last/first (1.0 fallback when first == 0)
        expected_ratio = curve0[-1] / curve0[0] if curve0[0] > 0 else 1.0
        assert results["simplification_ratio"] == pytest.approx(
            expected_ratio, abs=1e-12
        )
        # decay rate = degree-1 polyfit SLOPE on normalized depth in [0, 1]
        if L > 1:
            x = np.arange(L, dtype=np.float64) / float(max(1, L - 1))
            expected_slope = float(np.polyfit(x, curve0, 1)[0])
            assert results["betti_0_decay_rate"] == pytest.approx(
                expected_slope, abs=1e-9
            )
        assert results["max_betti_1_layer"] == int(np.argmax(curve1))
        assert results["max_betti_1"] == int(np.max(curve1))


def test_betti_curve_geodesic_recovers_clusters_and_loops():
    """Redesigned _count_betti (kNN-geodesic, Naitzat et al.) validated
    against ground truth: beta_0 recovers the cluster count and beta_1
    detects loops, where the old median-Euclidean threshold collapsed
    beta_0 to ~1. Deterministic (seeded)."""
    import numpy as np
    from blme.tasks.topology.betti_curve import _count_betti

    rng = np.random.default_rng(0)

    # K = 4 well-separated blobs in 10D -> beta_0 = 4, no loops.
    centers = rng.standard_normal((4, 10)) * 30
    X = np.vstack([centers[i] + rng.standard_normal((12, 10)) for i in range(4)])
    b0, b1 = _count_betti(X, maxdim=1, n_neighbors=5)
    assert b0 == 4, f"beta_0 should recover 4 clusters, got {b0}"
    assert b1 == 0

    # Noisy circle -> one connected component, one loop.
    th = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    circ = np.c_[np.cos(th), np.sin(th)] + 0.03 * rng.standard_normal((40, 2))
    b0c, b1c = _count_betti(circ, maxdim=1, n_neighbors=4)
    assert (b0c, b1c) == (1, 1), f"circle should be (1,1), got {(b0c, b1c)}"

    # High-dim gaussian noise -> no spurious persistent loops.
    _, b1n = _count_betti(rng.standard_normal((40, 10)), maxdim=1, n_neighbors=6)
    assert b1n == 0

    # Degenerate guard: < 3 points returns without error.
    assert _count_betti(np.zeros((2, 5)), maxdim=1) == (2, 0)


def test_persistence_entropy(mock_model, mock_tokenizer):
    """Persistence entropy at early, middle, and late layers. The normalized
    (relative) entropy keys must equal PE / log2(m) with m the number of
    finite bars — the [0, 1] scale-free form stated in the paper appendix."""
    from blme.tasks.topology.persistence_entropy import PersistenceEntropyTask

    task = PersistenceEntropyTask(config={"num_samples": 5})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # May error if get_layers returns None for this architecture
    if "error" not in results:
        assert "pe_simplification_ratio" in results
        assert any("pe_h0" in k for k in results.keys())
        for key in list(results):
            if not key.endswith("_pe_h0_norm"):
                continue
            prefix = key[: -len("_pe_h0_norm")]
            m = results[f"{prefix}_num_features_h0"]
            pe = results[f"{prefix}_pe_h0"]
            expected = pe / np.log2(m) if m >= 2 else 0.0
            assert results[key] == pytest.approx(expected, abs=1e-12)
            assert 0.0 <= results[key] <= 1.0 + 1e-9


def test_persistent_homology_normalized_summaries(mock_model, mock_tokenizer):
    """The scale-normalized persistence keys must equal raw / mean-pairwise-
    distance (the cross-model normalization stated in the paper appendix)."""
    from blme.tasks.topology.homology import PersistentHomologyTask

    task = PersistentHomologyTask(config={"num_samples": 5})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)
    if "error" in results:
        pytest.skip("architecture not supported by get_layers")

    norm_keys = [k for k in results if k.endswith("_mean_persistence_h0_norm")]
    assert norm_keys, "normalized persistence keys missing"
    for key in norm_keys:
        prefix = key[: -len("_mean_persistence_h0_norm")]
        mpd = results[f"{prefix}_mean_pairwise_distance"]
        assert mpd > 0
        for stat in ("mean_persistence_h0", "max_persistence_h0",
                     "mean_persistence_h1"):
            assert results[f"{prefix}_{stat}_norm"] == pytest.approx(
                results[f"{prefix}_{stat}"] / mpd, rel=1e-9
            )


def test_zigzag_duplicate_points_do_not_inflate_components():
    """Audit 2026-07-16: _layer_topology lacked the zero-distance edge floor
    (betti_curve.py had it since campaign 3) — duplicate points produced
    explicit-zero csr entries that csgraph treats as NO edge, inflating
    beta_0 (8 dups + 5 far -> 9 instead of 2; e2e: a corpus with one line
    repeated 6x gave h0_mean_components 7.0 vs 1.0)."""
    import numpy as np
    from blme.tasks.topology.zigzag import _layer_topology

    rng = np.random.default_rng(0)
    dup = np.tile(rng.normal(size=(1, 8)), (8, 1))          # 8 identical points
    far = rng.normal(size=(10, 8)) + 100.0                   # distinct cluster
    # (>k points per cluster so kNN edges stay within clusters)
    X = np.vstack([dup, far])
    betti0, _ = _layer_topology(X, n_neighbors=5)
    assert betti0 == 2, f"duplicates inflated beta_0: {betti0}"


def test_scale_stratified_boundary_and_infinite_bars():
    """Audit 2026-07-16 anchors: death EXACTLY at frac*scale is COARSE
    (strict <), and infinite-death bars are excluded entirely."""
    import numpy as np
    from blme.tasks.topology.homology import _scale_stratified_summary

    scale = 2.0   # threshold = 0.5 * 2.0 = 1.0
    dgms = [
        np.array([[0.0, 0.5],          # fine (death < 1.0)
                  [0.0, 1.0],          # boundary -> coarse (strict <)
                  [0.0, 1.5],          # coarse
                  [0.0, np.inf]]),     # essential bar -> excluded
        np.array([[0.2, 0.9]]),        # h1 fine
    ]
    out = _scale_stratified_summary(dgms, scale, frac=0.5)
    assert out["h0_fine_count"] == 1
    assert out["h0_coarse_count"] == 2          # boundary bar counted coarse
    assert out["h1_fine_count"] == 1 and out["h1_coarse_count"] == 0
    total = 0.5 + 1.0 + 1.5
    assert out["h0_fine_persistence_frac"] == pytest.approx(0.5 / total)
