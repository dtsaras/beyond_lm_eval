"""geometry_hubness: near-duplicate rows are counted, kept by default, and the
other choice is recorded (docs/validation/decisions.yaml, geometry_hubness)."""
import numpy as np
import torch

from blme.tasks.geometry.hubness import GlobalHubnessTask, _hubness_stats_from_occurrences, _k_occurrences, _unit_rows

K = [5, 10]


def _table(n_twins=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    E = torch.randn(300, 16, generator=g, dtype=torch.float64)
    if n_twins:
        base = torch.randn(1, 16, generator=g, dtype=torch.float64)
        # Untrained-row pattern: one direction, tiny perturbations (1 - cos ~ 1e-8).
        twins = base + 1e-5 * torch.randn(n_twins, 16, generator=g, dtype=torch.float64)
        E = torch.cat([E, twins])
    return E


def _direct(E):
    U, *_ = _unit_rows(E, torch.device("cpu"))
    occ = _k_occurrences(U, K, 64)
    out = {}
    for k in K:
        out.update(_hubness_stats_from_occurrences(occ[k], k))
    return out


def test_no_twins_leaves_the_headline_and_adds_no_alternative():
    E = _table()
    res = GlobalHubnessTask._score(E, K, 64, torch.device("cpu"))
    assert res["_meta_n_near_duplicate_rows"] == 0
    assert not any(k.startswith("_meta_without_near_duplicates_") for k in res)
    for key, v in _direct(E).items():
        assert res[key] == v


def test_twins_are_kept_by_default_and_the_dedup_value_is_recorded():
    E = _table(n_twins=25)
    res = GlobalHubnessTask._score(E, K, 64, torch.device("cpu"))
    assert res["_meta_n_near_duplicate_rows"] == 25
    assert res["_meta_near_duplicates_dropped"] is False
    assert res["_meta_n_rows_ranked"] == 325
    full, dedup = _direct(E), _direct(E[:300])
    for key in full:
        assert res[key] == full[key]
        assert res[f"_meta_without_near_duplicates_{key}"] == dedup[key]
    # The twins are mutual neighbours: ranking them changes the counts.
    assert res["hubness_k10_skew"] != res["_meta_without_near_duplicates_hubness_k10_skew"]


def test_dropping_twins_swaps_headline_and_record():
    E = _table(n_twins=25)
    res = GlobalHubnessTask._score(E, K, 64, torch.device("cpu"), drop_near_duplicates=True)
    assert res["_meta_near_duplicates_dropped"] is True
    assert res["_meta_n_rows_ranked"] == 300
    full, dedup = _direct(E), _direct(E[:300])
    for key in full:
        assert res[key] == dedup[key]
        assert res[f"_meta_with_near_duplicates_{key}"] == full[key]


def test_gap_is_one_minus_cosine():
    E = _table(n_twins=25)
    U, *_ = _unit_rows(E, torch.device("cpu"))
    _, nn = _k_occurrences(U, K, 64, return_nn=True)
    S = (U @ U.T).numpy()
    np.fill_diagonal(S, -2)
    np.testing.assert_allclose(nn * nn / 2, 1 - S.max(1), rtol=1e-6, atol=1e-12)


def test_nearest_other_gap_matches_the_ranking_pass():
    from blme.tasks.geometry.hubness import nearest_other_gap
    E = _table(n_twins=25)
    U, *_ = _unit_rows(E, torch.device("cpu"))
    _, nn = _k_occurrences(U, K, 64, return_nn=True)
    np.testing.assert_allclose(nearest_other_gap(E, batch_size=37), nn * nn / 2, rtol=1e-12, atol=0)


def test_superposition_drops_injected_twins_only_when_asked():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))
    import _family_d as fd
    from blme.tasks.geometry.superposition_strength import SuperpositionStrengthTask
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    W = model.get_output_embeddings().weight
    with torch.no_grad():
        W[10:30] = W[5] + 1e-6 * torch.randn_like(W[10:30])
    off = SuperpositionStrengthTask({}).evaluate(model, tok, None)
    on = SuperpositionStrengthTask({"drop_near_duplicates": True}).evaluate(model, tok, None)
    assert off["_meta_near_duplicates_dropped"] is False and off["_meta_n_near_duplicate_rows"] is None
    assert on["_meta_n_near_duplicate_rows"] == 21
    assert on["superposition_ratio"] != off["superposition_ratio"]
