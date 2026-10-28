"""topology_zigzag_persistence follows Gardinazzi et al. 2025 / ZigZagLLMs@2d10c07.

The reference extraction is re-typed below from
``src/representation/rep-extraction``: forward hooks on the modules
``get_target_layers(option="norm2")`` names (``*.layers.{i}.post_attention_layernorm``
plus the final norm, helpers.py:5-38), the last token of each prompt
(extract_activations.py:68-72) and prompts tokenised with
``add_special_tokens=False`` (dataset_utils.py:14-22).
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
pytest.importorskip("dionysus")

from blme.tasks.topology.zigzag import ZigzagPersistenceTask, _paper_clouds  # noqa: E402


def _reference_extract(model, tok, texts, names):
    mods = dict(model.named_modules())
    tmp, reps = {}, {n: [] for n in names}
    hooks = [mods[n].register_forward_hook(
        lambda m, i, o, n=n: tmp.__setitem__(n, o)) for n in names]
    try:
        with torch.no_grad():
            for t in texts:
                enc = tok(t.strip(), add_special_tokens=False, return_tensors="pt",
                          max_length=2048, truncation=True)
                model(**enc)
                T = int(enc["input_ids"].shape[1])
                for n in names:
                    reps[n].append(tmp[n][0, T - 1].double())
    finally:
        for h in hooks:
            h.remove()
    return [torch.stack(reps[n]).numpy() for n in names]


def _arch(name):
    if name == "llama":
        m = fd.tiny_llama(n_layers=3)
        names = [f"model.layers.{i}.post_attention_layernorm" for i in range(3)] + ["model.norm"]
    elif name.startswith("neox"):
        m = fd.tiny_neox(n_layers=3, parallel=(name == "neox_parallel"))
        names = ([f"gpt_neox.layers.{i}.post_attention_layernorm" for i in range(3)]
                 + ["gpt_neox.final_layer_norm"])
    else:
        m = fd.tiny_gpt2()
        L = len(m.transformer.h)
        names = [f"transformer.h.{i}.ln_2" for i in range(L)] + ["transformer.ln_f"]
    return m, names


@pytest.mark.parametrize("arch", ["llama", "neox_sequential", "neox_parallel", "gpt2"])
def test_clouds_are_the_reference_last_token_norm2_rows(arch):
    model, names = _arch(arch)
    tok = fd.tokenizer()
    clouds, _, info = _paper_clouds(model, tok, fd.TEXTS, window=2048)
    ref = _reference_extract(model, tok, fd.TEXTS, names)
    assert len(clouds) == len(names) == info["n_sites"]
    for a, b in zip(clouds, ref):
        np.testing.assert_array_equal(a, b)
    assert info["sites_are_norm_outputs"] is True


def test_task_reads_the_paper_sites_not_block_outputs():
    """Before the fix the task mean-pooled the L block outputs (with BOS kept);
    the paper reads L + 1 clouds: each post-attention norm and the final norm."""
    model = fd.tiny_llama(n_layers=3)
    tok = fd.tokenizer()
    res = ZigzagPersistenceTask({"num_samples": len(fd.TEXTS), "n_neighbors": 4}).evaluate(
        model, tok, fd.dataset())
    assert "error" not in res, res
    assert res["_meta_zz_n_clouds"] == 3 + 1
    assert res["_meta_zz_sites_are_norm_outputs"] is True
    assert res["_meta_zz_site_norm_module"] == "post_attention_layernorm"
    assert res["_meta_zz_n_docs_starting_with_bos"] == 0
    # The reference's other pooling is kept for comparison only.
    assert "_meta_meanpool_zz_true_adjacent_mean" in res


def test_site_check_flags_a_block_whose_mlp_reads_the_raw_residual():
    """OLMo-2-style blocks normalise sublayer OUTPUTS: the MLP input is no norm
    output, and the derived flag must say so instead of assuming it."""
    model = fd.tiny_llama(n_layers=2)
    for blk in model.model.layers:
        blk.post_attention_layernorm = torch.nn.Identity()
    _, _, info = _paper_clouds(model, fd.tokenizer(), fd.TEXTS[:4], window=2048)
    assert info["sites_are_norm_outputs"] is False
    assert info["site_norm_module"] is None


def test_default_neighbourhood_is_the_papers_k4():
    """k_NN = 4 (Fig. 2, Sec. 4.2); the task shipped k = 5 (the README example)."""
    from blme.tasks.config_loader import resolve_task_config
    model, tok = fd.tiny_llama(n_layers=3), fd.tokenizer()
    cfg = resolve_task_config("topology_zigzag_persistence", {"num_samples": len(fd.TEXTS)})
    default = ZigzagPersistenceTask(cfg).evaluate(model, tok, fd.dataset())
    k4 = ZigzagPersistenceTask({**cfg, "n_neighbors": 4}).evaluate(model, tok, fd.dataset())
    k5 = ZigzagPersistenceTask({**cfg, "n_neighbors": 5}).evaluate(model, tok, fd.dataset())
    assert default["_meta_zz_knn"] == 4
    assert default["zz_true_adjacent_mean"] == k4["zz_true_adjacent_mean"]
    assert default["_meta_zz_true_n_bars"] != k5["_meta_zz_true_n_bars"]


def test_missing_dionysus_is_an_error_not_a_fallback(monkeypatch):
    """The paper's barcode has no substitute: without dionysus the task used to
    return the GF(2)/band proxies with the zz_true_* keys silently absent."""
    monkeypatch.setitem(sys.modules, "dionysus", None)       # import raises ImportError
    res = ZigzagPersistenceTask({"num_samples": len(fd.TEXTS)}).evaluate(
        fd.tiny_llama(n_layers=3), fd.tokenizer(), fd.dataset())
    assert set(res) == {"error"} and "dionysus" in res["error"]


def test_reference_sample_size_is_the_default():
    """Sec. 4.1: 16 subsets of 500 prompts (the study ran one subset of 40)."""
    from blme.tasks.config_loader import resolve_task_config
    cfg = resolve_task_config("topology_zigzag_persistence")
    assert (cfg["subset_size"], cfg["num_subsets"]) == (500, 16)
    assert cfg["num_samples"] >= 500 * 16


def test_draw_subsets_blocks_draws_and_single_pool():
    from blme.tasks.topology.zigzag import _draw_subsets, _subset_meta
    blocks = _draw_subsets(40, 10, 4, seed=0)
    assert [list(b) for b in blocks] == [list(range(i * 10, i * 10 + 10)) for i in range(4)]
    assert _subset_meta(blocks, 40)["_meta_zz_subsets_disjoint"] is True
    draws = _draw_subsets(25, 10, 4, seed=0)
    assert len(draws) == 4 and all(len(set(d)) == 10 and max(d) < 25 for d in draws)
    meta = _subset_meta(draws, 25)
    assert meta["_meta_zz_subsets_disjoint"] is False and 0 < meta["_meta_zz_subset_overlap"] < 1
    assert [list(d) for d in _draw_subsets(7, 10, 4, seed=0)] == [list(range(7))]


def test_descriptors_are_means_over_subsets():
    """Before the fix one subset of every prompt was scored; now each subset is
    scored and the descriptors are averaged, with the overlap reported."""
    from blme.tasks.topology.zigzag import _draw_subsets, _zigzag_paper_metric
    model, tok = fd.tiny_llama(n_layers=3), fd.tokenizer()
    cfg = {"num_samples": len(fd.TEXTS), "subset_size": 10, "num_subsets": 3,
           "n_neighbors": 4, "seed": 7}
    res = ZigzagPersistenceTask(cfg).evaluate(model, tok, fd.dataset())
    assert res["_meta_zz_n_subsets"] == 3 and res["_meta_zz_subset_size"] == 10
    assert res["_meta_zz_pool_size"] == len(fd.TEXTS) and res["_meta_zz_subset_overlap"] > 0
    clouds, _, _ = _paper_clouds(model, tok, fd.TEXTS, window=512)
    per = [_zigzag_paper_metric([c[idx] for c in clouds], n_neighbors=4, max_points=10)
           for idx in _draw_subsets(len(fd.TEXTS), 10, 3, seed=7)]
    want = float(np.mean([p["zz_true_adjacent_mean"] for p in per]))
    assert res["zz_true_adjacent_mean"] == pytest.approx(want, rel=0, abs=1e-15)
    assert res["_meta_zz_true_adjacent_mean_subset_sd"] == pytest.approx(
        float(np.std([p["zz_true_adjacent_mean"] for p in per])), abs=1e-15)


_PROXY_KEYS = ("h1_total_layer_persistence", "h1_loop_layer_sum", "h1_num_bands",
               "h1_max_band_length", "h1_first_layer", "h1_last_layer",
               "betti1_fraction_layers", "h0_mean_components", "h0_persistence_range",
               "h0_first_layer_components", "h0_last_layer_components", "zz_h1_turnover_rate")


def test_proxies_and_constant_barcode_summaries_are_meta():
    """The band/GF(2) proxies have no counterpart in the paper; zz_true_full_depth
    and _adjacent_min were 0.0 on every model the dossier ran, and n_bars counts
    N. They were feature columns before the fix."""
    res = ZigzagPersistenceTask({"num_samples": len(fd.TEXTS)}).evaluate(
        fd.tiny_llama(n_layers=3), fd.tokenizer(), fd.dataset())
    feats = {k for k in res if not k.startswith("_meta_")}
    assert feats == {"zz_true_adjacent_mean", "zz_true_mean_bar_len"}, feats
    for k in _PROXY_KEYS + ("zz_true_full_depth", "zz_true_adjacent_min", "zz_true_n_bars"):
        assert f"_meta_{k}" in res, k


def _reference_inter_layer_persistence(bars, n, p):
    """plots_main_text.ipynb cells 6-7 re-typed on an effective barcode:
    find_ph_sim / find_betti_layers, row-normalised, |i-j|^p weighted."""
    betti = np.zeros(n)
    for b, d in bars:
        for x in range(int(b), int(d)):
            betti[x] += 1
    psim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            lo, hi = min(i, j), max(i, j)
            psim[i, j] = sum(1 for b, d in bars if b <= lo and d > hi)
    lp = np.array([psim[i] / betti[i] if betti[i] else np.zeros(n) for i in range(n)])
    w = np.array([[1e-10 if (p < 0 and i == j) else abs(i - j) ** p for j in range(n)]
                  for i in range(n)])
    return np.sum(lp * w, axis=1) / np.sum(w, axis=1)


def test_per_layer_curves_follow_the_reference_notebook():
    from blme.tasks.topology.zigzag import ZZ_ALPHAS, _alpha_tag, _bar_curves, _effective_bars
    model, tok = fd.tiny_llama(n_layers=3), fd.tokenizer()
    clouds, _, _ = _paper_clouds(model, tok, fd.TEXTS, window=512)
    bars, L = _effective_bars(clouds, 4)
    assert bars and L == 4
    curves = _bar_curves(bars, L)
    for a in ZZ_ALPHAS:
        np.testing.assert_allclose(curves[f"inter_layer_persistence_alpha_{_alpha_tag(a)}"],
                                   _reference_inter_layer_persistence(bars, L, a),
                                   rtol=1e-12, atol=1e-15)
        B = curves[f"births_rel_freq_alpha_{_alpha_tag(a)}"]
        assert B.shape == (L,) and abs(B.sum() - 1.0) < 1e-12
    res = ZigzagPersistenceTask({"num_samples": len(fd.TEXTS)}).evaluate(model, tok, fd.dataset())
    assert len(res["_meta_zz_betti1_curve"]) == L
    assert len(res["_meta_zz_inter_layer_persistence_alpha_2"]) == L


def test_a_barcode_failure_propagates(monkeypatch):
    import blme.tasks.topology.zigzag as zz

    def boom(*a, **k):
        raise RuntimeError("zigzag engine failed")
    monkeypatch.setattr(zz, "_true_zigzag_barcode", boom)
    with pytest.raises(RuntimeError, match="zigzag engine failed"):
        ZigzagPersistenceTask({"num_samples": len(fd.TEXTS)}).evaluate(
            fd.tiny_llama(n_layers=3), fd.tokenizer(), fd.dataset())
