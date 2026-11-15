"""geometry_contextualization: the MEV baselines and their corrections.

kawine/contextual analyze.py scores a word's MEV with sklearn
``PCA(n_components=1).explained_variance_ratio_[0]`` and the random-word
baseline with ``TruncatedSVD(n_components=100).explained_variance_ratio_[0]``.
Both baselines draw rows without replacement, so a draw as large as the pool
is a permutation of it and the baseline equals the sklearn ratio of the whole
pool: size-matched with every occurrence count equal to the pool size, and
the literal 1K reading on a pool of at most 1000 rows. evaluate() then
reports each correction as MEV minus its baseline per layer, averaged over
the reported layers.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from blme.tasks.geometry.contextualization import (
    _baseline_mev, _max_explainable_variance, _max_explainable_variance_truncsvd)

decomp = pytest.importorskip("sklearn.decomposition")


def _pool(seed, n=300, d=128):
    g = np.random.default_rng(seed)
    return g.standard_normal((n, d)) * np.linspace(0.5, 3.0, d) + 2.0 * g.standard_normal(d)


def _truncsvd_ratio(X):
    # arpack: the exact leading component of the official n_components=100 fit.
    return float(decomp.TruncatedSVD(n_components=100, algorithm="arpack", random_state=0)
                 .fit(X).explained_variance_ratio_[0])


def _pca_ratio(X):
    return float(decomp.PCA(n_components=1).fit(X).explained_variance_ratio_[0])


@pytest.mark.parametrize("seed", [0, 1])
def test_truncsvd_mev_is_the_official_truncated_svd_ratio(seed):
    X = _pool(seed)
    assert _max_explainable_variance_truncsvd(X) == pytest.approx(_truncsvd_ratio(X), rel=1e-9)
    assert abs(_truncsvd_ratio(X) - _pca_ratio(X)) > 1e-3     # the two conventions differ


@pytest.mark.parametrize("seed", [0, 1])
def test_pool1k_baseline_is_the_truncated_svd_ratio_of_the_pool(seed):
    X = _pool(seed)
    got = _baseline_mev(X, occ_counts=[5, 9], size_matched=False)
    assert got == pytest.approx(_truncsvd_ratio(X), rel=1e-9)


@pytest.mark.parametrize("seed", [0, 1])
def test_size_matched_baseline_is_the_pca_ratio_of_a_full_draw(seed):
    X = _pool(seed, n=60, d=16)
    got = _baseline_mev(X, occ_counts=[60, 60, 60], size_matched=True)
    assert got == pytest.approx(_pca_ratio(X), rel=1e-9)
    assert _max_explainable_variance(X) == pytest.approx(_pca_ratio(X), rel=1e-9)


def test_evaluate_reports_corrections_as_layer_means(monkeypatch):
    """The four baseline/correction keys are the layer means of the per-layer
    values, and each correction is MEV minus its baseline at that layer; the
    pool1k baseline equals the TruncatedSVD ratio of the recorded pool."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "invariance"))
    import _family_d as fd
    import blme.tasks.geometry.contextualization as C

    pools = []
    real = C._baseline_mev

    def spy(all_vectors, occ_counts=None, n_draws=200, seed=2, size_matched=True):
        v = real(all_vectors, occ_counts, n_draws=n_draws, seed=seed, size_matched=size_matched)
        if not size_matched:
            pools.append((np.array(all_vectors), v))
        return v

    monkeypatch.setattr(C, "_baseline_mev", spy)
    res = C.ContextualizationTask({"num_samples": 50, "min_word_occurrences": 5}).evaluate(
        fd.tiny_gpt2(), fd.tokenizer(), dataset=None)
    per = [res["per_layer"][k] for k in sorted(res["per_layer"])]
    assert len(per) >= 2
    for key, parts in (("mev_corrected_sizematched", ("mev", "mev_baseline")),
                       ("mev_corrected_pool1k", ("mev", "mev_baseline_pool1k"))):
        for p in per:
            assert p[key] == pytest.approx(p[parts[0]] - p[parts[1]], rel=1e-12, abs=1e-15)
    for key in ("mev_baseline", "mev_baseline_pool1k", "mev_corrected_sizematched",
                "mev_corrected_pool1k"):
        assert res[f"mean_{key}"] == pytest.approx(np.mean([p[key] for p in per]), rel=1e-12), key
    assert pools and all(X.shape[0] <= 1000 for X, _ in pools)
    for X, v in pools:
        n_comp = min(100, X.shape[1] - 1)
        ref = float(decomp.TruncatedSVD(n_components=n_comp, algorithm="arpack", random_state=0)
                    .fit(X.astype(np.float64)).explained_variance_ratio_[0])
        assert v == pytest.approx(ref, rel=1e-6)
