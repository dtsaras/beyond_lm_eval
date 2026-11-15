"""geometry_mahalanobis: the pooled headline keys through evaluate().

The mean-pooled ID and OOD representations are fed in place of the forward
pass (as tests/tasks/test_mahalanobis_task.py does), and every split's fit
and scored rows are recorded on the way to BLME's own helpers. The reference
refits each split with sklearn's LedoitWolf, scores the recorded rows by
sqrt((x - mu)^T P (x - mu)), pools the distances over the 5 splits, and
derives the seven headline keys from them.
"""
import numpy as np
import pytest
import torch

import blme.tasks.geometry.mahalanobis as M

LedoitWolf = pytest.importorskip("sklearn.covariance").LedoitWolf
N, D = 40, 12


def _reference_distances(X_fit, X):
    lw = LedoitWolf().fit(X_fit)
    diff = X - X_fit.mean(axis=0)
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, lw.precision_, diff))


def test_pooled_keys_match_a_ledoit_wolf_refit(monkeypatch):
    g = np.random.default_rng(0)
    A = g.standard_normal((D, D)) / np.sqrt(D)
    X_id = g.standard_normal((N, D)) @ A
    X_ood = X_id[g.permutation(N)] * 1.3 + 0.4 * g.standard_normal((N, D))
    feeds = iter([X_id, X_ood])
    monkeypatch.setattr(M, "_collect_mean_pooled_hidden_states",
                        lambda *a, **k: torch.from_numpy(next(feeds)))
    fits, scored = [], []
    real_fit, real_score = M._fit_gaussian, M._mahalanobis_scores
    monkeypatch.setattr(M, "_fit_gaussian",
                        lambda X, diag=None: fits.append(np.array(X)) or real_fit(X, diag))
    monkeypatch.setattr(M, "_mahalanobis_scores",
                        lambda X, mu, P: scored.append(np.array(X)) or real_score(X, mu, P))

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return list(range(10))

        def decode(self, ids):
            return "decoded"

    res = M.MahalanobisOODTask({"num_samples": N, "ood_strategy": "shuffle",
                                "id_corpus": "dataset"}).evaluate(
        model=None, tokenizer=Tok(), dataset=[{"text": "a"}] * N, cache=None)
    assert "error" not in res, res
    assert res["_meta_n_splits"] == 5 and len(fits) == 5 and len(scored) == 10
    d_id = np.concatenate([_reference_distances(f, s) for f, s in zip(fits, scored[0::2])])
    d_ood = np.concatenate([_reference_distances(f, s) for f, s in zip(fits, scored[1::2])])
    for f, s in zip(fits, scored[0::2]):          # every split holds out its scored rows
        assert not (f[:, None, :] == s[None, :, :]).all(-1).any()
    mid, mood = d_id.mean(), d_ood.mean()
    want = {
        "mean_mahalanobis_id": mid,
        "mean_mahalanobis_ood": mood,
        "ood_separation_gap": mood - mid,
        "ood_separation_ratio": mood / mid,
        "mahalanobis_sq_per_dim_id": np.mean(d_id ** 2) / D,
        "mahalanobis_sq_per_dim_ood": np.mean(d_ood ** 2) / D,
        "ood_separation_gap_per_dim": (np.mean(d_ood ** 2) - np.mean(d_id ** 2)) / D,
    }
    for key, v in want.items():
        assert res[key] == pytest.approx(v, rel=1e-9), key
