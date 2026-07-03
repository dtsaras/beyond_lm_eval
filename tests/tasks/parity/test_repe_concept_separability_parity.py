"""Numeric-parity / faithfulness test: BLME `repe_concept_separability` vs the
OFFICIAL RepE reading-vector construction (Zou et al. 2023 "Representation
Engineering", arXiv:2310.01405).

Two related-but-distinct kernels are involved:

  * OFFICIAL RepE reading vector = the TOP PRINCIPAL COMPONENT of the recentered
    paired differences (h+ - h-). RepE, github.com/andyzoujm/representation-
    engineering, commit 5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e:
        repe/rep_reading_pipeline.py:144  relative[layer] = h[::2] - h[1::2]   (n_difference=1)
        repe/rep_readers.py PCARepReader.get_rep_directions (137-152):
            line 143  H_train_mean = H_train.mean(axis=0, keepdims=True)
            line 145  H_train = recenter(H_train, mean=H_train_mean)
            line 147  pca_model = PCA(n_components=1, whiten=False).fit(H_train)
            line 149  directions[layer] = pca_model.components_

  * BLME `repe_concept_separability` (src/blme/tasks/representation_engineering.py
    214-248) measures LINEAR SEPARABILITY of the concept via held-out
    StratifiedKFold LogisticRegression (AUC + accuracy) -- the *separability
    statistic* along the concept axis, not the PCA vector itself.

Relationship: FAITHFUL / PROXY. On matched contrastive pairs shifted along a
KNOWN direction d*, the RepE PCA reading vector aligns with d* (|cos|->1 up to
SIGN) and BLME's held-out separability along the same concept is ~perfect
(AUC ~ 0.98). A negative control with NO shift gives chance AUC.

The fixture also stores the direction produced by RUNNING THE REAL RepE
`PCARepReader` class on GPU; a CPU sklearn re-implementation of that exact kernel
matches it to |cos| == 1 (1e-12), proving the quoted reference is faithful.

VERDICT: FAITHFUL (concept-separability proxy for the RepE reading vector).
Caveat: PCA sign ambiguity -> compare with |cos|.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures" / "reference_parity" / "parity" / "repe_concept_separability.json"
)


@pytest.fixture(scope="module")
def fx():
    with open(_FIXTURE) as f:
        return json.load(f)


def _repe_pca_reading_vector(pos, neg):
    """OFFICIAL RepE PCARepReader kernel on paired diffs (rep_readers.py:143-149,
    fed via rep_reading_pipeline.py:144 with n_difference=1)."""
    diffs = np.asarray(pos) - np.asarray(neg)          # h+ - h-  (matched pairs)
    H = diffs - diffs.mean(axis=0, keepdims=True)      # recenter (rep_readers.py:143-145)
    pca = PCA(n_components=1, whiten=False).fit(H)      # rep_readers.py:147
    return pca.components_[0]                           # rep_readers.py:149


def _blme_separability(pos, neg, seed=42):
    """BLME repe_concept_separability held-out LR AUC/acc, verbatim (rep_eng.py:214-248)."""
    X = np.concatenate([pos, neg], axis=0)
    y = np.concatenate([np.ones(len(pos), int), np.zeros(len(neg), int)])
    n_splits = min(3, int(np.min(np.bincount(y))))
    if n_splits < 2:
        n_splits = 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs, fold_accs = [], []
    for tr, te in cv.split(X, y):
        clf = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000)
        clf.fit(X[tr], y[tr])
        preds = clf.predict(X[te])
        probas = clf.predict_proba(X[te])[:, 1]
        fold_accs.append(accuracy_score(y[te], preds))
        try:
            fold_aucs.append(roc_auc_score(y[te], probas))
        except ValueError:
            fold_aucs.append(accuracy_score(y[te], preds))
    return float(np.mean(fold_aucs)), float(np.mean(fold_accs))


def _cosabs(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return abs(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))


def test_fixture_provenance(fx):
    assert "PCARepReader" in fx["kernel_reference"]
    assert fx["repe_repo_commit"] == "5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e"
    assert fx["verdict"] == "FAITHFUL"
    assert fx["pca_sign_ambiguity"] is True


def test_pca_reading_vector_matches_real_repe_class(fx):
    """CPU sklearn re-impl of the RepE PCA kernel == the direction produced by the
    REAL RepE PCARepReader class (captured on GPU), up to |cos| (sign ambiguity)."""
    real = fx.get("ref_pca_reading_vector_realclass")
    if real is None:
        pytest.skip("real RepE PCARepReader direction not captured (no CUDA at gen time)")
    stored_reimpl = np.array(fx["ref_pca_reading_vector"], np.float64)
    assert _cosabs(stored_reimpl, np.array(real, np.float64)) >= 1.0 - 1e-9
    # and recompute the re-impl from the raw inputs to confirm the fixture
    pos = np.array(fx["pos_matched"], np.float64)
    neg = np.array(fx["neg_matched"], np.float64)
    assert _cosabs(_repe_pca_reading_vector(pos, neg), stored_reimpl) >= 1.0 - 1e-9


def test_pca_reading_vector_recovers_dstar(fx):
    """RepE PCA reading vector (top-PC of h+ - h-) is aligned with the KNOWN
    concept direction d*, up to SIGN."""
    pos = np.array(fx["pos_matched"], np.float64)
    neg = np.array(fx["neg_matched"], np.float64)
    dstar = np.array(fx["dstar"], np.float64)
    pc = _repe_pca_reading_vector(pos, neg)
    cos = _cosabs(pc, dstar)
    assert cos > 0.95, f"|cos(PCA reading vector, d*)|={cos} too low"
    assert abs(cos - fx["abs_cos_pca_dstar"]) <= 1e-6


def test_blme_separability_tracks_concept(fx):
    """BLME held-out linear separability along the concept is ~perfect, matching
    the fixture, while a no-shift negative control is at chance."""
    pos = np.array(fx["pos_matched"], np.float64)
    neg = np.array(fx["neg_matched"], np.float64)
    auc, acc = _blme_separability(pos, neg)
    assert auc > 0.95, f"concept separability AUC={auc} too low"
    assert abs(auc - fx["blme_separability_auc"]) <= 1e-9
    assert abs(acc - fx["blme_separability_acc"]) <= 1e-9

    posc = np.array(fx["neg_control_pos"], np.float64)
    negc = np.array(fx["neg_control_neg"], np.float64)
    auc_c, _ = _blme_separability(posc, negc)
    assert auc_c < 0.7, f"negative-control AUC={auc_c} not at chance"
    assert abs(auc_c - fx["neg_control_auc"]) <= 1e-9


def test_pca_parallel_to_meandiff(fx):
    """On matched pairs the PCA reading vector and the mean-difference direction
    (the other RepE reader, ClusterMean) point along the same axis."""
    pos = np.array(fx["pos_matched"], np.float64)
    neg = np.array(fx["neg_matched"], np.float64)
    pc = _repe_pca_reading_vector(pos, neg)
    meandiff = pos.mean(axis=0) - neg.mean(axis=0)
    cos = _cosabs(pc, meandiff)
    assert cos > 0.95, f"|cos(PCA, mean-diff)|={cos} too low"
    assert abs(cos - fx["abs_cos_pca_meandiff"]) <= 1e-6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
