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

  * BLME `repe_concept_separability` (src/blme/tasks/repe/concept_separability.py):
    the headline (``lat_test_acc``, ``lat_layer_frac``) is the RepE LAT reading
    itself -- ``_pca_reader`` is PCARepReader on one layer (direction, recentring
    mean, sign from the training labels) and ``_choice_accuracy`` the pairwise
    evaluation. The held-out logistic-regression AUC / accuracy
    (``_layer_cv``) is the legacy estimator, ``_meta_`` only.

Both shipped kernels are called here on the fixture: matched contrastive pairs
shifted along a KNOWN direction d*, plus a no-shift negative control. The
fixture also stores the direction produced by RUNNING THE REAL RepE
`PCARepReader` class on GPU; the shipped ``_pca_reader`` must return that
direction (|cos| == 1 up to float32), with the sign that puts the positive
item above the negative one. The released classes are run live, layer
selection included, in test_repe_concept_separability_lat_parity.py.

Caveat: PCA sign ambiguity -> directions are compared with |cos|; the sign is
tested through the reader's own sign rule.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from sklearn.decomposition import PCA

from blme.tasks.repe.concept_separability import (
    ConceptSeparabilityTask, _choice_accuracy, _pca_reader,
)

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


def _shipped_reader(pos, neg):
    """The shipped PCARepReader kernel on pairs laid out as the task lays them
    out: rows 2i, 2i + 1 are pair i, the positive item first."""
    pos, neg = np.asarray(pos, np.float32), np.asarray(neg, np.float32)
    H = np.empty((2 * len(pos), pos.shape[1]), dtype=np.float32)
    H[0::2], H[1::2] = pos, neg
    return _pca_reader(H, [[1, 0]] * len(pos))


def _blme_separability(pos, neg):
    """The legacy (_meta_) logistic probe, through the shipped ``_layer_cv`` and
    the splitter evaluate() builds for ungrouped caller data."""
    X = np.concatenate([pos, neg], axis=0)
    y = np.concatenate([np.ones(len(pos), int), np.zeros(len(neg), int)])
    n_splits = max(2, min(3, int(np.min(np.bincount(y)))))
    split = ConceptSeparabilityTask._splitter([{} for _ in y], y, False, n_splits)
    return ConceptSeparabilityTask._layer_cv(X, y, n_splits, split)


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


def test_shipped_reader_returns_the_real_repe_direction(fx):
    """``_pca_reader`` (the kernel behind lat_test_acc) == the direction the REAL
    RepE PCARepReader produced, and its sign rule orients it along +d*, the
    side the positive items were shifted to."""
    pos = np.array(fx["pos_matched"], np.float64)
    neg = np.array(fx["neg_matched"], np.float64)
    direction, mean, sign, evr = _shipped_reader(pos, neg)
    real = fx.get("ref_pca_reading_vector_realclass") or fx["ref_pca_reading_vector"]
    assert _cosabs(direction, real) >= 1.0 - 1e-6
    assert _cosabs(direction, fx["dstar"]) == pytest.approx(fx["abs_cos_pca_dstar"], abs=1e-5)
    assert mean.ravel() == pytest.approx((pos - neg).mean(axis=0), rel=1e-5, abs=1e-6)
    assert 0.0 < evr <= 1.0
    shift = (pos - neg).mean(axis=0)
    assert sign in (1.0, -1.0)
    assert sign * float(direction @ shift) > 0.0


def test_shipped_reader_choice_accuracy_on_held_out_pairs(fx):
    """Reader fitted on the first half of the pairs, choice accuracy on the
    second: ~perfect on the shifted pairs, chance on the no-shift control."""
    def held_out(pos, neg):
        pos, neg = np.asarray(pos, np.float32), np.asarray(neg, np.float32)
        h = len(pos) // 2
        direction, mean, sign, _ = _shipped_reader(pos[:h], neg[:h])
        return _choice_accuracy(pos[h:], neg[h:], mean, direction, sign)

    assert held_out(fx["pos_matched"], fx["neg_matched"]) >= 0.95
    assert 0.2 <= held_out(fx["neg_control_pos"], fx["neg_control_neg"]) <= 0.8


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
    """The legacy logistic probe (``_meta_`` AUC / accuracy, shipped ``_layer_cv``)
    is ~perfect along the concept, matching the fixture, while a no-shift
    negative control is at chance."""
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
