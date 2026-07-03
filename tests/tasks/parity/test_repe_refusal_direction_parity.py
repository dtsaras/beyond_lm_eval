"""Numeric-parity test: BLME `repe_refusal_direction` kernel vs the OFFICIAL
refusal-direction difference-of-means (Arditi et al. 2024, "Refusal in Language
Models Is Mediated by a Single Direction", arXiv:2406.11717).

BLME kernel (src/blme/tasks/representation_engineering.py):
    line 551  mu_h = X[y == 1].mean(axis=0)          # harmful
    line 552  mu_n = X[y == 0].mean(axis=0)          # harmless
    line 553  full_direction = mu_h - mu_n           <-- the refusal direction
Held-out separability (558-583): per-fold direction = train_h.mean - train_n.mean,
project test activations, roc_auc.

OFFICIAL reference kernel (refusal_direction, github.com/andyrdt/refusal_direction,
commit 9d852fae1a9121c78b29142de733cb1340770cc3):
    pipeline/submodules/generate_directions.py
        line 43  mean_activations_harmful  = get_mean_activations(harmful ...)
        line 44  mean_activations_harmless = get_mean_activations(harmless ...)
        line 46  mean_diff = mean_activations_harmful - mean_activations_harmless
    (get_mean_activations accumulates (1/n) * sum of last-token activations,
     i.e. the arithmetic mean of the class's activations.)

The refusal direction IS the difference of class means -> BLME `full_direction`
(mu_h - mu_n) is IDENTICAL to the reference `mean_diff`. The fixture embeds
synthetic labelled activations (two Gaussians shifted along a KNOWN d*) and the
reference direction; this test recomputes BLME's kernel and asserts EXACT parity
(<1e-9), |cos|==1, perfect held-out separability, and the d* anchor.

VERDICT: PARITY (exact).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures" / "reference_parity" / "parity" / "repe_refusal_direction.json"
)

TOL = 1e-9


@pytest.fixture(scope="module")
def fx():
    with open(_FIXTURE) as f:
        return json.load(f)


def _blme_refusal_direction(harmful, harmless):
    """BLME full-layer refusal direction, verbatim (representation_engineering.py:551-553)."""
    X = np.concatenate([harmful, harmless], axis=0)
    y = np.concatenate([np.ones(len(harmful), int), np.zeros(len(harmless), int)])
    mu_h = X[y == 1].mean(axis=0)
    mu_n = X[y == 0].mean(axis=0)
    return mu_h - mu_n


def _blme_heldout_auc(harmful, harmless, seed=42):
    """BLME held-out projection separability, verbatim (representation_engineering.py:558-583)."""
    X = np.concatenate([harmful, harmless], axis=0)
    y = np.concatenate([np.ones(len(harmful), int), np.zeros(len(harmless), int)])
    n_splits = min(3, int(min(len(harmful), len(harmless))))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs, fold_gaps = [], []
    for tr, te in cv.split(X, y):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        d = Xtr[ytr == 1].mean(axis=0) - Xtr[ytr == 0].mean(axis=0)
        unit = d / np.linalg.norm(d)
        scores = Xte @ unit
        fold_aucs.append(float(roc_auc_score(yte, scores)))
        fold_gaps.append(float(scores[yte == 1].mean() - scores[yte == 0].mean()))
    return float(np.mean(fold_aucs)), float(np.mean(fold_gaps))


def _ref_mean_diff(harmful, harmless):
    """OFFICIAL refusal_direction get_mean_diff kernel (generate_directions.py:46)."""
    return np.asarray(harmful).mean(axis=0) - np.asarray(harmless).mean(axis=0)


def _cosabs(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return abs(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))


def test_fixture_provenance(fx):
    assert "get_mean_diff" in fx["reference"]
    assert fx["refusal_repo_commit"] == "9d852fae1a9121c78b29142de733cb1340770cc3"
    assert fx["verdict"] == "PARITY"


def test_refusal_direction_exact_parity(fx):
    """BLME (mu_h - mu_n) == refusal_direction get_mean_diff EXACTLY (<1e-9)."""
    harmful = np.array(fx["harmful"], np.float64)
    harmless = np.array(fx["harmless"], np.float64)
    ref = np.array(fx["ref_direction"], np.float64)

    blme = _blme_refusal_direction(harmful, harmless)
    # cross-check the reference kernel reproduces the fixture-stored direction too
    ref_recomputed = _ref_mean_diff(harmful, harmless)
    assert np.max(np.abs(ref_recomputed - ref)) <= TOL

    max_abs_diff = float(np.max(np.abs(blme - ref)))
    assert max_abs_diff <= TOL, f"max|BLME-REF|={max_abs_diff} > {TOL}"
    assert _cosabs(blme, ref) >= 1.0 - 1e-12


def test_heldout_separability_perfect(fx):
    """Along the refusal direction the two classes are perfectly separable."""
    harmful = np.array(fx["harmful"], np.float64)
    harmless = np.array(fx["harmless"], np.float64)
    auc, gap = _blme_heldout_auc(harmful, harmless)
    assert auc >= 0.99, f"held-out AUC={auc} unexpectedly low"
    assert gap > 0.0
    assert abs(auc - fx["heldout_separability_auc"]) <= 1e-9


def test_dstar_anchor(fx):
    """The refusal direction is aligned with the KNOWN separating direction d*."""
    harmful = np.array(fx["harmful"], np.float64)
    harmless = np.array(fx["harmless"], np.float64)
    dstar = np.array(fx["dstar"], np.float64)
    blme = _blme_refusal_direction(harmful, harmless)
    cos = _cosabs(blme, dstar)
    assert cos > 0.9, f"|cos(refusal_dir, d*)|={cos} too low"
    assert abs(cos - fx["abs_cos_blme_dstar"]) <= 1e-9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
