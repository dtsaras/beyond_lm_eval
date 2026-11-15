"""repe_refusal_direction: the shipped direction / separability kernels vs the
OFFICIAL refusal-direction difference-of-means (Arditi et al. 2024, "Refusal in
Language Models Is Mediated by a Single Direction", arXiv:2406.11717).

Shipped kernels (src/blme/tasks/repe/refusal.py), called here as evaluate()
composes them:
    ``_ClassSums``            float64 class means per (position, layer);
                              mean_diff = harmful.mean() - harmless.mean()
    ``direction_norm``        torch.linalg.vector_norm(mean_diff[pos, layer])
    ``_layer_separability``   held-out projection AUC / gap (headline
                              ``separability_auc``, ``mean_projection_gap``)
    ``_rel_energy``           headline ``rel_direction_energy``

OFFICIAL reference kernel (refusal_direction, github.com/andyrdt/refusal_direction,
commit 9d852fae1a9121c78b29142de733cb1340770cc3):
    pipeline/submodules/generate_directions.py
        line 43  mean_activations_harmful  = get_mean_activations(harmful ...)
        line 44  mean_activations_harmless = get_mean_activations(harmless ...)
        line 46  mean_diff = mean_activations_harmful - mean_activations_harmless
    (get_mean_activations accumulates (1/n) * sum of last-token activations,
     i.e. the arithmetic mean of the class's activations.)

The fixture embeds synthetic labelled activations (two Gaussians shifted along a
KNOWN d*) and the reference direction. The released pipeline itself, selection
included, is run live in test_repe_refusal_direction_arditi_parity.py.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
from blme.tasks.repe.refusal import _ClassSums, _layer_separability, _rel_energy  # noqa: E402

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures" / "reference_parity" / "parity" / "repe_refusal_direction.json"
)

TOL = 1e-9


@pytest.fixture(scope="module")
def fx():
    with open(_FIXTURE) as f:
        return json.load(f)


def _class_sums(rows):
    """One class at one (position, layer) site, accumulated prompt by prompt."""
    rows = np.asarray(rows, np.float64)
    c = _ClassSums(1, 1, rows.shape[1])
    for r in rows:
        c.add(torch.tensor(r, dtype=torch.float64).reshape(1, 1, -1))
    return c


def _blme_refusal_direction(harmful, harmless):
    """mean_diff at the one site, as evaluate() forms it."""
    return (_class_sums(harmful).mean() - _class_sums(harmless).mean())[0, 0].numpy()


def _blme_heldout(harmful, harmless, seed=42, cv_splits=3):
    """(direction norm, AUC, gap) from the shipped kernel with evaluate()'s arguments."""
    X = np.concatenate([harmful, harmless], axis=0)
    y = np.concatenate([np.ones(len(harmful), int), np.zeros(len(harmless), int)])
    return _layer_separability(X, y, min(cv_splits, len(harmful), len(harmless)), seed)


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
    # direction_norm, by both routes evaluate() has to it.
    want = float(np.linalg.norm(ref))
    assert float(torch.linalg.vector_norm(torch.tensor(blme))) == pytest.approx(want, rel=1e-12)
    assert _blme_heldout(harmful, harmless)[0] == pytest.approx(want, rel=1e-12)


def test_rel_direction_energy_is_the_unbiased_separation(fx):
    """rel_direction_energy = [||ref||^2 - tr S_h / n_h - tr S_n / n_n] / mean ||x||^2,
    with ref the released mean difference and S the class covariances."""
    harmful = np.array(fx["harmful"], np.float64)
    harmless = np.array(fx["harmless"], np.float64)
    ref = np.array(fx["ref_direction"], np.float64).ravel()
    bias = sum(float(np.trace(np.atleast_2d(np.cov(c, rowvar=False)))) / len(c)
               for c in (harmful, harmless))
    denom = float(np.mean(np.sum(np.concatenate([harmful, harmless]) ** 2, axis=1)))
    want = (float(ref @ ref) - bias) / denom
    got = _rel_energy(_class_sums(harmful), _class_sums(harmless), 0, 0)
    assert got == pytest.approx(want, rel=1e-9)
    assert abs(float(ref @ ref) / denom - want) > 1e-4      # not the biased ratio


def test_heldout_separability_perfect(fx):
    """Along the refusal direction the two classes are perfectly separable."""
    harmful = np.array(fx["harmful"], np.float64)
    harmless = np.array(fx["harmless"], np.float64)
    _, auc, gap = _blme_heldout(harmful, harmless)
    assert auc >= 0.99, f"held-out AUC={auc} unexpectedly low"
    assert abs(auc - fx["heldout_separability_auc"]) <= 1e-9
    # The gap is the classes' separation along the unit direction: close to
    # the norm of the released mean difference (each fold refits it).
    assert gap == pytest.approx(float(np.linalg.norm(fx["ref_direction"])), rel=0.15)
    # Shuffled labels: no held-out separation left.
    y_null = np.random.default_rng(0).permutation(
        np.concatenate([np.ones(len(harmful), int), np.zeros(len(harmless), int)]))
    X = np.concatenate([harmful, harmless], axis=0)
    _, auc_null, gap_null = _layer_separability(X, y_null, 3, 42)
    assert auc_null < 0.8 and abs(gap_null) < 0.5 * gap


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
