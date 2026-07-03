"""Numeric-parity test: BLME `repe_task_vectors` reading-vector kernel vs the
OFFICIAL RepE `ClusterMeanRepReader`.

Task: representation-engineering "task / reading vector" (Zou et al. 2023
"Representation Engineering", arXiv:2310.01405; Ilharco et al. 2023 "Editing
Models with Task Arithmetic", arXiv:2212.04089).

BLME kernel (src/blme/tasks/representation_engineering.py):
    line 121  mean_pos = A_pos.mean(dim=0)
    line 122  mean_neg = A_neg.mean(dim=0)
    line 125  v = mean_pos - mean_neg          <-- the reading / task vector

OFFICIAL reference kernel (RepE, github.com/andyzoujm/representation-engineering,
commit 5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e):
    repe/rep_readers.py  class ClusterMeanRepReader.get_rep_directions (196-216)
        line 212  H_pos_mean = H_train[pos_class].mean(axis=0, keepdims=True)
        line 213  H_neg_mean = H_train[neg_class].mean(axis=0, keepdims=True)
        line 215  directions[layer] = H_pos_mean - H_neg_mean

These are the SAME closed-form direction (mean of positives - mean of negatives).
The fixture embeds synthetic paired activations (two Gaussians shifted along a
KNOWN unit direction d*) and the direction captured by RUNNING the real RepE
ClusterMeanRepReader class. This test recomputes BLME's kernel on the identical
matrices and asserts EXACT parity (<1e-9) and |cos|==1, plus the d* anchor.

VERDICT: PARITY (exact).
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures" / "reference_parity" / "parity" / "repe_task_vectors.json"
)

TOL = 1e-9


@pytest.fixture(scope="module")
def fx():
    with open(_FIXTURE) as f:
        return json.load(f)


def _blme_task_vector_kernel(pos, neg):
    """BLME repe_task_vectors kernel, verbatim (representation_engineering.py:121-125)."""
    A_pos = torch.tensor(pos)
    A_neg = torch.tensor(neg)
    mean_pos = A_pos.mean(dim=0)
    mean_neg = A_neg.mean(dim=0)
    v = mean_pos - mean_neg
    return v.numpy()


def _cosabs(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return abs(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))


def test_fixture_provenance(fx):
    assert fx["reference"].startswith("RepE ClusterMeanRepReader")
    assert fx["repe_repo_commit"] == "5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e"
    assert fx["verdict"] == "PARITY"


def test_task_vector_exact_parity(fx):
    """BLME (mean_pos - mean_neg) == RepE ClusterMeanRepReader EXACTLY (<1e-9)."""
    pos = np.array(fx["pos"], np.float64)
    neg = np.array(fx["neg"], np.float64)
    ref = np.array(fx["ref_direction"], np.float64)

    blme = _blme_task_vector_kernel(pos, neg)
    max_abs_diff = float(np.max(np.abs(blme - ref)))
    assert max_abs_diff <= TOL, f"max|BLME-REF|={max_abs_diff} > {TOL}"
    assert abs(max_abs_diff - fx["exact_max_abs_diff"]) <= 1e-12
    assert _cosabs(blme, ref) >= 1.0 - 1e-12


def test_dstar_anchor(fx):
    """Recovered direction is aligned with the KNOWN separating direction d*."""
    pos = np.array(fx["pos"], np.float64)
    neg = np.array(fx["neg"], np.float64)
    dstar = np.array(fx["dstar"], np.float64)
    blme = _blme_task_vector_kernel(pos, neg)
    cos = _cosabs(blme, dstar)
    assert cos > 0.9, f"|cos(reading_vector, d*)|={cos} too low"
    assert abs(cos - fx["abs_cos_blme_dstar"]) <= 1e-9


def test_sign_flip_symmetry(fx):
    """Swapping pos<->neg flips the sign of the reading vector but not its axis."""
    pos = np.array(fx["pos"], np.float64)
    neg = np.array(fx["neg"], np.float64)
    v = _blme_task_vector_kernel(pos, neg)
    v_swapped = _blme_task_vector_kernel(neg, pos)
    assert np.allclose(v, -v_swapped, atol=TOL)
    assert _cosabs(v, v_swapped) >= 1.0 - 1e-12


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
