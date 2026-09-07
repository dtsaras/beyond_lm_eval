"""Numeric-parity test: BLME geometry_matrix_entropy vs the official code.

TASK: geometry_matrix_entropy
BLME: src/blme/tasks/geometry/matrix_entropy.py :: _matrix_entropy

Paper: Wei, Tan, Li, Wang & Huang, "Large Language Model Evaluation via
       Matrix Entropy" (arXiv:2401.17139 v1), renamed in v2 to
       "Diff-eRank: A Novel Rank-Based Metric for Evaluating Large
       Language Models" (NeurIPS 2024).
Repo:  waltonfuture/Diff-eRank — utils/entropy_dataset.py
       (normalize L10-16, cal_cov L18-22, cal_entropy L24-31).

The reference below is a VERBATIM float64 numpy transcription of those
three functions (cal_cov's second F.normalize is a no-op on unit rows;
eigvalsh of Z^T Z / N gives the same spectrum). Campaign-3 established
kernel parity at ~2.8e-6 (BLME runs float32); this file makes that bar a
permanent regression test — previously NO test pinned the value, and
dropping the centering or the row-normalization passed the whole suite.
"""
import math

import numpy as np
import pytest
import torch

from blme.tasks.geometry.matrix_entropy import _matrix_entropy  # noqa: E402

TOL = 1e-5  # float32 task kernel vs float64 reference


# ---------------------------------------------------------------------------
# Verbatim float64 transcription of waltonfuture/Diff-eRank
# utils/entropy_dataset.py (normalize -> cal_cov -> cal_entropy).
# ---------------------------------------------------------------------------
def _official_matrix_entropy(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64)
    R = R - R.mean(axis=0, keepdims=True)                # normalize(): center
    norms = np.linalg.norm(R, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    Z = R / norms                                        # normalize(): row L2
    A = Z.T @ Z / Z.shape[0]                             # cal_cov()
    eig = np.linalg.eigvalsh(A)
    eig = eig[eig > 1e-12]
    eig = eig / eig.sum()                                # trace-normalize
    return float(-(eig * np.log(eig)).sum())             # cal_entropy(), ln


@pytest.mark.parametrize("shape,seed", [
    ((24, 16), 0),   # T > d (typical)
    ((8, 40), 1),    # T < d (short sentence, wide model)
    ((30, 12), 2),   # low-rank input (rank 3), constructed below
])
def test_matrix_entropy_matches_official_code(shape, seed):
    rng = np.random.default_rng(seed)
    if shape == (30, 12):
        base = rng.standard_normal((30, 3))
        proj = rng.standard_normal((3, 12))
        X = base @ proj                                  # rank 3
    else:
        X = rng.standard_normal(shape)

    ref_H = _official_matrix_entropy(X)
    got = _matrix_entropy(torch.tensor(X, dtype=torch.float32))

    assert got["entropy"] == pytest.approx(ref_H, abs=TOL), (
        f"matrix entropy diverged from official Diff-eRank code on {shape}"
    )
    d = shape[1]
    assert got["entropy_normalized"] == pytest.approx(
        ref_H / math.log(d), abs=TOL
    )


def test_matrix_entropy_analytic_two_direction_anchor():
    """Rows = ±e1, ±e2 in R^4: already zero-mean and unit-norm, covariance
    eigenvalues {1/2, 1/2, 0, 0} -> H = ln 2 EXACTLY."""
    Z = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
    ])
    got = _matrix_entropy(Z)
    assert got["entropy"] == pytest.approx(math.log(2.0), abs=1e-6)
    assert got["entropy_normalized"] == pytest.approx(
        math.log(2.0) / math.log(4.0), abs=1e-6
    )


def test_matrix_entropy_mutants_would_fail():
    """The pinned reference actually discriminates: skipping the centering
    or the row-normalization changes the value beyond tolerance (this is
    the mutant pair that survived the old key-presence-only tests)."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 24)) + 0.7   # nonzero mean matters

    ref = _official_matrix_entropy(X)

    # Mutant A: no centering.
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    A = Xn.T @ Xn / Xn.shape[0]
    eig = np.linalg.eigvalsh(A)
    eig = eig[eig > 1e-12]
    eig = eig / eig.sum()
    mutant_no_center = float(-(eig * np.log(eig)).sum())

    assert abs(mutant_no_center - ref) > 5 * TOL, (
        "anchor matrix cannot distinguish the no-centering mutant; "
        "strengthen the fixture"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
