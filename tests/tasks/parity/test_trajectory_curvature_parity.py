"""Numeric-parity test: BLME geometry_trajectory_curvature vs the official
Hosseini & Fedorenko curvature computation.

TASK: geometry_trajectory_curvature
BLME: src/blme/tasks/geometry/trajectory_curvature.py
      (_trajectory_angles, _mean_curvature, _pooled_layer_curvature)

Paper: Hosseini & Fedorenko, "Large language models implicitly learn to
       straighten neural sentence trajectories to construct a predictive
       representation of natural language", NeurIPS 2023, arXiv:2311.04930.
Repo:  eghbalhosseini/sent_sampling — utils/curvature_utils.py
       compute_model_curvature (L29-46): v_k = x_{k+1} - x_k;
       c_k = arccos(v_{k+1}·v_k / (||v_{k+1}||·||v_k||)); per-sentence mean
       C_sn = (1/k)Σc_i, then averaged over sentences.

Campaign-3 found the layer aggregation POOLED all angles (length-weighting
sentences) instead of the paper's mean-of-per-sentence-means; the fix
landed, and this file makes the aggregation contract a permanent parity
test (no external-reference test existed before). The transcription below
is the official per-sentence kernel in float64.
"""
import numpy as np
import pytest

from blme.tasks.geometry.trajectory_curvature import (  # noqa: E402
    _trajectory_angles,
    _mean_curvature,
    _pooled_layer_curvature,
)

TOL = 1e-12


# ---------------------------------------------------------------------------
# Verbatim float64 transcription of the official curvature kernel
# (sent_sampling utils/curvature_utils.py compute_model_curvature).
# ---------------------------------------------------------------------------
def _official_sentence_angles(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    V = np.diff(X, axis=0)                       # v_k = x_{k+1} - x_k
    angles = []
    for k in range(V.shape[0] - 1):
        a, b = V[k], V[k + 1]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        angles.append(float(np.arccos(np.clip(cos, -1.0, 1.0))))
    return np.asarray(angles)


def _official_layer_curvature(chunks, skip_first=1) -> float:
    """Paper aggregation: C_sn = mean of a sentence's angles, layer value =
    mean of C_sn over sentences (NOT a pooled mean over all angles)."""
    per_sentence = []
    for X in chunks:
        X = np.asarray(X, dtype=np.float64)[skip_first:]
        if X.shape[0] < 3:
            continue
        ang = _official_sentence_angles(X)
        if ang.size:
            per_sentence.append(float(np.mean(ang)))
    return float(np.mean(per_sentence)) if per_sentence else float("nan")


def _random_chunks(seed, n_sent=8):
    """Variable-length float64 trajectories — lengths differ so pooled-vs-
    mean-of-means aggregations genuinely disagree."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(5, 40, size=n_sent)
    return [rng.standard_normal((int(L), 12)) for L in lengths]


def test_angle_kernel_matches_official():
    rng = np.random.default_rng(0)
    for _ in range(50):
        X = rng.standard_normal((rng.integers(4, 30), 8))
        got = _trajectory_angles(X)
        ref = _official_sentence_angles(X)
        assert got.shape == ref.shape
        assert np.max(np.abs(got - ref)) < 1e-9 if ref.size else True


def test_layer_aggregation_matches_paper_mean_of_sentence_means():
    """The campaign-3 discrepancy: pooled angles vs the paper's per-sentence
    means. BLME must equal the official aggregation on variable-length
    trajectories, and the pooled variant must DIFFER (discriminating)."""
    chunks = _random_chunks(1)
    blme = _pooled_layer_curvature(chunks, skip_first=1)
    ref = _official_layer_curvature(chunks, skip_first=1)
    assert blme == pytest.approx(ref, abs=TOL)

    all_angles = np.concatenate([
        _official_sentence_angles(np.asarray(c, dtype=np.float64)[1:])
        for c in chunks
    ])
    pooled = float(np.mean(all_angles))
    assert abs(pooled - ref) > 1e-4, (
        "chunks cannot discriminate pooled from per-sentence aggregation; "
        "regenerate with more length variance"
    )


def test_mean_curvature_analytic_anchors():
    line = np.stack([np.arange(6.0), np.zeros(6)], axis=1)
    assert _mean_curvature(line) == pytest.approx(0.0, abs=1e-9)

    right = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    assert _mean_curvature(right) == pytest.approx(np.pi / 2, abs=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
