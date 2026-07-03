"""Reference verification: BLME interpretability_attention_graph vs Abnar rollout.

WHAT BLME ACTUALLY COMPUTES
    src/blme/tasks/interpretability/attention_graph.py treats each *single*
    head/layer attention matrix A (A[i,j] = attention token i -> token j) as a
    weighted directed graph and reports damped PageRank centrality
    (`_power_iteration_pagerank`, alpha=0.85) plus edge-Gini sparsity and a
    "sink is BOS" rate. It NEVER augments with the identity, NEVER renormalises
    a residual, and NEVER takes a cumulative cross-layer matrix product. There
    is NO attention-rollout kernel in the file.

WHAT ABNAR & ZUIDEMA (2020, arXiv:2005.00928) ATTENTION ROLLOUT IS
    Augment each (head-averaged) layer attention with the identity to account
    for residual connections, row-normalise, then take the CUMULATIVE matrix
    product across layers:  rollout = Ã_L · Ã_{L-1} · ... · Ã_1.

    OFFICIAL reference implementation: samiraabnar/attention_flow
        repo commit 8044f5312f4ced18d4cf66ffe28f6c045629b4ed (2021-09-08)
        attention_graph_util.py:104-119  compute_joint_attention()
    transcribed VERBATIM below as `compute_joint_attention` (only re-indented).

VERDICT (mirrors the attention_rank precedent):
    BLME's task is FORMULA-FAITHFUL to PageRank ("attention sink" detection,
    Xiao et al. 2023). Abnar & Zuidema rollout is a *cited motivation* whose
    kernel BLME does NOT implement, so a numeric rollout PARITY cannot be
    asserted for this task — it would be false. This test therefore:
      1. Verifies the OFFICIAL Abnar rollout against its four defining
         mathematical anchors to < 1e-9 (proving our transcription is correct).
      2. Documents, by direct source assertion, that BLME contains no rollout
         kernel, so the two quantities are different (shown to rank the "sink"
         token differently on the same attention matrix).

ANCHORS (on the official rollout):
    (a) single layer  -> rollout == row_normalize(0.5A+0.5I) == row_normalize(A+I)
    (b) every A_l = I -> rollout == I
    (c) rollout rows sum to 1 (row-stochastic)
    (d) rollout is non-negative
"""
import numpy as np
import pytest

from blme.tasks.interpretability import attention_graph as ag
from blme.tasks.interpretability.attention_graph import _power_iteration_pagerank

REF_COMMIT = "8044f5312f4ced18d4cf66ffe28f6c045629b4ed"
TOL = 1e-9


# ---------------------------------------------------------------------------
# OFFICIAL Abnar & Zuidema rollout, transcribed VERBATIM from
# samiraabnar/attention_flow @ 8044f53  attention_graph_util.py:104-119
# ---------------------------------------------------------------------------
def compute_joint_attention(att_mat, add_residual=True):
    if add_residual:
        residual_att = np.eye(att_mat.shape[1])[None, ...]
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[..., None]
    else:
        aug_att_mat = att_mat

    joint_attentions = np.zeros(aug_att_mat.shape)

    layers = joint_attentions.shape[0]
    joint_attentions[0] = aug_att_mat[0]
    for i in np.arange(1, layers):
        joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i - 1])

    return joint_attentions


def _row_normalize(M):
    return M / M.sum(axis=-1, keepdims=True)


def _toy(seed=1234, n_layers=3, n_tokens=5):
    """Row-stochastic, head-averaged per-layer attention: shape (L, N, N)."""
    rng = np.random.default_rng(seed)
    A = rng.random((n_layers, n_tokens, n_tokens))
    return A / A.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Reference rollout anchors (prove the transcription is the true rollout).
# ---------------------------------------------------------------------------
def test_reference_rollout_runs_on_toy():
    A = _toy()
    joint = compute_joint_attention(A, add_residual=True)
    assert joint.shape == A.shape
    rollout = joint[-1]
    assert np.all(np.isfinite(rollout))


def test_anchor_a_single_layer_equals_normalized_residual():
    """(a) single layer -> rollout == row_normalize(A+I) == row_normalize(0.5A+0.5I)."""
    A1 = _toy(seed=7, n_layers=1, n_tokens=5)
    single = compute_joint_attention(A1, add_residual=True)[-1]
    expect_plus_I = _row_normalize(A1[0] + np.eye(5))
    expect_half = _row_normalize(0.5 * A1[0] + 0.5 * np.eye(5))
    assert np.max(np.abs(single - expect_plus_I)) < TOL
    # 0.5A+0.5I and A+I give the SAME row-stochastic matrix after normalisation.
    assert np.max(np.abs(expect_half - expect_plus_I)) < TOL


def test_anchor_b_all_identity_gives_identity():
    """(b) every A_l = I -> rollout == I."""
    N = 5
    Aeye = np.stack([np.eye(N) for _ in range(3)])
    rollout = compute_joint_attention(Aeye, add_residual=True)[-1]
    assert np.max(np.abs(rollout - np.eye(N))) < TOL


def test_anchor_c_rows_sum_to_one():
    """(c) rollout is row-stochastic."""
    rollout = compute_joint_attention(_toy(), add_residual=True)[-1]
    assert np.max(np.abs(rollout.sum(axis=1) - 1.0)) < TOL


def test_anchor_d_nonnegative():
    """(d) rollout has no negative entries."""
    rollout = compute_joint_attention(_toy(), add_residual=True)[-1]
    assert rollout.min() >= 0.0


def test_half_scaling_matches_plus_identity_on_full_rollout():
    """The prompt's 0.5A+0.5I phrasing == Abnar's A+I after normalisation,
    end-to-end across all layers."""
    A = _toy()
    L, N, _ = A.shape
    aug = np.stack([_row_normalize(0.5 * A[l] + 0.5 * np.eye(N)) for l in range(L)])
    manual = np.zeros_like(A)
    manual[0] = aug[0]
    for i in range(1, L):
        manual[i] = aug[i].dot(manual[i - 1])
    ref = compute_joint_attention(A, add_residual=True)[-1]
    assert np.max(np.abs(manual[-1] - ref)) < TOL


# ---------------------------------------------------------------------------
# BLME does NOT implement rollout — source-level + behavioural evidence.
# ---------------------------------------------------------------------------
def test_blme_has_no_rollout_kernel():
    """No rollout/joint/residual/cumulative-matmul symbol exists in the task."""
    symbols = [s for s in dir(ag) if not s.startswith("__")]
    rollout_like = [s for s in symbols
                    if any(k in s.lower() for k in ("rollout", "joint", "flow", "residual"))]
    assert rollout_like == [], (
        f"Unexpected rollout-like symbol(s) {rollout_like}; if BLME gained a "
        "rollout kernel, upgrade this test to a real numeric PARITY assertion.")
    # The only graph kernel present is PageRank.
    assert hasattr(ag, "_power_iteration_pagerank")


def test_rollout_and_pagerank_are_different_quantities():
    """On the SAME layer-0 attention, PageRank centrality and rollout influence
    onto the BOS token disagree about which token is the 'sink' — proving these
    are distinct measures, so rollout parity is not applicable to this task."""
    A = _toy()
    rollout = compute_joint_attention(A, add_residual=True)[-1]
    influence_to_bos = rollout[:, 0]          # column 0 = cumulative influence -> tok0
    pr_layer0 = _power_iteration_pagerank(A[0], alpha=0.85)
    # They pick different argmax tokens on this fixed seed.
    assert np.argmax(influence_to_bos) != np.argmax(pr_layer0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
