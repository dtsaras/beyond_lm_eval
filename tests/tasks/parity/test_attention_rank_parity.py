"""Numeric-parity test: BLME interpretability_attention_rank effective rank.

BLME's per-head attention "effective rank" is Roy & Vetterli's (2007) SVD-entropy
effective rank:
        erank(A) = exp(H(p)),  p_i = sigma_i / sum_j sigma_j,
        H(p)     = -sum_i p_i ln p_i        (natural log)
where sigma are the singular values of the (T x T) attention matrix.

OFFICIAL references:
  (A) Roy & Vetterli (2007), "The effective rank: a measure of effective
      dimensionality", EUSIPCO. Definition transcribed independently below
      (numpy SVD -> normalise -> exp(entropy)). This is exactly what BLME's
      `_effective_rank` computes. Target tolerance: 1e-12.

  (B) Dong, Cordonnier & Loukas (2021), "Attention is not all you need" (ICML),
      arXiv:2103.03404. Their official repo
          github.com/twistedcubic/attention-rank-collapse
          commit 38b5df6dc2add25f6d945e48a6baf96862368c20
      measures rank collapse with a RANK-1 RESIDUAL, not effective rank:
          paper-plotting/utils.py (compute_low_rank / l1_matrix_norm /
            linf_matrix_norm / composite_norm) +
          paper-plotting/walking_casual_transformers.ipynb cell 19:
              rank_one  = compute_low_rank(P, k=1)
              residuals = P - rank_one
              ratio     = norm_fn(residuals) / norm_fn(P)   # norm = sqrt(l1*linf)
      This is a DIFFERENT quantity. BLME is FORMULA-FAITHFUL to Roy-Vetterli;
      Dong (2021) is the motivation, not the kernel. The residual is transcribed
      here only to confirm the two measures move oppositely on anchor matrices
      (rank-1: erank->1, residual->0 ; full-rank: erank->n, residual large).

Anchors: rank-1 matrix -> erank == 1.0 ; n x n orthogonal -> erank == n.
"""
import numpy as np
import pytest
import torch

from blme.tasks.interpretability.attention_rank import _effective_rank

TOL = 1e-12
ANCHOR_TOL = 1e-12


# ---------------------------------------------------------------------------
# (A) Independent Roy & Vetterli (2007) effective rank.
# ---------------------------------------------------------------------------
def roy_vetterli_effective_rank(matrix: np.ndarray) -> float:
    s = np.linalg.svd(matrix, compute_uv=False)
    total = s.sum()
    if total <= 0:
        return float("nan")
    p = s / total
    p = p[p > 0]  # 0*log0 == 0
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


# ---------------------------------------------------------------------------
# (B) Dong et al. (2021) rank-1 residual ratio — exact transcription of
# twistedcubic/attention-rank-collapse @ 38b5df6 (utils.py + notebook cell 19).
# ---------------------------------------------------------------------------
def _dong_low_rank(x: np.ndarray, k: int = 1) -> np.ndarray:
    U, s, Vh = np.linalg.svd(x)
    return np.einsum("ij,j,jk->ik", U[:, :k], s[:k], Vh[:k, :])


def _dong_l1_norm(x: torch.Tensor) -> torch.Tensor:
    return x.abs().sum(axis=-2 % x.ndim).max(axis=-1).values


def _dong_linf_norm(x: torch.Tensor) -> torch.Tensor:
    return _dong_l1_norm(x.transpose(-2, -1))


def _dong_composite_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_dong_l1_norm(x) * _dong_linf_norm(x))


def dong_residual_ratio(P: np.ndarray) -> float:
    rank_one = _dong_low_rank(P, k=1)
    res = torch.tensor(P - rank_one, dtype=torch.float64)
    Pt = torch.tensor(P, dtype=torch.float64)
    return float(_dong_composite_norm(res) / _dong_composite_norm(Pt))


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_effective_rank_parity_row_stochastic():
    """BLME _effective_rank == Roy-Vetterli on a 6x6 row-stochastic attn map."""
    rng = np.random.default_rng(20240624)
    A = _softmax_rows(rng.standard_normal((6, 6)))
    assert np.allclose(A.sum(axis=1), 1.0)
    blme = _effective_rank(A)
    ref = roy_vetterli_effective_rank(A)
    assert abs(blme - ref) < TOL, f"BLME={blme} REF={ref}"


def test_effective_rank_parity_near_rank1():
    """Near-rank-1 stochastic matrix: erank ~ 1, parity holds, residual ~ 0."""
    rng = np.random.default_rng(7)
    base = _softmax_rows(rng.standard_normal((1, 6)))
    perturb = 1e-3 * rng.standard_normal((6, 6))
    B = _softmax_rows(np.log(np.repeat(base, 6, axis=0) + 1e-9) + perturb)
    blme = _effective_rank(B)
    ref = roy_vetterli_effective_rank(B)
    assert abs(blme - ref) < TOL, f"BLME={blme} REF={ref}"
    assert blme < 1.1                      # nearly collapsed
    assert dong_residual_ratio(B) < 1e-2   # Dong residual small => same regime


def test_rank1_anchor_effective_rank_is_one():
    """Ground truth: an exact rank-1 row-stochastic matrix has effective rank 1.0."""
    rng = np.random.default_rng(123)
    v = rng.random(6)
    v = v / v.sum()
    R1 = np.outer(np.ones(6), v)           # all rows identical -> rank 1
    blme = _effective_rank(R1)
    assert abs(blme - 1.0) < ANCHOR_TOL, f"erank={blme}"
    assert abs(blme - roy_vetterli_effective_rank(R1)) < TOL
    # Dong residual must vanish on a rank-1 matrix.
    assert dong_residual_ratio(R1) < 1e-10


def test_orthogonal_anchor_effective_rank_is_n():
    """Ground truth: an n x n orthogonal matrix (all sigma=1) has effective rank n."""
    rng = np.random.default_rng(321)
    Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    blme = _effective_rank(Q)
    assert abs(blme - 6.0) < 1e-9, f"erank={blme}"
    assert abs(blme - roy_vetterli_effective_rank(Q)) < TOL


def test_identity_anchor_effective_rank_is_n():
    """Identity (a permutation attention map): erank == n = 6."""
    I = np.eye(6)
    blme = _effective_rank(I)
    assert abs(blme - 6.0) < 1e-9, f"erank={blme}"
    assert abs(blme - roy_vetterli_effective_rank(I)) < TOL


def test_closed_form_singular_values_2_1():
    """Exact closed form on known singular values sigma=[2,1].

    p = [2/3, 1/3]; erank = exp(-sum p ln p) computed with natural log.
    The Roy-Vetterli definition is base-consistent: erank = exp_b(H_b(p)) is
    base-invariant, so this anchors the *normalisation + entropy* mechanics
    rather than the log base. Expected value ~ 1.88988.
    """
    M = np.diag([2.0, 1.0])  # singular values exactly [2, 1]
    p = np.array([2.0, 1.0]) / 3.0
    expected = float(np.exp(-np.sum(p * np.log(p))))  # ~1.88988
    blme = _effective_rank(M)
    assert abs(blme - expected) < TOL, f"BLME={blme} expected={expected}"
    assert abs(blme - roy_vetterli_effective_rank(M)) < TOL


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
