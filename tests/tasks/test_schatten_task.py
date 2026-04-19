"""Tests for geometry_schatten (new round-7 task).

Implements the Schatten-p norm family (Wei et al. 2025 — "From Internal
Representations to Text Quality", arXiv:2509.25359) plus the L1,2-based
Matrix Nuclear-Norm fast approximation (Li et al. 2024 — arXiv:2410.10672,
reference code https://github.com/MLGroupJLU/MatrixNuclearNorm).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))


def test_schatten_norms_math_identities():
    """Smoke-test the core formulas against analytic expectations on a
    simple 3×3 matrix with known singular values [3, 2, 1]."""
    from blme.tasks.geometry.schatten import _schatten_p_norm, _matrix_nuclear_norm_fast

    S = np.array([3.0, 2.0, 1.0])
    # Schatten-1 (nuclear) = sum σ = 6.
    assert _schatten_p_norm(S, p=1) == pytest.approx(6.0)
    # Schatten-2 (Frobenius) = sqrt(Σ σ²) = sqrt(14) ≈ 3.742.
    assert _schatten_p_norm(S, p=2) == pytest.approx(np.sqrt(14.0))
    # Schatten-∞ (spectral) = max σ = 3.
    assert _schatten_p_norm(S, p=float("inf")) == pytest.approx(3.0)
    # Schatten-4 = (Σ σ⁴)^{1/4} = (81+16+1)^{1/4} = 98^{1/4}.
    assert _schatten_p_norm(S, p=4) == pytest.approx(98.0 ** 0.25)


def test_matrix_nuclear_norm_fast_formula_identities():
    """Verify the Li et al. 2024 MNN computation matches the formula:
    sum of the top-D column L2-norms."""
    from blme.tasks.geometry.schatten import _matrix_nuclear_norm_fast

    # Diagonal matrix with columns [3,0,0], [0,2,0], [0,0,1].
    # Column L2 norms = [3, 2, 1]; sum of top-3 = 6.
    diag = torch.diag(torch.tensor([3.0, 2.0, 1.0]))
    assert _matrix_nuclear_norm_fast(diag) == pytest.approx(6.0)

    # All zeros: MNN = 0.
    zeros = torch.zeros(5, 5)
    assert _matrix_nuclear_norm_fast(zeros) == pytest.approx(0.0)

    # Random 10×20 matrix: MNN sums the TOP D = min(10, 20) = 10 column
    # L2 norms, sorted descending. Verify exact match to reference impl.
    torch.manual_seed(0)
    X = torch.randn(10, 20)
    col_l2 = torch.sqrt((X * X).sum(dim=0))
    top10 = torch.sort(col_l2, descending=True).values[:10]
    expected = float(top10.sum().item())
    got = _matrix_nuclear_norm_fast(X)
    assert got == pytest.approx(expected, rel=1e-5)


def test_rankme_matches_garrido_definition():
    """RankMe (Garrido 2023) normalises *raw* singular values:
    ``p_i = σ_i / Σ σ_j`` then ``exp(H(p))``. Different from our
    Roy-Vetterli ``effective_rank`` which uses σ² / Σσ²."""
    from blme.tasks.geometry.schatten import _rankme

    # For a rank-1 matrix (only one non-zero sigma), RankMe = 1.
    S = np.array([5.0, 0.0, 0.0, 0.0])
    assert _rankme(S) == pytest.approx(1.0)
    # For a uniform spectrum (all equal σ), RankMe = N.
    S = np.array([2.0, 2.0, 2.0, 2.0])
    assert _rankme(S) == pytest.approx(4.0)
    # Must differ from Roy-Vetterli effective_rank for non-uniform σ.
    from blme.tasks.geometry.utils import effective_rank
    S = np.array([3.0, 2.0, 1.0])
    rm = _rankme(S)
    er = effective_rank(S)
    assert rm != pytest.approx(er, rel=1e-3)  # they're different formulas
    # Both should be in (1, 3].
    assert 1.0 < rm <= 3.0
    assert 1.0 < er <= 3.0


def test_schatten_task_registers_and_returns_expected_fields():
    """The full task, when evaluated on mock hidden states, returns all
    the Schatten-p norms + MNN + RankMe at headline positions."""
    import blme.tasks  # trigger registration
    from blme.registry import get_task

    cls = get_task("geometry_schatten")
    assert cls is not None
    task = cls({"num_samples": 4, "use_cache": False})

    # Stub model: a simple nn.Module that returns hidden_states.
    import torch.nn as nn

    class StubModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 3 layers × d_model=16
            self.p = nn.Linear(16, 16, bias=False)

        def forward(self, input_ids=None, **kw):
            torch.manual_seed(int(input_ids.sum().item()))
            B, T = input_ids.shape
            hs = tuple(torch.randn(B, T, 16) for _ in range(4))  # 4 = 3 layers + embedding

            class Out:
                def __init__(self, hidden_states):
                    self.hidden_states = hidden_states
            return Out(hs)

        def parameters(self):
            yield torch.zeros(1, device="cpu")

    class StubTok:
        def __call__(self, text, return_tensors="pt", truncation=True, max_length=128):
            # Deterministic ids derived from text length.
            ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
            return {"input_ids": ids}

        def encode(self, text, return_tensors="pt", truncation=True, max_length=128):
            return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)

    dataset = [{"text": "x"}] * 4
    result = task.evaluate(model=StubModel(), tokenizer=StubTok(),
                           dataset=dataset, cache=None)
    # Headline keys
    assert "schatten_1_last" in result
    assert "schatten_2_last" in result
    assert "schatten_inf_last" in result
    assert "matrix_nuclear_norm_last" in result
    assert "rankme_last" in result
    # Per-layer lists
    assert "schatten_1_per_layer" in result
    assert len(result["schatten_1_per_layer"]) == 3  # 3 blocks after embedding
