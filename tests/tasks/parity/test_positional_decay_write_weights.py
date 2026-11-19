"""The kernels behind geometry_positional_decay.positional_decay_nosink_write_weighted.

- _head_write_norms: head h's write ||o_{h,q}|| is the output projection
  applied to the concatenated head outputs with every other head zeroed, minus
  the bias (the bias is no head's). Checked for nn.Linear and GPT-2's Conv1D.
- _write_weighted_mean: sum w rho / sum w, which numpy.average gives.
(_row_distance_correlation is pinned in test_comprehensive_parity.py.)
"""
import numpy as np
import pytest
import torch
from transformers.pytorch_utils import Conv1D

from blme.tasks.geometry.positional_decay import _head_write_norms, _write_weighted_mean


def _reference(z, proj, n_heads):
    T, D = z.shape[1], z.shape[2]
    dh = D // n_heads
    out = np.zeros((T, n_heads))
    bias = proj.bias.detach().double()
    for h in range(n_heads):
        zh = torch.zeros_like(z, dtype=torch.float64)
        zh[..., h * dh:(h + 1) * dh] = z[..., h * dh:(h + 1) * dh].double()
        p = proj.double()
        o = p(zh)[0] - bias
        out[:, h] = torch.linalg.vector_norm(o, dim=-1).detach().numpy()
        proj.float()
    return out


@pytest.mark.parametrize("kind", ["linear", "conv1d"])
def test_head_write_norms_are_each_heads_isolated_write(kind):
    torch.manual_seed(0)
    n_heads, dh, T, d_out = 4, 5, 7, 12
    proj = torch.nn.Linear(n_heads * dh, d_out) if kind == "linear" else Conv1D(d_out, n_heads * dh)
    with torch.no_grad():
        proj.bias.normal_()
    z = torch.randn(1, T, n_heads * dh)
    got = _head_write_norms(z, proj, n_heads)
    np.testing.assert_allclose(got, _reference(z, proj, n_heads), rtol=1e-5, atol=1e-6)


def test_write_weighted_mean_is_the_weighted_average():
    rng = np.random.default_rng(0)
    w, r = rng.uniform(0, 3, 50), rng.uniform(-1, 1, 50)
    w[:5] = 0.0
    assert _write_weighted_mean(list(zip(w, r))) == pytest.approx(np.average(r, weights=w), rel=1e-12)
    assert np.isnan(_write_weighted_mean([(0.0, 0.3), (0.0, -0.2)]))


def test_batched_causal_rows_match_scalar_spearman():
    from blme.tasks.geometry.positional_decay import _causal_row_correlations, _row_distance_correlation

    rng = np.random.default_rng(129)
    for n in (4, 17, 64):
        head = rng.integers(0, 5, (n, n)).astype(np.float32)
        head[3, :] = 0.25
        head[0, -1] = np.nan
        if n > 4:
            head[4, 1] = np.nan
            head[5, 2] = np.inf
        for drop in (0, 1):
            expected = [_row_distance_correlation(head[q, drop:q]) for q in range(3, n)]
            np.testing.assert_allclose(_causal_row_correlations(head, drop),
                                       np.array(expected, dtype=float), atol=1e-14, rtol=0)
