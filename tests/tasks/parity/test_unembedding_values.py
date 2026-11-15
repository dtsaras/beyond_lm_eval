"""geometry_unembedding: the five headline keys through evaluate().

The unembedding is replaced by a matrix whose first half of rows is the
final-normed input embedding plus a little noise (cosine above the 0.9 cut)
and whose second half is unrelated and shares a large offset, so the
Roy-Vetterli effective rank of the raw matrix and of its column-centred copy
differ (test_comprehensive_parity's check compares the raw key to the
centred value on a zero-mean matrix, where the two coincide). The
reference applies the final LayerNorm with its own formula (F.layer_norm,
the module's weight and bias) and takes singular values with scipy.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.linalg import svdvals

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "invariance"))
import _family_d as fd  # noqa: E402
from blme.tasks.geometry.unembedding import UnembeddingDiagnosticsTask  # noqa: E402


def _erank(M):
    s = svdvals(np.asarray(M, dtype=np.float64))
    p = s[s > 0] / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def _unit(M):
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def test_unembedding_keys_match_an_independent_recompute():
    model = fd.tiny_gpt2()
    ln = model.transformer.ln_f
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        ln.weight.copy_(0.5 + torch.rand(ln.weight.shape, generator=g))
        ln.bias.copy_(0.2 * torch.randn(ln.bias.shape, generator=g))
    E = model.get_input_embeddings().weight.detach().double()
    V, d = E.shape
    E_n = F.layer_norm(E, (d,), ln.weight.detach().double(), ln.bias.detach().double(),
                       eps=ln.eps).numpy()
    rng = np.random.default_rng(1)
    W = np.where(np.arange(V)[:, None] < V // 2,
                 E_n + 0.05 * rng.standard_normal((V, d)),
                 rng.standard_normal((V, d)) + 4.0 * rng.standard_normal(d))
    model.lm_head.weight = torch.nn.Parameter(torch.tensor(W, dtype=torch.float32))
    W = model.lm_head.weight.detach().double().numpy()

    res = UnembeddingDiagnosticsTask({}).evaluate(model, fd.tokenizer(), dataset=[])
    assert res["unembedding_is_tied"] is False

    raw, cen = _erank(W), _erank(W - W.mean(axis=0))
    assert abs(raw - cen) > 0.5, "the fixture must separate raw from centred"
    assert res["unembedding_eff_rank"] == pytest.approx(raw, rel=1e-5)
    assert res["unembedding_eff_rank_centered"] == pytest.approx(cen, rel=1e-5)

    cos = np.sum(_unit(E_n) * _unit(W), axis=1)
    frac = float(np.mean(cos > 0.9))
    assert 0.2 < frac < 0.8
    assert res["embedding_alignment_mean"] == pytest.approx(cos.mean(), rel=1e-5)
    assert res["embedding_alignment_std"] == pytest.approx(cos.std(), rel=1e-5)
    assert res["embedding_high_alignment_frac"] == frac
