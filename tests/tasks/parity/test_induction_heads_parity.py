"""Numeric-parity test: BLME interpretability_induction_heads vs the OFFICIAL
TransformerLens induction-score kernel (Olsson et al. 2022, "In-context Learning
and Induction Heads", arXiv:2209.11895).

BLME computes the per-head induction (prefix-matching) score as the mean of the
attention "induction stripe" — the diagonal at offset (1 - N) for a repeated
random sequence [r_0..r_{N-1} r_0..r_{N-1}] of length 2N (query k attends to key
(k-N)+1, the token after the previous occurrence of the current token). This is
exactly TransformerLens's
    induction_score = pattern.diagonal(offset=1-seq_len).mean()
over the FULL diagonal (query rows k in [N-1, 2N-1], N+1 entries).

The test drives BLME's REAL extracted kernel
`blme.tasks.interpretability.induction._induction_score_per_head` on gpt2 and
asserts it reproduces the official TransformerLens per-head scores (frozen in the
fixture, TransformerLens v3.5.1; HF-eager attentions verified == TL patterns to
5.5e-6) to <1e-4.

History: BLME previously averaged only k in [N, 2N-2] (N-1 entries), dropping the
two endpoint stripe entries -> read ~0.03 below TransformerLens. Fixed 2026-07
(induction.py `_induction_score_per_head`, full diagonal) for exact parity with
the published metric; this test now pins the PARITY (was a documented-divergence
test). The change alters the `induction_score` study feature -> regenerate.
"""
import json
import os

import numpy as np
import pytest

from blme.tasks.interpretability.induction import _induction_score_per_head

FIXTURE = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "fixtures", "reference_parity", "parity", "induction_heads.json",
)
TOL = 1e-4


def _load_fixture():
    if not os.path.exists(FIXTURE):
        pytest.skip(f"fixture missing: {FIXTURE}")
    with open(FIXTURE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def blme_scores_and_fixture():
    fx = _load_fixture()
    token_ids = fx["token_ids"]
    N = fx["seq_len"]
    try:
        import torch
        from transformers import GPT2LMHeadModel
    except Exception as e:  # pragma: no cover
        pytest.skip(f"torch/transformers unavailable: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").to(device)
    model.eval()
    L, Hn = model.config.n_layer, model.config.n_head
    blme = np.zeros((L, Hn), dtype=np.float64)
    with torch.no_grad():
        for ids in token_ids:
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            out = model(input_ids, output_attentions=True)
            for l, layer_att in enumerate(out.attentions):
                # BLME's REAL kernel on the (H, T, T) attention pattern.
                blme[l] += np.asarray(_induction_score_per_head(layer_att[0], N))
    blme /= len(token_ids)
    return blme, fx


def test_blme_matches_official_transformerlens(blme_scores_and_fixture):
    """BLME's real induction kernel == official TransformerLens full-diagonal
    per-head induction_score (frozen fixture, v3.5.1) to <1e-4."""
    blme, fx = blme_scores_and_fixture
    ref = np.array(fx["ref_scores_LH"], dtype=np.float64)
    max_abs = float(np.abs(blme - ref).max())
    assert max_abs < TOL, f"BLME vs official TransformerLens max abs diff = {max_abs}"


def test_full_diagonal_equals_transformerlens_definition():
    """Unit check (no model): _induction_score_per_head == the exact
    TransformerLens kernel np.diagonal(offset=1-N).mean() on synthetic attention."""
    rng = np.random.default_rng(0)
    N = 6
    H, T = 3, 2 * N
    att = rng.random((H, T, T))
    att = att / att.sum(axis=-1, keepdims=True)  # row-stochastic
    blme = np.asarray(_induction_score_per_head(att, N))
    tl = np.array([np.diagonal(att[h], offset=1 - N).mean() for h in range(H)])
    assert np.max(np.abs(blme - tl)) < 1e-12
    # and it uses the full diagonal: N+1 entries.
    assert np.diagonal(att[0], offset=1 - N).size == N + 1


def test_induction_scores_in_unit_interval(blme_scores_and_fixture):
    blme, _ = blme_scores_and_fixture
    assert blme.min() >= -1e-9
    assert blme.max() <= 1.0 + 1e-9


def test_induction_top_heads_are_gpt2_known_heads(blme_scores_and_fixture):
    """Anchor: gpt2's induction heads (layers 5-7, e.g. L5H5) dominate."""
    blme, _ = blme_scores_and_fixture
    Hn = blme.shape[1]
    top_flat = np.argsort(blme, axis=None)[::-1][:5]
    top = {(int(i // Hn), int(i % Hn)) for i in top_flat}
    assert all(5 <= l <= 7 for (l, h) in top), f"top heads not in layers 5-7: {top}"
    assert blme.max() > 3.0 * blme.mean()
    assert (5, 5) in top, f"L5H5 not among top induction heads: {top}"


def test_reference_provenance(blme_scores_and_fixture):
    _, fx = blme_scores_and_fixture
    assert fx["transformer_lens_version"] == "3.5.1"
    assert fx["stripe_offset"] == 1 - fx["seq_len"]
    assert fx["max_hf_vs_tl_attention_abs_diff"] < 1e-4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
