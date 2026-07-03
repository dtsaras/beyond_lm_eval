"""Numeric-parity test: BLME interpretability_head_roles (previous-token
score) vs. the OFFICIAL TransformerLens previous-token kernel (Olsson et al.
2022, arXiv:2209.11895; broader context Clark 2019, Voita 2019, Wang 2022 IOI).

KERNEL & STRIPE OFFSET
----------------------
A previous-token head at query position k attends to key position k-1 (the
immediately preceding token). The key-minus-query offset is -1, so the
previous-token score is the mean of the attention-pattern DIAGONAL at offset
-1. This is exactly the TransformerLens prev-token kernel:
    prev_stripe      = pattern.diagonal(dim1=-2, dim2=-1, offset=-1)
    prev_token_score = reduce(prev_stripe, "b head pos -> head", "mean").

BLME (src/blme/tasks/interpretability/head_roles.py, HeadRolesTask.evaluate,
lines 85-92): for k in [1, T): s += att[h,k,k-1]; score = s/(T-1). This is the
FULL -1 diagonal (T-1 entries, k in [1, T-1]) -- identical to TransformerLens
(unlike the induction stripe, there are no endpoint entries to differ on).

(The duplicate-token score, head_roles.py lines 100-109, is content-dependent
rather than a fixed diagonal and is not a TransformerLens demo kernel, so it is
not asserted here; the previous-token score is the kernel TransformerLens
computes exactly this way.)

RESULT: EXACT PARITY. On gpt2 with the fixture sequences the per-head max abs
diff between BLME and OFFICIAL TransformerLens is ~8.8e-8 (< 1e-4).

The reference was produced by $SCRATCH/wave2/induction_headroles_verify.py
running gpt2 in TransformerLens v3.5.1 (isolated venv $SCRATCH/venvs/tlens2)
AND HF gpt2 (eager); HF-eager attentions were verified equal to TransformerLens
patterns to ~5.5e-6. Values are frozen in
tests/fixtures/reference_parity/parity/head_roles.json.

Anchors: gpt2's early-layer previous-token head L4H11 scores ~0.98, far above
the model-wide mean; all scores lie in [0,1].
"""
import json
import os

import numpy as np
import pytest

FIXTURE = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "fixtures", "reference_parity", "parity", "head_roles.json",
)

TOL = 1e-4


def _load_fixture():
    if not os.path.exists(FIXTURE):
        pytest.skip(f"fixture missing: {FIXTURE} (run induction_headroles_verify.py)")
    with open(FIXTURE) as f:
        return json.load(f)


def _blme_prev_token_scores(att):
    """EXACT transcription of head_roles.py lines 85-92 for one sample.
    att: (H,T,T) attention pattern. Returns (H,) previous-token scores."""
    H, T, _ = att.shape
    out = np.zeros(H, dtype=np.float64)
    for h in range(H):
        s, c = 0.0, 0
        for k in range(1, T):
            s += float(att[h, k, k - 1])
            c += 1
        out[h] = s / max(1, c)
    return out


@pytest.fixture(scope="module")
def blme_and_fixture():
    fx = _load_fixture()
    token_ids = fx["token_ids"]

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
                att = layer_att[0].double().cpu().numpy()
                blme[l] += _blme_prev_token_scores(att)
    blme /= len(token_ids)
    return blme, fx


def test_prev_token_per_head_parity(blme_and_fixture):
    """BLME per-head previous-token score == OFFICIAL TransformerLens
    diagonal(-1).mean() to < 1e-4 (exact parity)."""
    blme, fx = blme_and_fixture
    ref = np.array(fx["prev_token_ref_scores_LH"], dtype=np.float64)
    assert blme.shape == ref.shape, (blme.shape, ref.shape)
    max_abs = float(np.abs(blme - ref).max())
    assert max_abs < TOL, f"BLME vs official TransformerLens max abs diff = {max_abs}"
    # Live diff agrees with the frozen fixture record.
    assert abs(max_abs - fx["max_abs_diff_official_vs_blme"]) < 1e-6


def test_prev_token_scores_in_unit_interval(blme_and_fixture):
    """Anchor: attention-weight scores must lie in [0,1]."""
    blme, _ = blme_and_fixture
    assert blme.min() >= -1e-9
    assert blme.max() <= 1.0 + 1e-9


def test_prev_token_head_is_gpt2_L4H11(blme_and_fixture):
    """Anchor: gpt2's canonical previous-token head L4H11 tops the ranking and
    sits far above the model-wide mean."""
    blme, _ = blme_and_fixture
    Hn = blme.shape[1]
    top = np.unravel_index(int(blme.argmax()), blme.shape)
    assert (int(top[0]), int(top[1])) == (4, 11), f"top prev-token head = {top}"
    assert blme.max() > 0.9, f"top prev-token score unexpectedly low: {blme.max()}"
    assert blme.max() > 3.0 * blme.mean()


def test_reference_provenance(blme_and_fixture):
    """Reference came from TransformerLens v3.5.1; HF-eager == TL patterns."""
    _, fx = blme_and_fixture
    assert fx["transformer_lens_version"] == "3.5.1"
    assert fx["prev_token_stripe_offset"] == -1
    assert fx["verdict"] == "PASS"
    assert fx["max_hf_vs_tl_attention_abs_diff"] < 1e-4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
