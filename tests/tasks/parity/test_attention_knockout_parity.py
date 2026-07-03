"""Numeric-parity test: BLME causality_attention_knockout (per-head ablation
delta-NLL) vs. a FAITHFUL reimplementation of Michel et al. 2019 head
importance ("Are Sixteen Heads Really Better than One?", NeurIPS 2019,
arXiv:1905.10650; head-ablation lineage also Voita et al. 2019).

WHAT MICHEL DEFINES
-------------------
Michel masks head h with a variable xi_h that multiplies its output (Eq. 4:
MHAtt = sum_h xi_h * Att^h(...)). A head's importance is the loss change when
it is masked out (xi_h -> 0). Rather than one forward per head, Michel also
gives a single-backward PROXY (Eq. 5):
        I_h = E_{x~X} | d L(x) / d xi_h |   evaluated at xi = 1.

WHICH VARIANT BLME COMPUTES
---------------------------
BLME (src/blme/tasks/causality/attention_knockout.py, AttentionKnockoutTask.
evaluate) registers a forward_pre_hook on the attention output projection
(c_proj in gpt2), ZEROS head h's head_dim-sized slice of the pre-projection
tensor (lines 210-237), and records mean(loss) - baseline_mean_loss
(lines 224-237). This is DIRECT ABLATION delta-NLL (variant A / xi_h -> 0),
NOT the gradient proxy. VERDICT for the metric BLME reports: FAITHFUL to the
direct-ablation importance.

RESULT: EXACT PARITY. On gpt2 (eager) with the fixture text (22 tokens), the
per-head ablation delta-NLL from BLME equals the independent faithful reimpl
to max abs diff = 0.0 (< 1e-4). Reference values (both the ablation grid and
the Eq.5 gradient proxy) were produced by
$SCRATCH/wave2/attention_knockout_verify.py running gpt2 and frozen in
tests/fixtures/reference_parity/parity/attention_knockout.json.

ANCHORS
-------
(a) Knocking out a head never lowers loss below clean by more than noise
    (min ablation impact ~ -0.09 NLL on this short single text).
(b) Michel's key claim: the Eq.5 gradient proxy tracks true ablation
    importance. Because the proxy I_h = |dL/dxi_h| is a MAGNITUDE (abs value),
    it tracks the MAGNITUDE of the ablation delta-loss:
    Spearman(proxy, |ablation delta-NLL|) ~ 0.82 (full 12x12 grid).
(c) Ablation importances are ~non-negative (BLME clips negatives to 0 before
    the Gini); the raw signed grid has only small negative entries on a short
    single text.
"""
import json
import os

import numpy as np
import pytest

FIXTURE = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "fixtures", "reference_parity", "parity", "attention_knockout.json",
)

TOL = 1e-4


def _load_fixture():
    if not os.path.exists(FIXTURE):
        pytest.skip(
            f"fixture missing: {FIXTURE} (run wave2/attention_knockout_verify.py)"
        )
    with open(FIXTURE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def blme_and_fixture():
    fx = _load_fixture()
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    except Exception as e:  # pragma: no cover
        pytest.skip(f"torch/transformers unavailable: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(
        fx["model"], attn_implementation=fx["attn_implementation"]
    ).to(device)
    model.eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(fx["model"])

    # Sanity: the token ids BLME will produce for fx["text"] must equal the
    # frozen ids the reference was computed on (else the comparison is void).
    ids = tokenizer.encode(
        fx["text"], return_tensors="pt", truncation=True, max_length=128
    )
    assert ids[0].tolist() == fx["token_ids"], "tokeniser drift vs frozen fixture"

    # Run the ACTUAL BLME task on the same model + text.
    from blme.tasks.causality.attention_knockout import AttentionKnockoutTask

    task = AttentionKnockoutTask({"num_samples": 1})
    result = task.evaluate(model, tokenizer, [{"text": fx["text"]}])
    assert "error" not in result, result.get("error")
    return result, fx


def test_blme_is_direct_ablation_delta_nll(blme_and_fixture):
    """BLME reports the DIRECT ABLATION delta-NLL (variant A), not the
    gradient proxy. Its per-head impacts must match the independent faithful
    ablation reimplementation to < 1e-4 (exact parity)."""
    result, fx = blme_and_fixture
    blme_flat = np.array(result["per_head_impacts"], dtype=np.float64)

    # Faithful ablation grid, restricted to the layers BLME analysed and
    # flattened in the same (layer-major, head-minor) order.
    ablate_LH = np.array(fx["faithful_ablation_LH"], dtype=np.float64)
    sel = fx["blme_layers_selected"]
    faithful_sel = ablate_LH[sel, :].reshape(-1)

    assert blme_flat.shape == faithful_sel.shape, (blme_flat.shape, faithful_sel.shape)
    max_abs = float(np.abs(blme_flat - faithful_sel).max())
    assert max_abs < TOL, f"BLME vs faithful ablation max abs diff = {max_abs}"
    # Live diff agrees with the frozen fixture record (both were exact 0.0).
    assert abs(max_abs - fx["max_abs_diff_blme_vs_faithful"]) < TOL

    # Baseline (clean) NLL parity.
    assert abs(result["baseline_loss"] - fx["clean_nll_faithful"]) < TOL


def test_anchor_a_no_head_helps_beyond_noise(blme_and_fixture):
    """ANCHOR (a): knocking out a head never decreases loss below clean by
    more than noise. On this short single text the most-negative ablation
    impact is small (~ -0.1 NLL)."""
    _, fx = blme_and_fixture
    ablate = np.array(fx["faithful_ablation_LH"], dtype=np.float64)
    assert ablate.min() > -0.2, f"a head 'helped' too much: {ablate.min()}"
    # The recorded min matches the live grid statistic.
    assert abs(ablate.min() - fx["min_ablation_impact"]) < 1e-9


def test_anchor_b_proxy_tracks_ablation_magnitude(blme_and_fixture):
    """ANCHOR (b) -- Michel's key claim. The Eq.5 gradient proxy
    I_h = |dL/dxi_h| tracks the MAGNITUDE of the true ablation delta-loss.
    Spearman(proxy, |ablation|) is strongly positive."""
    _, fx = blme_and_fixture
    ablate = np.array(fx["faithful_ablation_LH"], dtype=np.float64).reshape(-1)
    proxy = np.array(fx["faithful_gradient_proxy_LH"], dtype=np.float64).reshape(-1)

    def spearman(x, y):
        xr = np.argsort(np.argsort(x)).astype(float)
        yr = np.argsort(np.argsort(y)).astype(float)
        xr -= xr.mean()
        yr -= yr.mean()
        d = np.sqrt((xr**2).sum() * (yr**2).sum())
        return float((xr * yr).sum() / d) if d > 0 else float("nan")

    rho = spearman(proxy, np.abs(ablate))
    assert rho > 0.5, f"proxy failed to track ablation magnitude: rho={rho}"
    # Matches the frozen record.
    assert abs(rho - fx["spearman_proxy_vs_absablation_full"]) < 1e-6

    # The proxy is a MAGNITUDE, so it should NOT track the SIGN of the ablation
    # (that is why signed-Spearman is near zero) -- documents the distinction.
    assert abs(fx["spearman_proxy_vs_signed_ablation_full"]) < 0.3


def test_anchor_c_ablation_importances_nonnegative(blme_and_fixture):
    """ANCHOR (c): ablation importances are ~non-negative; BLME clips negatives
    to 0 before computing the Gini, and its reported max/mean impacts are the
    clipped-positive statistics, so they are non-negative."""
    result, _ = blme_and_fixture
    assert result["max_knockout_impact"] >= 0.0
    assert result["mean_knockout_impact"] >= 0.0
    assert result["head_impact_gini_coefficient"] >= 0.0
    # The raw per-head grid may contain small negatives, but the strongly
    # positive impacts dominate (a genuinely important head raises NLL a lot).
    blme_flat = np.array(result["per_head_impacts"], dtype=np.float64)
    assert blme_flat.max() > 0.5, f"no strongly-important head found: {blme_flat.max()}"


def test_reference_provenance(blme_and_fixture):
    """Reference is the faithful Michel reimplementation; metric is direct
    ablation delta-NLL and parity was exact."""
    _, fx = blme_and_fixture
    assert fx["importance_definition"] == "direct_ablation_delta_NLL"
    assert "1905.10650" in fx["reference_paper"]
    assert fx["verdict"] == "PARITY"
    assert fx["max_abs_diff_blme_vs_faithful"] < TOL


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
