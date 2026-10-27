"""Direct mechanism tests for geometry_representation_sensitivity's S_unit.

S_unit = mean ||n||^2 ||gamma (W_y - p W)||^2 with n the unit-gain normalised
state. Moving scale between the final norm and the unembedding
(gamma -> D gamma, beta -> D beta, W -> W D^-1) leaves every logit unchanged:
S_unit must not move, while the raw ||grad_h log p||^2 moves by 1/D^2.
"""
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import invariance as inv  # noqa: E402
from blme.tasks.geometry.information_geometry import RepresentationSensitivityTask  # noqa: E402

DOCS = [{"text": t} for t in (
    "The committee met on Tuesday to discuss the new library budget.",
    "Rain fell steadily over the harbour while the ferries waited.",
    "She tuned the violin, then played the opening bars twice.",
    "A small dog slept beneath the wooden bench in the park.",
)]


@pytest.fixture(scope="module")
def gpt2():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"gpt2 not cached: {e}")
    return model, tok


def _run(model, tok):
    with inv.limit_threads(1):
        return RepresentationSensitivityTask({"num_samples": len(DOCS)}).evaluate(model, tok, DOCS)


def test_s_unit_invariant_under_final_norm_unembedding_gauge(gpt2):
    model, tok = gpt2
    ids = tok(DOCS[0]["text"], return_tensors="pt")["input_ids"]
    base = _run(model, tok)
    assert base["_meta_final_norm_gain_form"] == "weight"
    assert base["S_unit"] == base["S_unit"]          # defined on gpt2

    # Power-of-two scalar: exact (logits bit-identical).
    g2 = _run(inv.final_norm_unembedding_gauge(model, D=2.0, verify_on=ids, atol=0.0), tok)
    inv.assert_invariant(base, g2, ["S_unit"], rtol=0.0)
    assert g2["representation_sensitivity"] == pytest.approx(
        base["representation_sensitivity"] / 4, rel=1e-12)

    # A random log-normal diagonal: float32 rounding only.
    gd = _run(inv.final_norm_unembedding_gauge(model, D=None, seed=0), tok)
    inv.assert_invariant(base, gd, ["S_unit"], rtol=1e-4)
    inv.assert_moves(base, gd, "representation_sensitivity", min_rel=0.1)


def test_s_unit_invariant_under_residual_gauge(gpt2):
    model, tok = gpt2
    ids = tok(DOCS[0]["text"], return_tensors="pt")["input_ids"]
    base = _run(model, tok)
    g = _run(inv.residual_scale_gauge(model, 2.0, rescale_eps=True, verify_on=ids, atol=0.0), tok)
    inv.assert_invariant(base, g, ["S_unit", "representation_sensitivity"], rtol=1e-6)
