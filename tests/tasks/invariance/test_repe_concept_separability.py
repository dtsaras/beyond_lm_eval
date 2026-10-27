"""Direct mechanism tests for repe_concept_separability.

The LAT reading (PCA of pair differences, choice accuracy by projection) is
scale-free; the legacy logistic-regression AUC (an unstandardised L2 penalty)
is not, and is the known negative.
"""
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("sklearn")
transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import invariance as inv  # noqa: E402
from blme.tasks.repe.concept_separability import ConceptSeparabilityTask  # noqa: E402
from repe_width_padding import tiny_llama, zero_pad_llama_width  # noqa: E402

HEADLINE = ("lat_test_acc", "lat_layer_frac")


@pytest.fixture(scope="module")
def gpt2():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"gpt2 not cached: {e}")
    return model, tok


def test_lat_headline_invariant_under_residual_gauge(gpt2):
    """gpt2 with every residual state x2 (eps rescaled; logits bit-identical):
    the LAT headline does not move; the legacy LR AUC does."""
    model, tok = gpt2
    with inv.limit_threads(1):
        ids = tok("The capital of France is Paris.", return_tensors="pt")["input_ids"]
        gauged = inv.residual_scale_gauge(model, 2.0, rescale_eps=True, verify_on=ids, atol=0.0)
        base = ConceptSeparabilityTask({}).evaluate(model, tok, None)
        g = ConceptSeparabilityTask({}).evaluate(gauged, tok, None)
    inv.assert_invariant(base, g, HEADLINE, rtol=0.0)
    inv.assert_moves(base, g, "_meta_max_auc", min_rel=1e-3)


def test_lat_headline_invariant_under_width_padding():
    """4x function-preserving width padding of a Llama model."""
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = tiny_llama(len(tok))
    base = ConceptSeparabilityTask({}).evaluate(model, tok, None)
    pad = ConceptSeparabilityTask({}).evaluate(zero_pad_llama_width(model), tok, None)
    inv.assert_invariant(base, pad, HEADLINE + ("_meta_layer_lat_test_acc",), rtol=0.0)
