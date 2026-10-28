"""Direct mechanism test for geometry_perplexity's bits_per_byte headline.

Vocabulary: appending never-emitted rows with -inf logits leaves the
next-token distribution unchanged, so bits per byte must not move. (Across
tokenizers bpb is the comparable unit by construction; across model sizes it
is size-linked by meaning, a capability, and is not normalised.)
"""
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import invariance as inv  # noqa: E402
from blme.tasks.geometry.perplexity import RarePPLTask  # noqa: E402

DOCS = [{"text": t} for t in (
    "The committee met on Tuesday to discuss the new library budget.",
    "Rain fell steadily over the harbour while the ferries waited.",
    "She tuned the violin, then played the opening bars twice.",
)]


def test_bits_per_byte_invariant_under_masked_vocabulary_padding():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-gpt2", dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"tiny-random-gpt2 not cached: {e}")
    cfg = {"num_samples": len(DOCS), "use_cache": False}
    with inv.limit_threads(1):
        base = RarePPLTask(cfg).evaluate(model, tok, DOCS)
        pad = RarePPLTask(cfg).evaluate(inv.pad_vocabulary(model, 64, kind="masked"), tok, DOCS)
    assert base["_meta_ppl_byte_attribution"] in ("offsets", "token_pieces")
    inv.assert_invariant(base, pad, ["bits_per_byte", "ppl_overall", "mean_nll_nats"], rtol=1e-12)
