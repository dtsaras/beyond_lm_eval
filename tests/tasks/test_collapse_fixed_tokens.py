"""collapse.qdl_entropy_min reads the first 64 tokens of each passage.

At the compression valley one singular value dominates, so the Eq. 1 entropy
grows with the number of rows; on whole passages it moved with the tokens per
passage (gpt2: 0.120 / 0.208 / 0.370 at mean T = 34 / 63 / 128, 2026-11-15
review). In a causal LM the first 64 rows do not depend on later tokens, so
cutting passages that stay >= 64 tokens leaves the headline bit-identical.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from blme.cache import load_default_corpus  # noqa: E402
from blme.tasks.geometry.collapse import RepresentationCollapseTask  # noqa: E402


def test_headline_ignores_tokens_past_the_window():
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    long = [d for d in load_default_corpus(400) if len(tok(d["text"])["input_ids"]) >= 200][:8]
    assert len(long) == 8
    cut = [{"text": tok.decode(tok(d["text"])["input_ids"][:100])} for d in long]
    a = RepresentationCollapseTask({"num_samples": 8, "use_cache": False}).evaluate(model, tok, long)
    b = RepresentationCollapseTask({"num_samples": 8, "use_cache": False}).evaluate(model, tok, cut)
    assert a["_meta_qdl_n_docs"] == b["_meta_qdl_n_docs"] == 8
    assert a["qdl_entropy_min"] == b["qdl_entropy_min"]
    assert a["_meta_qdl_entropy_min_whole_passage"] != b["_meta_qdl_entropy_min_whole_passage"]
