"""geometry_positional_decay scores exactly the first 256 content tokens of long documents.

Rowwise Spearman magnitude grows with row length (gpt2 headline -0.228 / -0.294
/ -0.337 for 15 / 40 / whole-passage rows, 2026-11-15 review), so every model
must score the same row lengths. An appended EOS is stripped before the cut,
so an EOS-appending tokenizer still scores 256 content tokens.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from blme.tasks.geometry.positional_decay import PositionalAttentionDecayTask  # noqa: E402


def test_rows_are_the_first_256_tokens_of_long_documents(monkeypatch):
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    seen = []
    orig = model.forward

    def spy(*a, **k):
        seen.append(int(k["input_ids"].shape[1]))
        return orig(*a, **k)
    monkeypatch.setattr(model, "forward", spy)
    r = PositionalAttentionDecayTask({"num_samples": 3, "seq_tokens": 128}).evaluate(model, tok, None)
    assert seen and set(seen) == {128}
    assert r["_meta_documents"] == "long" and r["_meta_seq_tokens"] == 128
    assert r["positional_decay_nosink_write_weighted"] == r["positional_decay_nosink_write_weighted"]
