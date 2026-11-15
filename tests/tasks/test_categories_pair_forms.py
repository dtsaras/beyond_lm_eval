"""geometry_categories relation pairs use one token form for both words.

On byte-level BPE tokenizers a word and its space-prefixed form are different
tokens; taking the bare form for one word of a pair and the spaced form for
the other puts the leading-space direction into E[t2] - E[t1]. The 2026-11-15
review measured gpt2 present_past_consistency 0.104 with mixed forms against
0.270 with consistent ones (22 of 49 pairs were mixed).
"""
import sys
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

from blme.tasks.geometry.categories import CategoryGeometryTask  # noqa: E402


def _forms(tok, task):
    out = []
    for rel, pairs in task.relation_pairs.items():
        for t1, t2 in pairs:
            s1, s2 = tok.convert_ids_to_tokens([t1, t2])
            out.append((rel, s1, s2))
    return out


def _gpt2():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"gpt2 tokenizer not cached: {e}")
    cfg = transformers.GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=len(tok))
    return transformers.GPT2LMHeadModel(cfg).eval(), tok


def test_every_relation_pair_has_one_form():
    model, tok = _gpt2()
    task = CategoryGeometryTask({})
    task.evaluate(model, tok, None)
    forms = _forms(tok, task)
    assert forms, "no relation pairs mapped"
    for rel, s1, s2 in forms:
        assert s1.startswith("Ġ") == s2.startswith("Ġ"), (rel, s1, s2)
