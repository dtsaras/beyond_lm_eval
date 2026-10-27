"""Direct mechanism test for geometry_tokenizer_efficiency's fertility.

Vocabulary: adding tokens that never occur in the corpus leaves every
segmentation, so fertility and the token entropy do not move; vocab_size and
the utilisation share (distinct / V) do.
"""
import copy
import sys
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import invariance as inv  # noqa: E402
from blme.tasks.geometry.tokenizer_efficiency import TokenizerEfficiencyTask  # noqa: E402


def test_fertility_invariant_under_unused_vocabulary():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"gpt2 tokenizer not cached: {e}")
    big = copy.deepcopy(tok)
    big.add_tokens([f"<unused_token_{i}>" for i in range(2000)])
    base = TokenizerEfficiencyTask({}).evaluate(None, tok, None)
    pad = TokenizerEfficiencyTask({}).evaluate(None, big, None)
    inv.assert_invariant(base, pad, ["fertility", "token_entropy"], rtol=0.0)
    assert pad["vocab_size"] == base["vocab_size"] + 2000
    inv.assert_moves(base, pad, "_meta_vocab_utilization", min_rel=0.02)
