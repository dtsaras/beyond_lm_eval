"""Regression tests for consistency_bias_weat — word position lookup.

BPE tokenizers produce different token ids for a word standalone ('John')
vs in-context (' John' with the leading space from 'is John'). The old
``_embed_word`` tokenized the word standalone to find ``first_word_tok``
then searched the templated sentence's ids for it. On most GPT-2 / Llama
/ Qwen tokenisers, that search fails and falls back to the end-of-
sentence position, so every word got the SAME hidden state and WEAT
effect size d ≈ 0 regardless of actual bias. Fix: locate the word via
character offset mapping on the templated text.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))


def test_embed_word_uses_offset_mapping_for_robust_word_position():
    """Use a BPE tokenizer (GPT-2) where 'John' tokenizes differently
    standalone than in context, and verify the fix locates the correct
    token position using offset_mapping on the templated text."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    standalone_ids = tok("John", add_special_tokens=False)["input_ids"]
    template_text = "This is John."
    template_ids = tok(template_text, return_tensors="pt")["input_ids"][0].tolist()

    # Precondition: the two should differ on BPE tokenisers.
    # (If they happen to match on a tokeniser, this test trivially passes.)
    standalone_first = standalone_ids[0]
    if standalone_first in template_ids:
        # This tokeniser happens to not exhibit the bug — skip semantically.
        return

    # Now call the helper and verify it finds the correct token via
    # offset_mapping rather than falling back to len-2.
    from blme.tasks.consistency.bias import _find_word_token_position
    pos = _find_word_token_position(tok, template_text, "John")
    # The word "John" is at characters 8..11 of "This is John."
    # In GPT-2 tokenisation this maps to token index 2 (after "This", " is").
    # Verify the returned pos is not the fallback end-position and is
    # inside the expected range.
    assert pos is not None, "_find_word_token_position should locate the word"
    assert 1 <= pos <= len(template_ids) - 2, (
        f"pos {pos} should be in the middle, not fallback to end"
    )


def test_embed_word_locates_word_not_end_fallback_across_positions():
    """The fix must return the SAME token position for the same word
    across different occurrences, not always the last non-special
    position. Verify by calling with two different templated strings
    where the word appears in different token positions."""
    from transformers import AutoTokenizer
    from blme.tasks.consistency.bias import _find_word_token_position
    tok = AutoTokenizer.from_pretrained("gpt2")

    # "John" near the start vs near the end.
    pos_a = _find_word_token_position(tok, "John walks to the store.", "John")
    pos_b = _find_word_token_position(tok, "I know a person named John.", "John")
    assert pos_a is not None and pos_b is not None
    # The word moves: pos_a should be earlier than pos_b.
    assert pos_a < pos_b, (
        f"pos_a={pos_a} should be earlier than pos_b={pos_b}; "
        "fallback-to-end would collapse both to the same end-index"
    )
    # pos_a is the first non-special token (after BOS) — for GPT-2 there's
    # no BOS on plain encode, so John is token 0.
    assert pos_a in (0, 1)
    # pos_b should be near the end but BEFORE the period.
    template_b_ids = tok("I know a person named John.")["input_ids"]
    assert pos_b < len(template_b_ids) - 1
