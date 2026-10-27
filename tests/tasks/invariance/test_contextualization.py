"""geometry_contextualization: position 0 dropped for every tokenizer, with
the reference protocol kept as _meta_ (real model; the task runs its own
forward pass)."""
from __future__ import annotations

import pytest

from _family_c import load

TASK = "geometry_contextualization"

# Every sentence opens with "Yesterday", which occurs nowhere else: in a causal
# LM without BOS its position-0 state is context-free (identical in every
# sentence), so the reference protocol tracks it with SelfSim exactly 1.
SENTENCES = [
    "Yesterday the dog ran to the park and the cat sat on the wall.",
    "Yesterday the market fell and the bank raised the rate again.",
    "Yesterday the river rose and the town moved to the hill.",
    "Yesterday the team won the game and the fans sang in the street.",
    "Yesterday the storm hit the coast and the trees fell on the road.",
    "Yesterday the choir sang and the hall was full of the music.",
    "Yesterday the chef cooked the fish and the guests ate the soup.",
]


@pytest.fixture(scope="module")
def gpt2():
    return load("gpt2")


def _run(model, tok, **cfg):
    from blme.registry import get_task
    from blme.core import _register_all_tasks
    _register_all_tasks()
    config = {"num_samples": len(SENTENCES), "min_word_occurrences": 5, "layers": None, **cfg}
    return get_task(TASK)(config=config).evaluate(
        model, tok, dataset=[{"text": t} for t in SENTENCES])


def test_position0_is_dropped_on_a_bos_less_tokenizer(gpt2):
    """gpt2 adds no BOS, so position 0 is the first real word -- the sink. The
    headline drops it; the reference protocol (special tokens only) keeps it
    and tracks the context-free 'Yesterday' as one extra word."""
    model, tok = gpt2
    r = _run(model, tok)
    assert r["_meta_token_policy"] == "exclude_pos0"
    assert r["_meta_word_unit"] == "token_id"
    assert r["_meta_n_position0_occurrences_dropped"] == len(SENTENCES)
    tracked = [v["_meta_n_words_tracked"] for v in r["per_layer"].values()]
    assert r["_meta_paper_protocol_n_words_tracked_mean"] == pytest.approx(
        sum(tracked) / len(tracked) + 1)
    # The context-free word reads SelfSim = 1 at every layer and inflates the
    # reference protocol's mean (known negative).
    assert (r["_meta_paper_protocol_mean_self_similarity_corrected"]
            > r["mean_self_similarity_corrected"] + 1e-3)
