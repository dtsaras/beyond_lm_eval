"""geometry_contextualization: sink rows (position 0 and massive tokens)
dropped for every tokenizer, with the reference protocol kept as _meta_ (real
model; the task runs its own forward pass)."""
from __future__ import annotations

import pytest

from _family_c import inv, load

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
    assert r["_meta_token_policy"] == "exclude_sinks"
    assert r["_meta_n_position0_occurrences_dropped"] == len(SENTENCES)
    assert r["_meta_sink_n_massive_dropped"] == 0
    tracked = [v["_meta_n_words_tracked"] for v in r["per_layer"].values()]
    assert r["_meta_paper_protocol_n_words_tracked_mean"] == pytest.approx(
        sum(tracked) / len(tracked) + 1)
    # The context-free word reads SelfSim = 1 at every layer and inflates the
    # reference protocol's mean (known negative).
    assert (r["_meta_paper_protocol_mean_self_similarity_corrected"]
            > r["mean_self_similarity_corrected"] + 1e-3)


def test_declarations(gpt2):
    from blme.tasks.geometry.contextualization import ContextualizationTask as C
    assert C.TOKEN_POLICY == "exclude_sinks"
    assert C.SIZE_INVARIANT == {}
    assert {"paper", "code", "llm_precedent"} <= set(C.SOURCES)
    r = _run(*gpt2)
    assert set(C.HEADLINE_KEYS) == {k for k in r if k.startswith("mean_")}


def _run_policy(model, tok, monkeypatch, texts=SENTENCES, policy=None, hook=None):
    from _sinks import massive_row_hook
    from blme.registry import get_task
    from blme.core import _register_all_tasks
    from blme.tasks.common import get_layers
    _register_all_tasks()
    cls = get_task(TASK)
    config = {"num_samples": len(texts), "min_word_occurrences": 5, "layers": None}
    with monkeypatch.context() as m:
        if policy is not None:
            m.setattr(cls, "TOKEN_POLICY", policy)
        ds = [{"text": t} for t in texts]
        if hook is None:
            return cls(config=config).evaluate(model, tok, dataset=ds)
        with massive_row_hook(model, len(get_layers(model)) // 2, *hook):
            return cls(config=config).evaluate(model, tok, dataset=ds)


def test_sink_token_is_dropped_at_every_layer(gpt2, monkeypatch):
    """A sink at the LAST position (the final ".", which no other token
    attends to): block L/2's output x1000 there. exclude_sinks drops that token
    at every reported layer, so it equals the passages without the final "."
    bit for bit; exclude_pos0 tracks the massive "." and its MEV correction
    flips sign (0.120 -> -0.202 on gpt2). With no massive row the two policies
    agree exactly."""
    from blme.tasks.geometry.contextualization import ContextualizationTask as C
    model, tok = gpt2
    keys = list(C.HEADLINE_KEYS)
    new = _run_policy(model, tok, monkeypatch)
    old = _run_policy(model, tok, monkeypatch, policy="exclude_pos0")
    assert new["_meta_sink_n_massive_dropped"] == 0
    inv.assert_invariant(old, new, keys, rtol=0.0)

    cut = [t[:-1] for t in SENTENCES]
    assert all(t.endswith(".") for t in SENTENCES)
    new_inj = _run_policy(model, tok, monkeypatch, hook=(-1, 1000.0))
    assert new_inj["_meta_n_massive_occurrences_dropped"] == len(SENTENCES)
    assert new_inj["_meta_massive_token_share"] > 0.5 > new["_meta_massive_token_share"]
    inv.assert_invariant(_run_policy(model, tok, monkeypatch, texts=cut), new_inj, keys,
                         rtol=0.0)
    old_inj = _run_policy(model, tok, monkeypatch, policy="exclude_pos0", hook=(-1, 1000.0))
    old_cut = _run_policy(model, tok, monkeypatch, texts=cut, policy="exclude_pos0")
    inv.assert_moves(old_cut, old_inj, "mean_mev_corrected_sizematched", 0.5)


def test_mid_sequence_sink_moves_only_the_old_policy(gpt2, monkeypatch):
    """A sink at position 3 (block L/2 output x1000), which later tokens attend
    to: exclude_pos0's pool-1k MEV correction falls 0.328 -> 0.073 on gpt2;
    exclude_sinks moves 0.328 -> 0.336 (the token removed, and its effect on
    the others through attention)."""
    model, tok = gpt2
    key = "mean_mev_corrected_pool1k"
    new = _run_policy(model, tok, monkeypatch)
    old = _run_policy(model, tok, monkeypatch, policy="exclude_pos0")
    new_inj = _run_policy(model, tok, monkeypatch, hook=(3, 1000.0))
    old_inj = _run_policy(model, tok, monkeypatch, policy="exclude_pos0", hook=(3, 1000.0))
    d_old, d_new = abs(old_inj[key] - old[key]), abs(new_inj[key] - new[key])
    assert d_old > 0.1 and d_new < 0.25 * d_old, (d_new, d_old)
