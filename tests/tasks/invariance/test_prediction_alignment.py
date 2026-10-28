"""geometry_prediction_alignment: position-0 pair dropped, centring and the
uniform control over emittable rows, key hygiene -- real gpt2 weights with
injected prediction statistics."""
from __future__ import annotations

import pytest
import torch

from _family_c import TEXTS, inv, load

TASK = "geometry_prediction_alignment"


class _StatsCache:
    """A populated cache that serves fixed prediction statistics."""

    is_populated = True

    def __init__(self, stats):
        self._stats = stats

    def get_prediction_stats(self, num_samples=None):
        return self._stats, None


@pytest.fixture(scope="module")
def gpt2():
    return load("gpt2")


@pytest.fixture(scope="module")
def stats(gpt2):
    from blme.tasks.geometry.utils import collect_prediction_stats
    model, tok = gpt2
    with inv.limit_threads(1):
        st, _ = collect_prediction_stats(model, tok, [{"text": t} for t in TEXTS],
                                         num_samples=len(TEXTS))
    return {"hidden": [h.clone() for h in st["hidden"]],
            "labels": [y.clone() for y in st["labels"]]}


def _run(model, tok, stats):
    from blme.core import _register_all_tasks
    from blme.registry import get_task
    _register_all_tasks()
    with inv.limit_threads(1):
        return get_task(TASK)(config={"num_samples": len(TEXTS)}).evaluate(
            model, tok, dataset=[{"text": t} for t in TEXTS], cache=_StatsCache(stats))


def test_position0_pair_is_dropped(gpt2, stats):
    """The position-0 state is the context-free sink; its 'prediction' of the
    second token is unconditioned. Replacing every document's position-0
    hidden state moves nothing once that pair is dropped."""
    model, tok = gpt2
    base = _run(model, tok, stats)
    g = torch.Generator().manual_seed(0)
    altered = {"hidden": [torch.cat([torch.randn(1, h.shape[1], generator=g) * 50, h[1:]])
                          for h in stats["hidden"]],
               "labels": stats["labels"]}
    other = _run(model, tok, altered)
    inv.assert_invariant(base, other, ["prediction_alignment_mean",
                                       "prediction_alignment_std"], rtol=0.0)
    assert base["_meta_n_position0_pairs_dropped"] == len(TEXTS)
    # _StatsCache serves no hidden states: the masks came from a forward pass.
    assert base["_meta_sink_mask_source"] == "forward_pass"


class _SinkCache(_StatsCache):
    """Prediction statistics plus per-document block outputs (the mask source)."""

    _last_block_substituted = True

    def __init__(self, stats, per_doc):
        super().__init__(stats)
        self._per_doc = per_doc
        self._sample_lengths = [int(x.shape[0]) for x in per_doc[0]]

    def get_hidden_states(self, layer_idx="all", num_samples=None, per_sample=False,
                          replace_final_norm=False, include_embedding=False):
        assert layer_idx == "all" and per_sample
        return {k: list(v) for k, v in self._per_doc.items()}


@pytest.fixture(scope="module")
def cached(gpt2):
    """gpt2's real cache contents: prediction stats and every block output."""
    from blme.cache import ModelOutputCache
    model, tok = gpt2
    cache = ModelOutputCache(model, tok, dataset=[{"text": t} for t in TEXTS],
                             num_samples=len(TEXTS))
    with inv.limit_threads(1):
        cache.populate(need_hidden=True, need_attentions=False, need_logits=True)
    st, _ = cache.get_prediction_stats(num_samples=len(TEXTS))
    per_doc = cache.get_hidden_states(layer_idx="all", num_samples=len(TEXTS),
                                      per_sample=True, replace_final_norm=True)
    return ({"hidden": [h.clone() for h in st["hidden"]],
             "labels": [y.clone() for y in st["labels"]]},
            {k: [x.clone() for x in v] for k, v in per_doc.items()})


POS = 3


def _sink_injected(stats, per_doc):
    """Token 3 of every document is a sink: x1000 at block L/2 (the mask
    source) and, as a sink's final state is, dominated by one direction at
    the read layer (+1000 x its norm on coordinate 0)."""
    mid = len(per_doc) // 2
    pd = {k: [x.clone() for x in v] for k, v in per_doc.items()}
    for x in pd[mid]:
        x[POS] *= 1000.0
    hid = []
    for h in stats["hidden"]:
        h = h.clone()
        h[POS, 0] += 1000.0 * float(h[POS].norm())
        hid.append(h)
    return {"hidden": hid, "labels": stats["labels"]}, pd


def _pair_deleted(stats, per_doc):
    """The control: pair 3 (and position 3's rows) removed from every document."""
    def cut(x):
        return torch.cat([x[:POS], x[POS + 1:]])
    return ({"hidden": [cut(h) for h in stats["hidden"]],
             "labels": [cut(y) for y in stats["labels"]]},
            {k: [cut(x) for x in v] for k, v in per_doc.items()})


def test_massive_source_pair_is_dropped(gpt2, cached, monkeypatch):
    """A first-period-style sink at position 3: massive at block L/2, and a
    final state dominated by one direction. exclude_pos0 scores its pair and
    the margin moves; exclude_sinks drops it, giving the pair-deleted corpus's
    value bit for bit, and with no massive row it equals exclude_pos0 exactly."""
    from blme.tasks.geometry.consistency import PredictionAlignmentTask as P
    model, tok = gpt2
    keys = list(P.HEADLINE_KEYS)

    def run(stats, per_doc, policy=None):
        with monkeypatch.context() as m:
            if policy is not None:
                m.setattr(P, "TOKEN_POLICY", policy)
            with inv.limit_threads(1):
                return P(config={"num_samples": len(TEXTS)}).evaluate(
                    model, tok, dataset=[{"text": t} for t in TEXTS],
                    cache=_SinkCache(stats, per_doc))

    stats, per_doc = cached
    new, old = run(stats, per_doc), run(stats, per_doc, "exclude_pos0")
    assert new["_meta_sink_mask_source"] == "shared_cache"
    assert new["_meta_sink_n_massive_dropped"] == 0
    inv.assert_invariant(old, new, keys, rtol=0.0)

    inj = _sink_injected(stats, per_doc)
    new_inj, old_inj = run(*inj), run(*inj, "exclude_pos0")
    assert new_inj["_meta_n_massive_pairs_dropped"] == len(TEXTS)
    assert new_inj["_meta_massive_token_share"] > 0.5 > new["_meta_massive_token_share"]
    # exclude_pos0 reads the sink pair: std 0.0074 -> 0.0112 on gpt2.
    inv.assert_moves(old, old_inj, "prediction_alignment_std", 0.2)
    # exclude_sinks is the corpus without that pair (on 12 short passages that
    # removal alone moves std by 0.0009).
    inv.assert_invariant(run(*_pair_deleted(stats, per_doc)), new_inj, keys, rtol=0.0)


def test_cache_and_forward_mask_routes_agree(gpt2):
    """The masks from the shared cache's block outputs and from the task's own
    forward pass give the same result on the same passages."""
    from blme.cache import ModelOutputCache
    from blme.tasks.geometry.consistency import PredictionAlignmentTask as P
    model, tok = gpt2
    ds = [{"text": t} for t in TEXTS]
    cache = ModelOutputCache(model, tok, dataset=ds, num_samples=len(TEXTS))
    with inv.limit_threads(1):
        cache.populate(need_hidden=True, need_attentions=False, need_logits=True)
        a = P(config={"num_samples": len(TEXTS)}).evaluate(model, tok, dataset=ds, cache=cache)
        b = P(config={"num_samples": len(TEXTS), "use_cache": False}).evaluate(
            model, tok, dataset=ds)
    assert (a["_meta_sink_mask_source"], b["_meta_sink_mask_source"]) == (
        "shared_cache", "forward_pass")
    inv.assert_invariant(a, b, list(P.HEADLINE_KEYS) + ["_meta_sink_n_massive_dropped"],
                         rtol=1e-6)


def test_unused_vocabulary_rows_move_nothing(gpt2, stats):
    """Appending 64 unused zero rows to the (tied) unembedding used to move the
    centroid and the uniform-row control (dossier: +1.6% headline on gpt2 at
    +10% rows, margin_uniform 0.0213 -> -0.0001); centring and the uniform draw
    now use only the rows the tokenizer can emit."""
    model, tok = gpt2
    base = _run(model, tok, stats)
    padded = _run(inv.pad_vocabulary(model, 64, kind="zero"), tok, stats)
    assert padded["_meta_n_unused_rows"] == 64
    keys = [k for k in base if isinstance(base[k], float)]
    inv.assert_invariant(base, padded, keys, rtol=0.0)


def test_margin_uniform_is_not_a_headline(gpt2, stats):
    r = _run(*gpt2, stats)
    assert "prediction_alignment_margin_uniform" not in r
    assert "_meta_prediction_alignment_margin_uniform" in r


def test_declarations():
    from blme.tasks.geometry.consistency import PredictionAlignmentTask as P
    assert P.TOKEN_POLICY == "exclude_sinks"
    assert set(P.SIZE_INVARIANT) <= set(P.HEADLINE_KEYS)
    assert {"paper", "code", "llm_precedent"} <= set(P.SOURCES)
    assert P.ADAPTATION_NOTE.count(". ") <= 8
