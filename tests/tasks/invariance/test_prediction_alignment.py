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
