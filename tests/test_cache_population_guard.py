"""POPULATION guard: an unusable shared cache must never destroy a task.

`is_populated` says a cache HAS been filled, not that it was filled with the
observations a given task needs. A cache populated for an attention-only
consumer returns None from `get_hidden_states`, and a task that dereferences
that None raises, while a task that returns an error instead of self-collecting
silently loses every headline just because another task co-ran.

This class was "swept" in round 12 BY READING THE CODE and reported as 2 sites.
Executing the check across all 51 paper-core tasks put it at 15: four raised
(cka, collapse, lid, hsic), five returned errors after earlier repairs, and six
more (cknna, lipschitz, mp_bulk_deviation, procrustes_linearity, rsa,
trajectory_curvature) ran fine with NO cache and refused with an unusable one.

The guard is therefore executed, not read.
"""
import json

import pytest

transformers = pytest.importorskip("transformers")

@pytest.fixture(scope="module")
def gpt2_and_attention_only_cache():
    from blme.cache import ModelOutputCache
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2").eval()
    docs = [{"text": f"Document {i}: ordinary English content for measurement."}
            for i in range(10)]
    cache = ModelOutputCache(model, tok, dataset=docs, num_samples=10)
    cache.populate(need_hidden=False, need_attentions=True, need_logits=False)
    # FIXTURE INTEGRITY: populated, right document count, no hidden states.
    assert cache.is_populated
    assert cache.get_hidden_states(layer_idx=-1, num_samples=10) is None
    return model, tok, docs, cache


def test_no_task_is_destroyed_by_an_unusable_cache(gpt2_and_attention_only_cache):
    """A task is at fault only if it WORKS with no cache and FAILS with an
    unusable one.

    There is deliberately NO allow-list. A hardcoded set of "legitimate"
    declines is exactly how a real defect gets excused, so the comparison is
    made against each task's own no-cache behaviour instead: a task that
    declines for its own reasons (too few points, wrong attention backend)
    declines identically either way and is not counted, while a task that
    refuses ONLY because the cache is unusable is a defect no matter what its
    error message says.
    """
    from blme.core import _register_all_tasks
    from blme.registry import get_task
    from blme.tasks.config_loader import resolve_task_config
    _register_all_tasks()
    model, tok, docs, cache = gpt2_and_attention_only_cache
    tasks = [t["task"] if isinstance(t, dict) else t
             for t in json.load(open("verification_campaign/tasks.json"))]

    def outcome(name, cls, cfg, c):
        try:
            res = cls(config=dict(cfg)).evaluate(model, tok, dataset=docs, cache=c)
        except Exception as e:                       # noqa: BLE001
            return f"RAISED {type(e).__name__}: {e}"[:110]
        if isinstance(res, dict) and "error" in res:
            return f"ERROR {res['error']}"[:110]
        return "ok"

    broken = []
    for name in tasks:
        cls = get_task(name)
        if cls is None:
            continue
        cfg = resolve_task_config(name, {"num_samples": 10, "use_cache": True})
        with_cache = outcome(name, cls, cfg, cache)
        if with_cache == "ok":
            continue
        without = outcome(name, cls, cfg, None)
        if without == "ok":
            broken.append(f"{name}: no cache -> ok | unusable cache -> {with_cache}")

    assert not broken, (
        "tasks that work WITHOUT a cache but fail with an UNUSABLE one "
        "(the cache is destroying them, not the data):\n  " + "\n  ".join(broken))


def test_the_guard_itself_is_not_vacuous(gpt2_and_attention_only_cache):
    """If the fixture stopped producing an unusable cache the test above would
    pass for the wrong reason, so assert the precondition separately."""
    _model, _tok, _docs, cache = gpt2_and_attention_only_cache
    assert cache.is_populated
    assert cache.get_hidden_states(layer_idx="all", num_samples=10) is None
    assert len(getattr(cache, "_sample_lengths", []) or []) >= 10
