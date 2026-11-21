"""The library-wide guards, run on the two exploratory tasks.

geometry_readout_dual_spectrum and dynamics_context_influence are not in
the paper-core fixture, so the population guards that iterate it
(declarations, unusable cache, disk offload) do not reach them. This file
runs the same checks on them, and an end-to-end ``blme.evaluate`` on
tiny-random-gpt2 and gpt2 (CPU).
"""
import math
import os
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import blme  # noqa: E402
import test_task_declarations as decl  # noqa: E402
from blme.core import _register_all_tasks  # noqa: E402
from blme.registry import get_task  # noqa: E402
from blme.tasks.config_loader import resolve_task_config  # noqa: E402

TASKS = ("geometry_readout_dual_spectrum", "dynamics_context_influence")
# Small but real settings: every code path, a few seconds per run.
SMALL = {
    "geometry_readout_dual_spectrum": {"n_rows": 200},
    "dynamics_context_influence": {"documents": "dataset", "num_samples": 3, "window": 128,
                                   "replace_lo": 16, "replace_hi": 32, "n_replacements": 2,
                                   "n_bootstrap": 10},
}
_register_all_tasks()


def test_imports_the_tree_under_test():
    src = Path(__file__).resolve().parents[2] / "src"
    assert Path(blme.__file__).resolve().is_relative_to(src.resolve())


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("check", [
    "test_token_policy_is_a_known_policy",
    "test_selecting_policy_is_read_by_the_task",
    "test_sources_name_paper_code_and_precedent",
    "test_adapted_tasks_carry_a_short_adaptation_note",
    "test_headline_keys_are_reported_keys",
    "test_size_invariant_entries_point_at_real_tests",
])
def test_declarations(task, check):
    getattr(decl, check)(task)


def _docs(n=12, words=300):
    base = ("The river bank is near the old market , and the story of the city "
            "was told at night . ").split()
    return [{"text": " ".join((base * 40)[i:i + words])} for i in range(n)]


@pytest.fixture(scope="module")
def gpt2():
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2").eval()
    return model, tok


def _run(task, model, tok, docs, cache=None):
    cfg = resolve_task_config(task, SMALL[task])
    with torch.no_grad():
        return get_task(task)(config=cfg).evaluate(model, tok, dataset=docs, cache=cache)


def _same(a, b):
    assert "error" not in a and "error" not in b, (a.get("error"), b.get("error"))
    for k in sorted(a.keys() | b.keys()):
        x, y = a.get(k), b.get(k)
        if isinstance(x, float) and math.isnan(x):
            assert isinstance(y, float) and math.isnan(y), k
        else:
            assert x == y, (k, x, y)


@pytest.mark.parametrize("task", TASKS)
def test_an_unusable_cache_changes_nothing(gpt2, task):
    """The POPULATION guard of tests/test_cache_population_guard.py."""
    from blme.cache import ModelOutputCache
    model, tok = gpt2
    docs = _docs()
    cache = ModelOutputCache(model, tok, dataset=docs, num_samples=10)
    cache.populate(need_hidden=False, need_attentions=True, need_logits=False)
    assert cache.is_populated and cache.get_hidden_states(layer_idx=-1, num_samples=10) is None
    _same(_run(task, model, tok, docs), _run(task, model, tok, docs, cache=cache))


@pytest.mark.parametrize("task", TASKS)
def test_disk_offload_changes_nothing(gpt2, task):
    """The POPULATION guard of tests/test_offload_parity_guard.py."""
    model, tok = gpt2
    docs = _docs()
    prev = os.environ.get("BLME_DISK_OFFLOAD")
    try:
        os.environ["BLME_DISK_OFFLOAD"] = "0"
        a = _run(task, model, tok, docs)
        os.environ["BLME_DISK_OFFLOAD"] = "1"
        b = _run(task, model, tok, docs)
    finally:
        if prev is None:
            os.environ.pop("BLME_DISK_OFFLOAD", None)
        else:
            os.environ["BLME_DISK_OFFLOAD"] = prev
    _same(a, b)


@pytest.mark.parametrize("model_name", ["hf-internal-testing/tiny-random-gpt2", "gpt2"])
def test_blme_evaluate_end_to_end(model_name):
    """Both tasks through the library entry point on CPU: the readout task at
    its shipped defaults on the default corpus, the influence task on four
    WikiText-103 train articles."""
    out = blme.evaluate(
        model_args=f"pretrained={model_name}", tasks=list(TASKS), device="cpu",
        task_configs={"dynamics_context_influence": {"num_samples": 4, "n_bootstrap": 20}},
        fail_on_task_error=True)
    res = out["results"]
    for task in TASKS:
        r = res[task]
        assert "error" not in r, r
        for k in get_task(task).HEADLINE_KEYS:
            assert isinstance(r[k], float) and math.isfinite(r[k]), (task, k, r[k])
    m1, m2 = res["geometry_readout_dual_spectrum"], res["dynamics_context_influence"]
    d = m1["_meta_d_model"]
    assert m1["_meta_n_rows"] <= m1["_meta_n_rows_target"] == 32 * d
    assert m2["_meta_pre_replacement_max_js"] == 0.0
    assert m2["_meta_window_tokens"] == (1024 if model_name == "gpt2" else 512)
    if model_name == "gpt2":
        assert m1["_meta_n_target_met"] and m2["_meta_window_is_configured"]
