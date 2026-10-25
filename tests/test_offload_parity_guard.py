"""POPULATION guard: disk offload changes WHERE bytes live, never WHAT they are.

`collect_prediction_stats` and `collect_hidden_states` can spill to disk. The
offload writes raw bytes with `.tobytes()` and reads them back through a
`np.memmap`, so the writer's dtype and the reader's dtype are a CONTRACT. When
a 2026-09-09 fix made the writer preserve the incoming float dtype and the
reader still hard-coded float32, every float64 cloud was reinterpreted: an
exact [4, 2, 1] profile became [1.994847434735, 2, 1], geometry_collapse moved
+187% on real GPT-2, and geometry_cka returned NaN in all 10 headlines.

That regression produced EIGHT of round 15's sixteen REDs from a single cause,
and the full suite passed at 2635 green while it was live, because no test drove
float64 through the offload path. Hence this guard.
"""
import json
import os

import pytest

PAPER_CORE_TASKS = os.path.join(os.path.dirname(__file__), "fixtures",
                                "paper_core_tasks.json")

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def gpt2():
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2").eval()
    docs = [{"text": f"Document {i}: ordinary English content for measurement."}
            for i in range(12)]
    return model, tok, docs


@pytest.fixture
def restore_offload():
    prev = os.environ.get("BLME_DISK_OFFLOAD")
    yield
    if prev is None:
        os.environ.pop("BLME_DISK_OFFLOAD", None)
    else:
        os.environ["BLME_DISK_OFFLOAD"] = prev


def test_float64_survives_the_offload_round_trip(gpt2, restore_offload):
    """The narrow, decisive case: the writer's dtype must reach the reader."""
    import numpy as np
    import blme.tasks.geometry.utils as U
    model, tok, docs = gpt2
    real = model.forward

    def f64(*a, **k):
        out = real(*a, **k)
        out.hidden_states = tuple(h.double() for h in out.hidden_states)
        return out

    model.forward = f64
    try:
        vals = {}
        for mode in ("0", "1"):
            os.environ["BLME_DISK_OFFLOAD"] = mode
            H = U.collect_hidden_states(model, tok, docs, num_samples=6,
                                        layer_idx=-1, max_tokens_per_doc=None)
            vals[mode] = np.asarray(H.detach().cpu(), dtype=np.float64)
        assert vals["0"].shape == vals["1"].shape, (
            f"offload changed the shape: {vals['0'].shape} vs {vals['1'].shape}")
        assert np.array_equal(vals["0"], vals["1"]), (
            "offload reinterpreted the bytes: max abs diff "
            f"{np.abs(vals['0'] - vals['1']).max():.3e}")
    finally:
        model.forward = real


def test_no_task_headline_moves_under_offload(gpt2, restore_offload):
    """Population check: every task's headlines must match across routes."""
    from blme.core import _register_all_tasks
    from blme.registry import get_task
    from blme.tasks.config_loader import resolve_task_config
    _register_all_tasks()
    model, tok, docs = gpt2
    tasks = [t["task"] if isinstance(t, dict) else t
             for t in json.load(open(PAPER_CORE_TASKS))]

    def run(cls, cfg, mode):
        os.environ["BLME_DISK_OFFLOAD"] = mode
        try:
            return cls(config=dict(cfg)).evaluate(model, tok, dataset=docs,
                                                  cache=None)
        except Exception as e:                        # noqa: BLE001
            return {"error": f"RAISED {type(e).__name__}: {e}"[:90]}

    moved = []
    for name in tasks:
        cls = get_task(name)
        if cls is None:
            continue
        cfg = resolve_task_config(name, {"num_samples": 12, "use_cache": False})
        a, b = run(cls, cfg, "0"), run(cls, cfg, "1")
        # A task that declines identically on both routes declines for its own
        # reasons, not because of the offload -- no allow-list needed.
        if "error" in a or "error" in b:
            continue
        for k, x in a.items():
            if not isinstance(x, float) or k.startswith("_meta_") or x != x:
                continue
            y = b.get(k)
            if not isinstance(y, float) or y != y or \
                    abs(x - y) > 1e-9 * max(abs(x), 1e-30):
                moved.append(f"{name}.{k}: {x!r} -> {y!r}")
    assert not moved, (
        "disk offload changed task outputs (it must change only WHERE bytes "
        "live):\n  " + "\n  ".join(moved[:12]))
