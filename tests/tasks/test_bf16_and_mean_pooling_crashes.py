"""bfloat16 storage and ``pooling: mean`` must not crash.

The study runs every model in bfloat16, and numpy has no bfloat16, so any bare
``.numpy()`` on a hidden-state tensor raises ``TypeError``. Three routes did
exactly that: disk offload in ``collect_hidden_states``, and the self-collect
routes of ``geometry_lipschitz`` and ``geometry_rsa``. The default cache widens
to float32 first, which hid them.

``geometry_phd_dimension`` with ``pooling: mean`` called ``.double()`` on numpy
arrays (an earlier narrowing fix written for tensors), raising AttributeError on
both its pooled and per-document routes.

Each test below fails on the pre-fix code.
"""
import os

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

TINY = "hf-internal-testing/tiny-random-gpt2"
DOCS = [{"text": f"The {w} measured the river at dawn and wrote it down."}
        for w in ("surveyor", "engineer", "student", "pilot", "farmer", "doctor",
                  "clerk", "sailor")]


@pytest.fixture(scope="module")
def tiny_bf16():
    tok = transformers.AutoTokenizer.from_pretrained(TINY)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        TINY, dtype=torch.bfloat16).eval()
    return model, tok


@pytest.fixture(scope="module")
def tiny_fp32():
    tok = transformers.AutoTokenizer.from_pretrained(TINY)
    model = transformers.AutoModelForCausalLM.from_pretrained(TINY).eval()
    return model, tok


@pytest.fixture
def restore_offload():
    prev = os.environ.get("BLME_DISK_OFFLOAD")
    yield
    if prev is None:
        os.environ.pop("BLME_DISK_OFFLOAD", None)
    else:
        os.environ["BLME_DISK_OFFLOAD"] = prev


def test_as_numpy_widens_only_bfloat16():
    from blme.tasks.geometry.utils import as_numpy
    b = torch.tensor([1.0, 3.140625, -2.5e-3], dtype=torch.bfloat16)
    assert as_numpy(b).dtype == np.float32
    assert np.array_equal(as_numpy(b), b.float().numpy())
    d = torch.tensor([1.0 + 2 ** -40], dtype=torch.float64)
    assert as_numpy(d).dtype == np.float64 and as_numpy(d)[0] == 1.0 + 2 ** -40


def test_disk_offload_round_trips_bfloat16_bit_exactly(tiny_bf16, restore_offload):
    from blme.tasks.geometry.utils import collect_hidden_states
    model, tok = tiny_bf16
    out = {}
    for mode in ("0", "1"):
        os.environ["BLME_DISK_OFFLOAD"] = mode
        out[mode] = collect_hidden_states(model, tok, DOCS, num_samples=len(DOCS),
                                          layer_idx="all", max_tokens_per_doc=None)
    assert set(out["0"]) == set(out["1"]) and out["0"]
    for k in out["0"]:
        a, b = out["0"][k], out["1"][k]
        assert a.dtype == b.dtype == torch.bfloat16
        assert torch.equal(a, b), f"layer {k}: offload changed the bytes"


@pytest.mark.parametrize("task_name", ["geometry_lipschitz", "geometry_rsa"])
def test_self_collect_on_a_bfloat16_model_matches_the_cache_route(tiny_bf16, task_name):
    """Self-collect (bf16 storage) and the cache (exact float32 widening) carry
    the same values, so every output must be identical, not merely finite."""
    from blme.cache import ModelOutputCache
    from blme.core import _register_all_tasks
    from blme.registry import get_task
    from blme.tasks.config_loader import resolve_task_config
    _register_all_tasks()
    model, tok = tiny_bf16
    cls = get_task(task_name)
    cfg = resolve_task_config(task_name, {"num_samples": len(DOCS)})
    fresh = cls(config=dict(cfg)).evaluate(model, tok, dataset=DOCS, cache=None)
    cache = ModelOutputCache(model, tok, dataset=DOCS, num_samples=len(DOCS))
    cache.populate(need_hidden=True, need_attentions=False, need_logits=False)
    cached = cls(config=dict(cfg)).evaluate(model, tok, dataset=DOCS, cache=cache)
    assert "error" not in fresh, fresh.get("error")
    headline = {k: v for k, v in fresh.items()
                if not k.startswith("_meta_") and isinstance(v, float)}
    assert headline, "no float headline keys to compare"
    for k, v in headline.items():
        w = cached[k]
        assert (np.isnan(v) and np.isnan(w)) or v == w, f"{k}: fresh {v!r} vs cache {w!r}"


def test_phd_mean_pooling_returns_float64_document_means(tiny_fp32):
    from blme.tasks.geometry.phd_dimension import (_gather_point_cloud,
                                                   _iter_per_document_clouds)
    model, tok = tiny_fp32
    pooled = _gather_point_cloud(model, tok, DOCS, num_samples=len(DOCS), max_length=64,
                                 layer_idx=-1, pooling="mean", cache=None, use_cache=False)
    assert pooled is not None and pooled.dtype == np.float64
    assert pooled.shape[0] == len(DOCS)
    per_doc = list(_iter_per_document_clouds(model, tok, DOCS, num_samples=len(DOCS),
                                             max_length=64, layer_idx=-1, pooling="mean"))
    assert per_doc and all(np.asarray(c).dtype == np.float64 for c in per_doc)
    # The pooled vector is the float64 mean of the document's token states.
    with torch.no_grad():
        ids = tok(DOCS[0]["text"], return_tensors="pt", truncation=True, max_length=64)
        h = model(**ids, output_hidden_states=True).hidden_states[-1][0].double().numpy()
    assert np.allclose(pooled[0], h.mean(0), rtol=0, atol=1e-6)


def test_refusal_direction_reports_missing_sklearn_instead_of_raising(tiny_fp32, monkeypatch):
    """Without scikit-learn the task must return an error, as concept
    separability does, not raise NameError deep inside cross-validation."""
    import blme.tasks.repe._common as repe_common
    from blme.tasks.repe import RefusalDirectionTask
    monkeypatch.setattr(repe_common, "HAS_SKLEARN", False)
    for name in ("LogisticRegression", "StratifiedKFold", "roc_auc_score", "accuracy_score"):
        monkeypatch.delattr(repe_common, name, raising=False)
    model, tok = tiny_fp32
    out = RefusalDirectionTask(config={}).evaluate(model, tok, dataset=None)
    assert "error" in out and "scikit-learn" in out["error"]


def test_hubness_duplicate_k_values_do_not_double_count(tiny_fp32):
    """`k_values: [1, 1]` must give exactly the `[1]` statistics.

    The counters are keyed by k but were scattered once per list entry, so a
    repeated k doubled every occurrence count (k1_max 1 -> 2 on a fixture
    whose true maximum is 1).
    """
    from blme.tasks.geometry.hubness import GlobalHubnessTask
    model, tok = tiny_fp32
    one = GlobalHubnessTask(config={"k_values": [1]}).evaluate(model, tok, dataset=None)
    two = GlobalHubnessTask(config={"k_values": [1, 1]}).evaluate(model, tok, dataset=None)
    assert two["_meta_k_values_duplicates_dropped"] == 1
    assert one["_meta_k_values_duplicates_dropped"] == 0
    for k, v in one.items():
        if k.startswith("_meta_k_values"):
            continue
        assert two[k] == v or (v != v and two[k] != two[k]), f"{k}: {v!r} vs {two[k]!r}"
