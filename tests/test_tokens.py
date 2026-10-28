"""Token-row selection, sink shares, per-document collection and the cache gate."""
import numpy as np
import pytest
import torch

from blme.tasks._tokens import (
    DocList, POLICIES, hidden_states_for_task, massive_token_share,
    position0_share, select_rows,
)

transformers = pytest.importorskip("transformers")

TINY = "hf-internal-testing/tiny-random-gpt2"


# ── select_rows on synthetic documents ───────────────────────────────────

def _docs():
    g = torch.Generator().manual_seed(0)
    return [torch.randn(3, 2, generator=g), torch.randn(1, 2, generator=g),
            torch.empty(0, 2), torch.randn(4, 2, generator=g)]


def test_all_keeps_every_row_with_its_document_and_position():
    docs = _docs()
    rows, meta = select_rows(docs, "all")
    assert torch.equal(rows, torch.cat(docs))
    assert meta.doc.tolist() == [0, 0, 0, 1, 3, 3, 3, 3]
    assert meta.position.tolist() == [0, 1, 2, 0, 0, 1, 2, 3]
    assert (meta.n_docs, meta.n_docs_used, meta.n_docs_empty,
            meta.n_docs_dropped_pos0_only) == (4, 3, 1, 0)


def test_exclude_pos0_drops_position_zero_and_counts_single_token_docs():
    docs = _docs()
    rows, meta = select_rows(docs, "exclude_pos0")
    assert torch.equal(rows, torch.cat([docs[0][1:], docs[3][1:]]))
    assert meta.doc.tolist() == [0, 0, 3, 3, 3]
    assert meta.position.tolist() == [1, 2, 1, 2, 3]
    assert 0 not in meta.position.tolist()
    # doc 1 has T == 1: it contributes nothing and is counted.
    assert meta.n_docs_dropped_pos0_only == 1
    assert meta.n_docs_empty == 1 and meta.n_docs_used == 2


def test_last_takes_one_row_per_document():
    docs = _docs()
    rows, meta = select_rows(docs, "last")
    assert torch.equal(rows, torch.stack([docs[0][2], docs[1][0], docs[3][3]]))
    assert meta.doc.tolist() == [0, 1, 3]
    assert meta.position.tolist() == [2, 0, 3]


def test_mean_pools_per_document_in_float64():
    docs = _docs()
    rows, meta = select_rows(docs, "mean")
    assert rows.dtype == torch.float64
    exp = np.stack([np.asarray(d, dtype=np.float64).mean(axis=0)
                    for d in (docs[0], docs[1], docs[3])])
    assert np.array_equal(rows.numpy(), exp)
    assert meta.position.tolist() == [-1, -1, -1]
    assert meta.doc.tolist() == [0, 1, 3]

    rows, meta = select_rows(docs, "mean_excl_pos0")
    exp = np.stack([np.asarray(d[1:], dtype=np.float64).mean(axis=0)
                    for d in (docs[0], docs[3])])
    assert np.array_equal(rows.numpy(), exp)
    assert meta.doc.tolist() == [0, 3]
    assert meta.n_docs_dropped_pos0_only == 1


def test_mean_accumulates_in_float64_not_the_input_dtype():
    # 2**24 + 1 + 1 is not representable step by step in float32.
    doc = torch.tensor([[16777216.0], [1.0], [1.0]], dtype=torch.float32)
    assert float(doc.sum(0)[0] / 3) != 16777218.0 / 3  # the float32 trap is real
    rows, _ = select_rows([doc], "mean")
    assert rows.dtype == torch.float64
    assert float(rows[0, 0]) == 16777218.0 / 3


def test_unpooled_policies_keep_the_input_dtype_and_pooled_return_float64():
    docs = [d.to(torch.bfloat16) for d in _docs()]
    for policy in ("all", "exclude_pos0", "last"):
        assert select_rows(docs, policy)[0].dtype == torch.bfloat16
    for policy in ("mean", "mean_excl_pos0"):
        assert select_rows(docs, policy)[0].dtype == torch.float64


def test_explicit_positions_are_used_not_row_order():
    doc = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    pos = [torch.tensor([5, 0, 2])]
    rows, meta = select_rows([doc], "last", positions=pos)
    assert torch.equal(rows, doc[0:1]) and meta.position.tolist() == [5]
    rows, meta = select_rows([doc], "exclude_pos0", positions=pos)
    assert torch.equal(rows, doc[[0, 2]]) and meta.position.tolist() == [5, 2]
    # A DocList carries its positions and doc ids.
    dl = DocList([doc], positions=pos, doc_ids=[7])
    rows, meta = select_rows(dl, "exclude_pos0")
    assert meta.position.tolist() == [5, 2] and meta.doc.tolist() == [7, 7]


def test_never_merges_across_documents():
    docs = [torch.full((4, 3), float(i)) for i in range(5)]
    for policy in ("mean", "mean_excl_pos0", "last"):
        rows, meta = select_rows(docs, policy)
        assert rows.shape == (5, 3)
        assert rows[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_empty_input_and_errors():
    rows, meta = select_rows([], "all")
    assert rows.shape[0] == 0 and meta.n_docs == 0
    rows, meta = select_rows([torch.zeros(1, 4)], "exclude_pos0")
    assert rows.shape == (0, 4) and meta.n_docs_dropped_pos0_only == 1
    with pytest.raises(ValueError):
        select_rows(_docs(), "first")
    with pytest.raises(ValueError):
        select_rows([torch.zeros(2, 3), torch.zeros(2, 4)], "all")
    with pytest.raises(TypeError):
        select_rows(torch.zeros(5, 3), "all")
    with pytest.raises(ValueError):
        select_rows([torch.zeros(2, 3)], "all", positions=[torch.tensor([0])])
    assert set(POLICIES) == {"all", "exclude_pos0", "exclude_sinks", "last", "mean",
                             "mean_excl_pos0", "mean_excl_sinks"}


def test_as_meta_reports_counts():
    meta = select_rows(_docs(), "exclude_pos0")[1].as_meta()
    assert meta == {"_meta_rows_policy": "exclude_pos0", "_meta_rows_n_rows": 5,
                    "_meta_rows_n_docs": 4, "_meta_rows_n_docs_used": 2,
                    "_meta_rows_n_docs_empty": 1,
                    "_meta_rows_n_docs_dropped_pos0_only": 1}


# ── massive-token and position-0 shares ──────────────────────────────────

def _census(docs, factor=10.0):
    """The census formula, written independently on numpy arrays."""
    X = np.concatenate([np.asarray(d, dtype=np.float64) for d in docs])
    pos = np.concatenate([np.arange(len(d)) for d in docs])
    n = np.linalg.norm(X, axis=1)
    med = np.median(n[pos > 0])
    ss = ((X - X.mean(0)) ** 2).sum(1)
    big = (n > factor * med) & (pos > 0)
    return ss[big].sum() / ss.sum(), ss[pos == 0].sum() / ss.sum()


def test_shares_match_the_census_formula():
    g = torch.Generator().manual_seed(1)
    docs = [torch.randn(16, 8, generator=g) for _ in range(6)]
    docs[0][0] *= 300.0          # a position-0 sink
    docs[2][5] *= 50.0           # an in-text massive row
    docs[4][9] *= 40.0
    exp_massive, exp_pos0 = _census(docs)
    assert massive_token_share(docs) == pytest.approx(exp_massive, rel=1e-12)
    assert position0_share(docs) == pytest.approx(exp_pos0, rel=1e-12)
    assert exp_massive > 0.01 and exp_pos0 > 0.5


def test_massive_share_ignores_position_zero_and_is_zero_without_outliers():
    g = torch.Generator().manual_seed(2)
    docs = [torch.randn(10, 4, generator=g) for _ in range(5)]
    for d in docs:
        d[0] *= 1000.0
    assert massive_token_share(docs) == 0.0
    assert position0_share(docs) > 0.99
    assert np.isnan(massive_token_share([torch.zeros(1, 4)]))
    assert np.isnan(position0_share([torch.zeros(3, 4)]))


def test_shares_use_recorded_positions():
    d = torch.ones(3, 2)
    d[1] *= 100.0
    # Row 1 is position 0 here, so it is the sink, not a massive token.
    pos = [torch.tensor([4, 0, 7])]
    assert massive_token_share([d], positions=pos) == 0.0
    assert position0_share([d], positions=pos) == pytest.approx(
        _census_pos([d], pos), rel=1e-12)


def _census_pos(docs, pos):
    X = np.concatenate([np.asarray(d, dtype=np.float64) for d in docs])
    P = np.concatenate([np.asarray(p) for p in pos])
    ss = ((X - X.mean(0)) ** 2).sum(1)
    return ss[P == 0].sum() / ss.sum()


# ── collect_hidden_states(per_sample=True) ───────────────────────────────

@pytest.fixture(scope="module")
def tiny():
    tok = transformers.AutoTokenizer.from_pretrained(TINY)
    model = transformers.AutoModelForCausalLM.from_pretrained(TINY).eval()
    docs = [{"text": f"Document {i} has some ordinary words in it, {'more ' * i}"}
            for i in range(6)]
    docs.insert(2, {"text": ""})       # skipped by the collector
    docs.append({"text": "x"})         # a one-token document
    return model, tok, docs


def _collect(model, tok, docs, **kw):
    from blme.tasks.geometry.utils import collect_hidden_states
    return collect_hidden_states(model, tok, docs, num_samples=len(docs),
                                 replace_final_norm=True, **kw)


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_per_sample_all_layers_concatenates_to_the_flat_return(tiny, offload, dtype):
    import copy
    model, tok, docs = tiny
    m = copy.deepcopy(model).to(dtype)
    flat = _collect(m, tok, docs, layer_idx="all", use_disk_offload=offload,
                    include_embedding=True)
    per = _collect(m, tok, docs, layer_idx="all", use_disk_offload=offload,
                   include_embedding=True, per_sample=True)
    assert type(flat) is dict and all(type(v) is not DocList for v in flat.values())
    assert sorted(per) == sorted(flat)
    for k in flat:
        assert isinstance(per[k], DocList)
        cat = torch.cat(list(per[k]))
        assert cat.dtype == flat[k].dtype == dtype
        assert torch.equal(cat, flat[k])
        assert per[k].doc_ids == [0, 1, 3, 4, 5, 6, 7]
        assert [p.tolist() for p in per[k].positions] == [
            list(range(c.shape[0])) for c in per[k]]
        if offload:
            assert all(c._is_view() for c in per[k])


@pytest.mark.parametrize("offload", [False, True])
def test_per_sample_single_layer_records_subsample_positions(tiny, offload):
    model, tok, docs = tiny
    full = _collect(model, tok, docs, layer_idx=-1, max_tokens_per_doc=None,
                    per_sample=True, use_disk_offload=offload)
    flat = _collect(model, tok, docs, layer_idx=-1, max_tokens_per_doc=4,
                    use_disk_offload=offload)
    sub = _collect(model, tok, docs, layer_idx=-1, max_tokens_per_doc=4,
                   per_sample=True, use_disk_offload=offload)
    assert type(flat) is torch.Tensor and type(sub) is DocList
    assert torch.equal(torch.cat(list(sub)), flat)
    n_layers = model.config.n_layer
    subsampled = 0
    for j, (chunk, pos, did) in enumerate(zip(sub, sub.positions, sub.doc_ids)):
        T = full[j].shape[0]
        if T > 4:
            g = torch.Generator().manual_seed(42 + did * 1009 + (n_layers - 1))
            assert pos.tolist() == torch.randperm(T, generator=g)[:4].tolist()
            subsampled += 1
        else:
            assert pos.tolist() == list(range(T))
        # Each subsampled row IS the full row at its recorded position.
        assert torch.equal(chunk, full[j][pos])
    assert subsampled >= 3
    rows, meta = select_rows(sub, "exclude_pos0")
    assert 0 not in meta.position.tolist()
    kept = sum(int((p != 0).sum()) for p in sub.positions)
    assert rows.shape[0] == kept


def test_default_return_is_unchanged_by_the_new_keywords(tiny):
    model, tok, docs = tiny
    from blme.tasks.geometry.utils import collect_hidden_states
    a = collect_hidden_states(model, tok, docs, num_samples=5, layer_idx=-1,
                              replace_final_norm=False)
    b = collect_hidden_states(model, tok, docs, num_samples=5, layer_idx=-1,
                              replace_final_norm=False, per_sample=False, max_length=512)
    assert type(a) is torch.Tensor and torch.equal(a, b)


def test_max_length_sets_the_window(tiny):
    model, tok, docs = tiny
    long_doc = [{"text": "word " * 100}]
    out = _collect(model, tok, long_doc, layer_idx=-1, max_tokens_per_doc=None,
                   max_length=16)
    assert out.shape[0] == 16


# ── hidden_states_for_task: the cache gate ───────────────────────────────

def _cache(model, tok, docs, n, hidden=True):
    from blme.cache import ModelOutputCache
    c = ModelOutputCache(model, tok, dataset=docs, num_samples=n)
    c.populate(need_hidden=hidden, need_attentions=not hidden, need_logits=False)
    return c


def _equal(a, b):
    if isinstance(a, dict):
        return sorted(a) == sorted(b) and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a.dtype == b.dtype and torch.equal(a, b)


class _Spy:
    """Records what a collector returned, so the test sees the task's tensors."""

    def __init__(self, fn):
        self.fn, self.out = fn, []

    def __call__(self, *a, **k):
        r = self.fn(*a, **k)
        self.out.append(r)
        return r

    def used(self):
        return [r for r in self.out if r is not None][-1]


def _run_task(name, cfg, model, tok, docs, cache, monkeypatch, collector_modules):
    from blme.core import _register_all_tasks
    from blme.registry import get_task
    import blme.cache as cache_mod
    _register_all_tasks()
    spies = []
    for mod in collector_modules:
        s = _Spy(mod.collect_hidden_states)
        monkeypatch.setattr(mod, "collect_hidden_states", s)
        spies.append(s)
    cls_spy = _Spy(cache_mod.ModelOutputCache.get_hidden_states)
    monkeypatch.setattr(cache_mod.ModelOutputCache, "get_hidden_states",
                        lambda self, *a, **k: cls_spy(self, *a, **k))
    spies.append(cls_spy)
    result = get_task(name)(config=dict(cfg)).evaluate(model, tok, dataset=docs, cache=cache)
    monkeypatch.undo()
    # Each route makes exactly one successful read: that is what the task used.
    reads = [r for s in spies for r in s.out if r is not None]
    assert len(reads) == 1, f"{name}: {len(reads)} successful hidden-state reads"
    return result, reads[0]


ROUTES = ["shared_cache", "no_cache", "attention_only_cache", "use_cache_false"]


def _route_cache(route, model, tok, docs):
    """The cache holds every document (7 non-empty), more than the N=6 asked."""
    if route in ("shared_cache", "use_cache_false"):
        return _cache(model, tok, docs, len(docs))
    if route == "attention_only_cache":
        return _cache(model, tok, docs, len(docs), hidden=False)
    return None


@pytest.mark.parametrize("route", ROUTES)
def test_accessor_reproduces_geometry_collapse(tiny, route, monkeypatch):
    import blme.tasks.geometry.collapse as mod
    model, tok, docs = tiny
    n = 6
    cfg = {"num_samples": n, "use_cache": route != "use_cache_false"}
    cache = _route_cache(route, model, tok, docs)
    result, used = _run_task("geometry_collapse", cfg, model, tok, docs, cache,
                             monkeypatch, [mod])
    got, info = hidden_states_for_task(
        model, tok, docs, cache, num_samples=n, layer_idx="all",
        use_cache=cfg["use_cache"], replace_final_norm=True)
    assert _equal(got, used)
    assert info["source"] == ("shared_cache" if route == "shared_cache" else "self_collected")
    assert info["last_is_block"] == result["_meta_last_entry_is_block_output"]


@pytest.mark.parametrize("route", ROUTES + ["undersized_cache"])
def test_accessor_reproduces_geometry_isoscore(tiny, route, monkeypatch):
    import blme.tasks.geometry.isotropy as mod
    model, tok, docs = tiny
    n = 6
    cfg = {"num_samples": n, "use_cache": route != "use_cache_false"}
    cache = (_cache(model, tok, docs, 4) if route == "undersized_cache"
             else _route_cache(route, model, tok, docs))
    _, used = _run_task("geometry_isoscore", cfg, model, tok, docs, cache,
                        monkeypatch, [mod])
    got, info = hidden_states_for_task(
        model, tok, docs, cache, num_samples=n, layer_idx=-1,
        use_cache=cfg["use_cache"], replace_final_norm=False,
        min_cached_docs=n, max_tokens_per_doc=None)
    assert _equal(got, used)
    assert info["source"] == ("shared_cache" if route == "shared_cache" else "self_collected")
    if route == "undersized_cache":
        assert info["cache_rejected"] == "too_few_docs"
        # The comparison can tell the gates apart: ungated, the accessor reads
        # the 3-document cache and no longer matches the task.
        ungated, _ = hidden_states_for_task(
            model, tok, docs, cache, num_samples=n, layer_idx=-1,
            replace_final_norm=False, max_tokens_per_doc=None)
        assert not _equal(ungated, used)


@pytest.mark.parametrize("route", ROUTES)
def test_accessor_reproduces_geometry_trajectory_curvature(tiny, route, monkeypatch):
    model, tok, docs = tiny
    n = 6
    cfg = {"num_samples": n, "use_cache": route != "use_cache_false"}
    cache = _route_cache(route, model, tok, docs)
    _, used = _run_task("geometry_trajectory_curvature", cfg, model, tok, docs,
                        cache, monkeypatch, [])
    got, info = hidden_states_for_task(
        model, tok, docs, cache, num_samples=n, layer_idx="all", per_sample=True,
        use_cache=cfg["use_cache"], replace_final_norm=True, fallback="private_cache")
    assert _equal(got, used)
    assert info["fallback"] == (None if route == "shared_cache" else "private_cache")


def test_accessor_gate_variants(tiny):
    from blme.tasks.geometry.utils import collect_hidden_states
    model, tok, docs = tiny
    c8 = _cache(model, tok, docs, 8)
    common = dict(num_samples=6, replace_final_norm=False)

    _, info = hidden_states_for_task(model, tok, docs, c8, layer_idx=-1, **common)
    assert info["source"] == "shared_cache" and info["cached_docs"] == 7  # "" skipped

    _, info = hidden_states_for_task(model, tok, docs, c8, layer_idx=-1,
                                     min_cached_docs=6, exact_cached_docs=True, **common)
    assert info["cache_rejected"] == "doc_count_mismatch"

    got, info = hidden_states_for_task(model, tok, docs, c8, layer_idx=-1,
                                       max_length=8, **common)
    assert info["cache_rejected"] == "window"
    exp = collect_hidden_states(model, tok, docs, num_samples=6, layer_idx=-1,
                                replace_final_norm=False, max_tokens_per_doc=None,
                                max_length=8)
    assert torch.equal(got, exp)

    _, info = hidden_states_for_task(model, tok, docs, c8, layer_idx=-1,
                                     max_tokens_per_doc=3, **common)
    assert info["cache_rejected"] == "token_cap"

    attn_only = _cache(model, tok, docs, 8, hidden=False)
    got, info = hidden_states_for_task(model, tok, docs, attn_only, layer_idx="all",
                                       fallback="none", **common)
    assert got is None and info["cache_rejected"] == "no_hidden_states"

    with pytest.raises(ValueError):
        hidden_states_for_task(model, tok, None, attn_only, layer_idx="all", **common)
    with pytest.raises(ValueError):
        hidden_states_for_task(model, tok, docs, None, layer_idx="all",
                               fallback="elsewhere", **common)
    with pytest.raises(TypeError):
        hidden_states_for_task(model, tok, docs, c8, num_samples=6)  # no rfn declared
