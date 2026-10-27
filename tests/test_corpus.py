"""The default corpus loader: filter, strip, nested evenly spread order, errors."""
import math
import sys

import pytest

import blme.cache as cache_mod
from blme.cache import (
    MIN_PASSAGE_CHARS, default_corpus_size, load_default_corpus,
    load_long_documents, nested_order,
)


# ── nested_order ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 16, 64, 100, 1000, 1691])
def test_nested_order_is_a_permutation_with_evenly_spread_prefixes(n):
    order = nested_order(n)
    assert sorted(order) == list(range(n))
    assert order[0] == 0
    for m in range(1, n + 1):
        picked = sorted(order[:m])
        gaps = [b - a for a, b in zip(picked, picked[1:])] + [n - picked[-1]]
        assert max(gaps) <= math.ceil(2 * n / m), (n, m, max(gaps))


@pytest.mark.parametrize("n", [7, 100, 1691])
def test_power_of_two_prefixes_are_the_stride_sample(n):
    order = nested_order(n)
    j = 1
    while (1 << j) <= n:
        m = 1 << j
        assert sorted(order[:m]) == [i * n // m for i in range(m)]
        j += 1


def test_nested_order_edge_cases():
    assert nested_order(0) == [] and nested_order(-3) == []
    assert nested_order(1) == [0]


# ── load_default_corpus on a synthetic split ─────────────────────────────

def _fake_rows(n_good=40):
    rows = []
    for i in range(n_good):
        rows.append({"text": f" = Heading {i} = \n"})                      # short
        rows.append({"text": ""})
        rows.append({"text": f" Passage {i:03d} " + "word " * 15 + "end . \n"})
    rows.append({"text": " " * 80 + "\n"})                                   # blank
    rows.append({"text": "x" * (MIN_PASSAGE_CHARS - 1) + "   \n"})           # 63 after strip
    rows.append({"text": "\n  " + "y" * MIN_PASSAGE_CHARS + " \n"})          # 64 after strip
    rows.append({"other": "no text field"})
    return rows


@pytest.fixture
def fake_split(monkeypatch):
    calls = []

    def fake_load_dataset(path, name=None, split=None, **kw):
        calls.append((path, name, split, kw))
        if path == "EleutherAI/wikitext_document_level":
            return [{"page": f"\n = Article {i} = \n\n " + "tok " * (20 * (i % 7) + 5) + "\n"}
                    for i in range(30)]
        return _fake_rows()

    datasets = pytest.importorskip("datasets")
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return calls


def test_filter_and_strip(fake_split):
    out = load_default_corpus(default_corpus_size(), diverse=False)
    texts = [d["text"] for d in out]
    assert len(texts) == 41                      # 40 passages + the 64-char row
    assert all(t == t.strip() and len(t) >= MIN_PASSAGE_CHARS for t in texts)
    assert not any(t.endswith("\n") for t in texts)
    assert "y" * MIN_PASSAGE_CHARS in texts and not any(t.startswith("x") for t in texts)
    path, name, split, kw = fake_split[0]
    assert (path, name, split) == ("Salesforce/wikitext", "wikitext-103-raw-v1", "validation")
    assert "trust_remote_code" not in kw


def test_draws_nest_spread_and_are_deterministic(fake_split):
    full = load_default_corpus(41)
    for n in (0, 1, 5, 16, 30, 41):
        draw = load_default_corpus(n)
        assert draw == full[:n]
        assert draw == load_default_corpus(n)
    pool = [d["text"] for d in load_default_corpus(41, diverse=False)]
    picked = sorted(pool.index(d["text"]) for d in load_default_corpus(16))
    assert picked == [i * 41 // 16 for i in range(16)]
    # The default is the spread draw, not the split-order prefix.
    assert load_default_corpus(8) != load_default_corpus(8, diverse=False)
    assert load_default_corpus(8, diverse=False) == [{"text": t} for t in pool[:8]]


def test_too_many_passages_raise(fake_split):
    with pytest.raises(ValueError, match="42 passages requested"):
        load_default_corpus(42)
    with pytest.raises(ValueError):
        load_default_corpus(-1)


def test_loader_failure_raises_instead_of_falling_back(monkeypatch):
    datasets = pytest.importorskip("datasets")

    def boom(*a, **k):
        raise ConnectionError("offline")

    monkeypatch.setattr(datasets, "load_dataset", boom)
    with pytest.raises(RuntimeError, match="Salesforce/wikitext") as ei:
        load_default_corpus(10)
    assert isinstance(ei.value.__cause__, ConnectionError)
    with pytest.raises(RuntimeError):
        load_long_documents(3)


def test_missing_datasets_package_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(RuntimeError, match="datasets"):
        load_default_corpus(3)


def test_no_hardcoded_fallback_text_remains():
    import inspect
    src = inspect.getsource(cache_mod)
    assert "quick brown fox" not in src and "FALLBACK CORPUS" not in src


# ── load_long_documents on a synthetic split ─────────────────────────────

class _WordTokenizer:
    def __call__(self, text, **kw):
        return {"input_ids": text.split()}


def test_long_documents_nest_strip_and_filter_by_tokens(fake_split):
    # 5 of the 30 fake articles are under MIN_PASSAGE_CHARS once stripped.
    all_docs = load_long_documents(25)
    with pytest.raises(ValueError, match="25"):
        load_long_documents(26)
    assert all(d["text"] == d["text"].strip() for d in all_docs)
    assert load_long_documents(10) == all_docs[:10]
    tok = _WordTokenizer()
    long_ = load_long_documents(8, min_tokens=60, tokenizer=tok)
    assert all(len(d["text"].split()) >= 60 for d in long_)
    assert load_long_documents(4, min_tokens=60, tokenizer=tok) == long_[:4]
    # The unfiltered nested order with the short articles skipped.
    assert [d for d in all_docs if len(d["text"].split()) >= 60][:8] == long_
    with pytest.raises(ValueError, match="tokens"):
        load_long_documents(25, min_tokens=60, tokenizer=tok)
    with pytest.raises(ValueError, match="tokenizer"):
        load_long_documents(3, min_tokens=60)
    assert fake_split[-1][:3] == ("EleutherAI/wikitext_document_level",
                                  "wikitext-103-raw-v1", "validation")


# ── the real split (skipped when it is not in the local HF cache) ────────

@pytest.fixture(scope="module")
def real_size():
    try:
        return default_corpus_size()
    except RuntimeError as e:
        pytest.skip(f"WikiText-103 not available: {e}")


def test_real_split(real_size):
    assert real_size == 1691
    full = load_default_corpus(real_size)
    assert load_default_corpus(200) == full[:200]
    assert all(d["text"] == d["text"].strip() for d in full)
    assert all(len(d["text"]) >= MIN_PASSAGE_CHARS for d in full)
    assert len({d["text"] for d in full}) >= real_size - 5   # a few exact repeats at most
    with pytest.raises(ValueError):
        load_default_corpus(real_size + 1)


def test_real_long_documents(real_size):
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    docs = load_long_documents(12, min_tokens=300, tokenizer=tok)
    assert len(docs) == 12
    assert all(len(tok(d["text"], verbose=False)["input_ids"]) >= 300 for d in docs)
    assert load_long_documents(5, min_tokens=300, tokenizer=tok) == docs[:5]
