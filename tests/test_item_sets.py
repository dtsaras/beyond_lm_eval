"""Parallel item sets (BLME_ITEM_SET): set 0 is the default corpus; set 1 is the
same draw on a disjoint pool of the same size."""
import pytest

import blme.cache as cache_mod
from blme.cache import (
    ITEM_SET_ENV, article_records, default_corpus_size, item_set, load_default_corpus,
    load_long_documents, nested_order, passage_split,
)


def _passage_rows(split, n_articles, per_article=5):
    rows = []
    for a in range(n_articles):
        rows.append({"text": f" = {split} article {a} = \n"})
        rows.append({"text": ""})
        for j in range(per_article):
            rows.append({"text": f" {split} passage {a}.{j} " + "word " * 15 + ". \n"})
    return rows


PASSAGES = {"validation": _passage_rows("validation", 6),    # 30 passages
            "test": _passage_rows("test", 8),                # 40 passages
            "train": _passage_rows("train", 40)}             # 200 passages
DOCS = {"train": [{"page": f"\n = Doc {i} = \n " + "tok " * (40 + 30 * (i % 5)) + "\n"} for i in range(60)]}


@pytest.fixture
def wikitext(monkeypatch):
    def load(path, name, split):
        if path == cache_mod.WIKITEXT_DOCUMENTS[0]:
            return DOCS[split]
        assert (path, name) == cache_mod.WIKITEXT_DATASET[:2]
        return PASSAGES[split]

    monkeypatch.setattr(cache_mod, "_load_split", load)
    monkeypatch.delenv(ITEM_SET_ENV, raising=False)
    return monkeypatch


def _texts(rows):
    return [r["text"] for r in rows]


@pytest.mark.parametrize("raw,want", [(None, 0), ("", 0), ("0", 0), (" 1 ", 1)])
def test_item_set_reads_the_environment(monkeypatch, raw, want):
    if raw is None:
        monkeypatch.delenv(ITEM_SET_ENV, raising=False)
    else:
        monkeypatch.setenv(ITEM_SET_ENV, raw)
    assert item_set() == want
    assert passage_split() == ("validation", "test")[want]


@pytest.mark.parametrize("raw", ["2", "-1", "one"])
def test_unknown_item_set_raises(monkeypatch, raw):
    monkeypatch.setenv(ITEM_SET_ENV, raw)
    with pytest.raises(ValueError, match=ITEM_SET_ENV):
        item_set()


def test_set_0_is_the_validation_corpus_in_nested_order(wikitext):
    expected = article_records(PASSAGES["validation"])
    got = load_default_corpus(20, include_document_ids=True)
    assert got == [expected[i] for i in nested_order(len(expected))[:20]]
    wikitext.setenv(ITEM_SET_ENV, "0")
    assert load_default_corpus(20, include_document_ids=True) == got


def test_set_1_draws_test_passages_by_the_same_process(wikitext):
    set0 = load_default_corpus(20, include_document_ids=True)
    wikitext.setenv(ITEM_SET_ENV, "1")
    set1 = load_default_corpus(20, include_document_ids=True)
    expected = article_records(PASSAGES["test"])
    assert set1 == [expected[i] for i in nested_order(len(expected))[:20]]
    assert all(t.startswith("test passage") for t in _texts(set1))
    assert not set(_texts(set0)) & set(_texts(set1))
    # The same filter and stripping, with or without article ids.
    assert _texts(load_default_corpus(20)) == _texts(set1)


def test_every_item_set_has_set_0s_size(wikitext):
    n0 = default_corpus_size()
    assert n0 == 30
    wikitext.setenv(ITEM_SET_ENV, "1")
    assert default_corpus_size() == n0
    assert len(load_default_corpus(default_corpus_size())) == n0


def test_long_documents_set_1_is_the_next_n_of_the_same_order(wikitext):
    set0 = load_long_documents(10, split="train", min_words=100)
    wikitext.setenv(ITEM_SET_ENV, "1")
    set1 = load_long_documents(10, split="train", min_words=100)
    wikitext.delenv(ITEM_SET_ENV)
    both = load_long_documents(20, split="train", min_words=100)
    assert _texts(both) == _texts(set0) + _texts(set1)
    assert len(set1) == 10 and not set(_texts(set0)) & set(_texts(set1))
    assert all(len(t.split()) >= 100 for t in _texts(set1))


def test_long_documents_set_1_raises_when_the_pool_is_short(wikitext):
    assert len(load_long_documents(36, split="train", min_words=100)) == 36   # 36 of 60 qualify
    wikitext.setenv(ITEM_SET_ENV, "1")
    with pytest.raises(ValueError, match="item set 1 needs 48"):
        load_long_documents(24, split="train", min_words=100)


def test_mahalanobis_extended_corpus_set_1_is_disjoint_and_the_same_size(wikitext):
    from blme.tasks.geometry.mahalanobis import _wikitext_passages_extended
    n = 45                                       # validation 30 + test 15 for set 0
    set0 = _wikitext_passages_extended(n)
    val, test = article_records(PASSAGES["validation"]), article_records(PASSAGES["test"])
    assert _texts(set0) == ([val[i]["text"] for i in nested_order(30)]
                            + [test[i]["text"] for i in nested_order(40)[:15]])
    wikitext.setenv(ITEM_SET_ENV, "1")
    set1 = _wikitext_passages_extended(n)
    assert len(set1) == n and not set(_texts(set0)) & set(_texts(set1))
    taken = set(_texts(set0))
    assert _texts(set1)[:25] == [test[i]["text"] for i in nested_order(40) if test[i]["text"] not in taken]
    assert all(r["article_id"].startswith("train:") for r in set1[25:])


def test_dispersion_rows_follow_the_item_set(wikitext):
    from blme.tasks.geometry.dispersion import wikitext_validation_rows
    rows0 = wikitext_validation_rows()
    assert rows0 == [r["text"] for r in PASSAGES["validation"] if r["text"] and not r["text"].isspace()]
    wikitext.setenv(ITEM_SET_ENV, "1")
    assert wikitext_validation_rows() == [r["text"] for r in PASSAGES["test"]
                                          if r["text"] and not r["text"].isspace()]


def test_envelope_records_the_item_set(monkeypatch):
    from blme.results import build_results_envelope
    monkeypatch.delenv(ITEM_SET_ENV, raising=False)
    env = build_results_envelope("pretrained=x", [], {}, {}, "cpu")
    assert env["environment"]["item_set"] == 0
    monkeypatch.setenv(ITEM_SET_ENV, "1")
    assert build_results_envelope("pretrained=x", [], {}, {}, "cpu")["environment"]["item_set"] == 1
