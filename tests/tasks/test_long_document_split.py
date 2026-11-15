"""geometry_schatten and geometry_positional_decay draw their articles from the
WikiText-103 TRAIN split, like every other long-document task.

The document-level validation split holds 60 articles (test: 62), so the old
default ``doc_split: validation`` could not supply the study's 200 articles:
``load_long_documents`` raised and geometry_schatten returned ``{"error": ...}``
for every model (GPU smoke run, 2026-11-16). Train holds about 29,000.

The tests drive the shipped defaults (``resolve_task_config``) through the real
``evaluate()`` on the real, offline-cached dataset.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import blme.cache as cache_mod  # noqa: E402
import dynamics_fixtures as dfx  # noqa: E402
from blme.tasks.config_loader import resolve_task_config  # noqa: E402
from blme.tasks.geometry.positional_decay import PositionalAttentionDecayTask  # noqa: E402
from blme.tasks.geometry.schatten import SchattenNormTask  # noqa: E402

TASKS = {
    "geometry_schatten": (SchattenNormTask, "_meta_schatten_doc_split"),
    "geometry_positional_decay": (PositionalAttentionDecayTask, "_meta_doc_split"),
}
STUDY_N = 200          # beyond_benchmarks configs/tasks.yaml: geometry_schatten
SEQ_TOKENS = 16        # short windows keep 200 forwards fast


@pytest.fixture(scope="module")
def real_documents():
    """Skip (not fail) where the document-level corpus is not cached."""
    try:
        cache_mod.load_long_documents(1, split="train")
    except RuntimeError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"wikitext_document_level not available: {e}")


@pytest.fixture(scope="module")
def tokenizers():
    fine = dfx.tokenizer()                         # tiny-random-gpt2: 1000-token BPE
    try:
        coarse = transformers.AutoTokenizer.from_pretrained("gpt2")
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"gpt2 tokenizer not cached: {e}")
    return fine, coarse


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257,
                                  n_positions=1024, attn_implementation="eager")
    return transformers.GPT2LMHeadModel(cfg).eval()


def _evaluate(name, model, tokenizer, monkeypatch, dataset=None, **overrides):
    """The task's real evaluate() on its shipped defaults, plus the loader calls it made."""
    cls, _ = TASKS[name]
    real = cache_mod.load_long_documents
    calls = []

    def spy(*a, **k):
        docs = real(*a, **k)
        calls.append({"split": k.get("split"), "min_words": k.get("min_words"),
                      "titles": [d["text"].split("\n")[0] for d in docs]})
        return docs

    with monkeypatch.context() as mp:
        mp.setattr(cache_mod, "load_long_documents", spy)
        out = cls(resolve_task_config(name, overrides)).evaluate(model, tokenizer, dataset)
    return out, calls


@pytest.mark.parametrize("name", sorted(TASKS))
def test_the_study_sample_count_runs(name, real_documents, model, tokenizers, monkeypatch):
    """200 articles (more than validation and test hold together) score without error."""
    cls, split_key = TASKS[name]
    out, calls = _evaluate(name, model, tokenizers[1], monkeypatch,
                           num_samples=STUDY_N, seq_tokens=SEQ_TOKENS)
    assert "error" not in out, out
    for key in cls.HEADLINE_KEYS:
        assert math.isfinite(out[key]), (key, out[key])
    assert len(calls) == 1 and calls[0]["split"] == "train"
    assert len(calls[0]["titles"]) == len(set(calls[0]["titles"])) == STUDY_N
    assert out[split_key] == "train"


def test_articles_do_not_depend_on_the_tokenizer_and_nest(real_documents, model, tokenizers,
                                                          monkeypatch):
    """Both tokenizers, and both tasks, read one article list; a smaller draw is its prefix."""
    fine, coarse = tokenizers
    lists = {}
    # 61 is one more article than the validation split holds.
    for name, n in (("geometry_schatten", STUDY_N), ("geometry_positional_decay", 61)):
        for label, tok in (("fine", fine), ("coarse", coarse)):
            out, calls = _evaluate(name, model, tok, monkeypatch,
                                   num_samples=n, seq_tokens=SEQ_TOKENS)
            assert "error" not in out, out
            lists[name, label] = calls[0]["titles"]
    reference = lists["geometry_schatten", "coarse"]
    assert len(reference) == len(set(reference)) == STUDY_N
    assert lists["geometry_schatten", "fine"] == reference
    assert lists["geometry_positional_decay", "fine"] == reference[:61]
    assert lists["geometry_positional_decay", "coarse"] == reference[:61]
    for name, n in (("geometry_schatten", 10), ("geometry_positional_decay", 5)):
        out, calls = _evaluate(name, model, fine, monkeypatch,
                               num_samples=n, seq_tokens=SEQ_TOKENS)
        assert "error" not in out, out
        assert calls[0]["titles"] == reference[:n]


@pytest.mark.parametrize("name", sorted(TASKS))
def test_shipped_default_is_the_train_split(name):
    assert resolve_task_config(name)["doc_split"] == "train"
    assert resolve_task_config(name)["documents"] == "long"


@pytest.mark.parametrize("name", sorted(TASKS))
def test_a_short_split_is_an_error_not_a_smaller_sample(name, real_documents, model,
                                                        tokenizers, monkeypatch):
    """An explicit doc_split is honoured and never topped up or truncated: validation
    holds 60 articles, so 60 run on it and 61 are an error that names the split."""
    _, split_key = TASKS[name]
    out, calls = _evaluate(name, model, tokenizers[1], monkeypatch, num_samples=60,
                           seq_tokens=SEQ_TOKENS, doc_split="validation")
    assert "error" not in out, out
    assert calls[0]["split"] == "validation" and len(calls[0]["titles"]) == 60
    assert out[split_key] == "validation"

    out, calls = _evaluate(name, model, tokenizers[1], monkeypatch, num_samples=61,
                           seq_tokens=SEQ_TOKENS, doc_split="validation")
    assert set(out) == {"error"} and not calls
    assert "61 articles requested" in out["error"] and "validation has 60" in out["error"]
    assert "train split" in out["error"]


@pytest.mark.parametrize("name", sorted(TASKS))
def test_documents_dataset_reads_no_split(name, model, tokenizers, monkeypatch):
    """documents: dataset scores the given passages; no article split is loaded or recorded."""
    _, split_key = TASKS[name]
    data = [{"text": t} for t in dfx.TEXTS[:3]]
    out, calls = _evaluate(name, model, tokenizers[1], monkeypatch, dataset=data,
                           num_samples=3, documents="dataset")
    assert "error" not in out, out
    assert not calls and out[split_key] is None
