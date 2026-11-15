"""Long-document tasks select their WikiText articles by word count, so every
model scores the same articles whatever its tokenizer.

A token-count threshold made the article list a function of the tokenizer: a
tokenizer that splits words finely admitted short articles that a coarser one
skipped. The fake split below mixes article lengths so that the two tokenizers
disagree under a token threshold (checked first, so the fixture discriminates)
and the real ``evaluate()`` of each task then receives the same list from both.
"""
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import blme.cache as cache_mod  # noqa: E402
import dynamics_fixtures as dfx  # noqa: E402

WORDS = ("photosynthesis", "chlorophyll", "mitochondria", "thermodynamics")
# Words per article by i % 4: under a token threshold the fine tokenizer admits
# the 100- and 1000-word articles where the coarse one does not.
LENGTHS = (100, 600, 1000, 4200)


def _article(i):
    n = LENGTHS[i % len(LENGTHS)]
    body = " ".join(WORDS[j % len(WORDS)] for j in range(n))
    return {"page": f"\n = Article {i} = \n\n {body} \n"}


@pytest.fixture
def fake_documents(monkeypatch):
    def fake_load_dataset(path, name=None, split=None, **kw):
        assert path == "EleutherAI/wikitext_document_level"
        return [_article(i) for i in range(24)]
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)


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


def _task(name):
    if name == "geometry_phd_dimension":
        from blme.tasks.geometry import phd_dimension as mod
        return mod, mod.PHDimensionTask, {"num_samples": 3, "n_reruns": 1}
    if name == "dynamics_fractal":
        from blme.tasks.dynamics import fractal as mod
        return mod, mod.FractalLikelihoodTask, {"num_samples": 2, "use_tokens": 400,
                                                "burn_in": 32}
    from blme.tasks.geometry import positional_decay as mod
    return mod, mod.PositionalAttentionDecayTask, {"num_samples": 3}


def test_min_words_filter_needs_no_tokenizer(fake_documents):
    docs = cache_mod.load_long_documents(6, min_words=1000, split="train")
    assert all(len(d["text"].split()) >= 1000 for d in docs)
    assert cache_mod.load_long_documents(3, min_words=1000, split="train") == docs[:3]
    with pytest.raises(ValueError, match="words"):
        cache_mod.load_long_documents(13, min_words=1000, split="train")


def test_token_threshold_would_split_the_tokenizers(fake_documents, tokenizers):
    """The fixture discriminates: a token threshold gives the two tokenizers
    different article lists."""
    fine, coarse = tokenizers
    for n, k in ((3, 300), (2, 4096)):
        a = cache_mod.load_long_documents(n, min_tokens=k, tokenizer=fine, split="train")
        b = cache_mod.load_long_documents(n, min_tokens=k, tokenizer=coarse, split="train")
        assert a != b


@pytest.mark.parametrize("name", ["geometry_phd_dimension", "dynamics_fractal",
                                  "geometry_positional_decay"])
def test_two_tokenizers_get_the_same_articles(name, fake_documents, tokenizers, model,
                                              monkeypatch):
    mod, cls, cfg = _task(name)
    real = cache_mod.load_long_documents
    seen = []

    def spy(*a, **k):
        docs = real(*a, **k)
        seen[-1].extend(d["text"].split("\n")[0] for d in docs)   # the title
        return docs

    monkeypatch.setattr(cache_mod, "load_long_documents", spy)
    if hasattr(mod, "load_long_documents"):
        monkeypatch.setattr(mod, "load_long_documents", spy)
    outs = []
    for tok in tokenizers:
        seen.append([])
        outs.append(cls(dict(cfg)).evaluate(model, tok, None))
        assert "error" not in outs[-1], outs[-1]
    assert seen[0] and seen[0] == seen[1]
    assert len(seen[0]) == cfg["num_samples"]
