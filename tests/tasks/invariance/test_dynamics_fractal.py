"""dynamics_fractal: natural long documents, the paper's eps and key hygiene."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamics_fixtures import inv, tiny_gpt2, tokenizer  # noqa: E402

import blme.tasks.dynamics.fractal as fr  # noqa: E402

DOC = ("The river rises in the hills north of the town and runs south through "
       "farmland before it joins the sea. ") * 40


def _run(monkeypatch, cfg, dataset=None):
    calls = []

    def fake_long_documents(n, min_tokens=None, tokenizer=None, split="validation"):
        calls.append({"n": n, "min_tokens": min_tokens, "split": split,
                      "tokenizer": tokenizer is not None})
        return [{"text": DOC + f" Article {i}."} for i in range(n)]

    monkeypatch.setattr(fr, "load_long_documents", fake_long_documents)
    with inv.limit_threads(1):
        out = fr.FractalLikelihoodTask(config=dict(cfg)).evaluate(
            tiny_gpt2(), tokenizer(), dataset)
    return out, calls


CFG = {"num_samples": 2, "use_tokens": 400, "burn_in": 32}


def test_documents_are_natural_long_articles(monkeypatch):
    """The task scores whole articles from load_long_documents (>= 4096 tokens,
    in the model's tokenizer), not splices of the passages it is handed."""
    passages = [{"text": f"Short passage number {i} about something else."}
                for i in range(50)]
    out, calls = _run(monkeypatch, CFG, dataset=passages)
    assert "error" not in out, out
    assert calls == [{"n": 2, "min_tokens": 4096, "split": "train", "tokenizer": True}]
    assert out["_meta_document_source"] == "wikitext_document_level:train"
    assert out["_meta_n_documents"] == 2
