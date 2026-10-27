"""dynamics_generation_diversity: word units, Self-BLEU orders and collection size."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamics_fixtures import inv  # noqa: E402,F401

from blme.tasks.dynamics.generation_diversity import GenerationDiversityTask  # noqa: E402

COMPLETIONS = [
    " The cat sat on the mat and looked at the rain.",
    " A dog ran across the field, barking at the birds.",
    " The cat looked at the dog and then at the mat.",
    " Rain fell on the field all day, and the birds hid.",
    " Nobody knew why the old mill had closed in spring.",
    " The mill stood empty, and the field grew wild.",
]


class _Out:
    pass


class _ScriptedModel(torch.nn.Module):
    """generate() returns the prompt followed by fixed completion strings,
    encoded with the given tokenizer, so two tokenizers emit the same TEXT."""

    def __init__(self, tok):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.tok = tok
        self.calls = 0

    def generate(self, input_ids, max_new_tokens, **kw):
        n, P = input_ids.shape
        eos = self.tok.eos_token_id
        rows = []
        for i in range(n):
            text = COMPLETIONS[(self.calls * n + i) % len(COMPLETIONS)]
            ids = self.tok(text)["input_ids"][:max_new_tokens]
            rows.append(ids + [eos] * (max_new_tokens - len(ids)))
        self.calls += 1
        out = _Out()
        out.sequences = torch.cat([input_ids, torch.tensor(rows)], dim=1)
        V = len(self.tok)
        out.scores = [torch.zeros(n, V) for _ in range(max_new_tokens)]
        return out


def _run(tok_name, cfg):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_name)
    return GenerationDiversityTask(config=dict(cfg)).evaluate(_ScriptedModel(tok), tok, None)


CFG = {"num_prompts": 2, "n_samples_per_prompt": 3, "max_new_tokens": 24, "seed": 0}
WORD_KEYS = ["mean_self_bleu", "distinct_1", "distinct_2", "distinct_3",
             "mean_distinct_1", "mean_distinct_2", "mean_distinct_3"]


def test_self_bleu_reported_at_texygen_orders_2_to_5():
    """Texygen reports Self-BLEU at orders 2-5; each equals Texygen's own
    calc_bleu (one sentence_bleu per order, uniform weights, method1), and
    order 4 is mean_self_bleu."""
    pytest.importorskip("nltk")
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from blme.tasks.dynamics.generation_diversity import _self_bleu_orders, _word_tokenize
    words = [_word_tokenize(t)[0] for t in COMPLETIONS]
    for i, hyp in enumerate(words):
        refs = words[:i] + words[i + 1:]
        got = _self_bleu_orders(hyp, refs)
        for n in (2, 3, 4, 5):
            want = sentence_bleu(refs, hyp, tuple(1.0 / n for _ in range(n)),
                                 smoothing_function=SmoothingFunction().method1)
            assert got[n] == want, (i, n)
    out = _run("gpt2", CFG)
    assert out["_meta_self_bleu_gram"] == 4
    for n in (2, 3, 5):
        assert f"_meta_self_bleu_{n}" in out
    assert out["_meta_self_bleu_2"] > out["_meta_self_bleu_3"] > out["mean_self_bleu"] \
        > out["_meta_self_bleu_5"]


def test_default_collection_is_texygens_500(monkeypatch):
    """With no per-prompt count configured the task generates 500 completions
    (Texygen's sample size) split over the 8 bundled prompts, and Self-BLEU
    scores the first 500. Pre-fix the default was 6 per prompt (36 or 48)."""
    import blme.tasks.dynamics.generation_diversity as gd
    from transformers import AutoTokenizer
    monkeypatch.setattr(gd, "_self_bleu_orders",
                        lambda h, r, orders=gd.SELF_BLEU_ORDERS: {n: 0.0 for n in orders})
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = _ScriptedModel(tok)
    sizes = []
    orig = model.generate

    def gen(input_ids, max_new_tokens, **kw):
        sizes.append(int(input_ids.shape[0]))
        return orig(input_ids, max_new_tokens, **kw)

    model.generate = gen
    out = GenerationDiversityTask(config={"max_new_tokens": 4, "seed": 0}).evaluate(
        model, tok, None)
    assert sizes == [63] * 4 + [62] * 4
    assert out["_meta_n_completions"] == 500
    assert out["_meta_self_bleu_n_scored"] == 500


def test_self_bleu_scores_the_first_sample_size_completions():
    cfg = dict(CFG, self_bleu_sample_size=4)
    out = _run("gpt2", cfg)
    assert out["_meta_n_completions"] == 6 and out["_meta_self_bleu_n_scored"] == 4


def test_word_unit_metrics_do_not_depend_on_the_tokenizer():
    """The same six completions through two tokenizers (gpt2, pythia) give the
    same Self-BLEU and distinct-n over words (Texygen's unit); the subword forms
    (the pre-fix headline) differ."""
    a = _run("gpt2", CFG)
    b = _run("EleutherAI/pythia-160m", CFG)
    assert a["_meta_n_completions"] == b["_meta_n_completions"] == 6
    inv.assert_invariant(a, b, WORD_KEYS, rtol=0.0)
    assert a["_meta_diversity_unit"].startswith(("nltk", "regex"))
    moved = [k for k in ("_meta_mean_self_bleu_subword", "_meta_distinct_1_subword",
                         "_meta_distinct_2_subword") if a[k] != b[k]]
    assert moved, "subword forms should depend on the tokenizer"
