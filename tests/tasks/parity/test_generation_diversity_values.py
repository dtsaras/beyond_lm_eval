"""dynamics_generation_diversity: the inline headline keys through evaluate().

generate() is scripted (as in tests/tasks/invariance/
test_dynamics_generation_diversity.py) to emit fixed completions, and its
step scores keep k_t logits at 0 and mask the rest with -inf, so the sampling
entropy at step t is exactly log k_t. The references are written from the
definitions: corpus Distinct-n (Li et al. 2016) over the word units, the
fraction of a completion's 4-grams that repeat an earlier one, and the first-
and last-quarter means of the step entropies over each completion's own
length.
"""
import math

import numpy as np
import pytest
import torch

from blme.tasks.dynamics.generation_diversity import GenerationDiversityTask, _word_tokenize

COMPLETIONS = [
    " the cat sat on the mat and the cat sat on the mat again",
    " a dog ran and a dog ran and a dog ran far away",
    " rain fell on the field all day and the birds hid",
    " the mill stood empty and the field grew wild",
    " stocks fell sharply as the bank raised rates once more",
    " the cat looked at the dog and then at the mat",
]
MAX_NEW = 24


def _k(t):
    return 1 + (7 * t) % 11            # step t keeps k_t logits: entropy log k_t


class _Scripted(torch.nn.Module):
    def __init__(self, tok):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.tok, self.calls = tok, 0

    def generate(self, input_ids, max_new_tokens, **kw):
        n = input_ids.shape[0]
        eos = self.tok.eos_token_id
        rows = []
        for i in range(n):
            ids = self.tok(COMPLETIONS[(self.calls * n + i) % len(COMPLETIONS)])["input_ids"]
            ids = ids[:max_new_tokens]
            rows.append(ids + [eos] * (max_new_tokens - len(ids)))
        self.calls += 1
        V = len(self.tok)
        scores = []
        for t in range(max_new_tokens):
            s = torch.full((n, V), -float("inf"))
            s[:, :_k(t)] = 0.0
            scores.append(s)
        out = type("Out", (), {})()
        out.sequences = torch.cat([input_ids, torch.tensor(rows)], dim=1)
        out.scores = scores
        return out


def _distinct_corpus(units, n):
    types = {tuple(u[i:i + n]) for u in units for i in range(len(u) - n + 1)}
    return len(types) / sum(len(u) for u in units)


def test_inline_keys_match_their_definitions():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    res = GenerationDiversityTask({"num_prompts": 2, "n_samples_per_prompt": 3,
                                   "max_new_tokens": MAX_NEW, "seed": 0}).evaluate(
        _Scripted(tok), tok, None)
    assert res["_meta_n_completions"] == 6
    ids = [tok(c)["input_ids"] for c in COMPLETIONS]
    words = [_word_tokenize(tok.decode(x, skip_special_tokens=True))[0] for x in ids]

    for n in (1, 2, 3):
        assert res[f"distinct_{n}"] == pytest.approx(_distinct_corpus(words, n), rel=1e-12), n

    rep = []
    for x in ids:
        grams = [tuple(x[j:j + 4]) for j in range(len(x) - 3)]
        rep.append(sum(1 for j, gm in enumerate(grams) if gm in grams[:j]) / len(grams))
    assert max(rep) > 0
    assert res["mean_phrase_repetition_rate"] == pytest.approx(np.mean(rep), rel=1e-12)

    H = np.array([math.log(_k(t)) for t in range(MAX_NEW)])
    q1 = [H[:len(x) // 4].mean() for x in ids]
    q4 = [H[3 * len(x) // 4:len(x)].mean() for x in ids]
    assert res["mean_sampling_entropy_first_quarter"] == pytest.approx(np.mean(q1), rel=1e-6)
    assert res["mean_sampling_entropy_last_quarter"] == pytest.approx(np.mean(q4), rel=1e-6)
    assert res["sampling_entropy_collapse_delta"] == pytest.approx(
        np.mean(np.subtract(q4, q1)), rel=1e-6, abs=1e-7)
    assert abs(np.mean(np.subtract(q4, q1))) > 0.1
