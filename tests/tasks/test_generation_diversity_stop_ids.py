"""dynamics_generation_diversity: completions are cut at the first of every
id generation stops on (generation-config EOS ids and the tokenizer's EOS),
not only at the tokenizer's EOS."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from blme.tasks.dynamics.generation_diversity import GenerationDiversityTask  # noqa: E402


def test_completions_are_cut_at_every_generation_stop_id():
    """The tiny gpt2's tokenizer EOS is 0 and its generation config stops on
    98; a multi-id stop list (as Llama 3 and Qwen3 ship) makes most rows
    stop. generate() stops on those ids, and the stop token used to stay in
    the completion because only the tokenizer's EOS cut it."""
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    gen_stops = [98] + list(range(100, 300))
    model.generation_config.eos_token_id = gen_stops
    stops = set(gen_stops) | {tok.eos_token_id}
    captured = []
    real = model.generate

    def spy(*a, **k):
        out = real(*a, **k)
        captured.extend(out.sequences[:, k["input_ids"].shape[1]:].tolist())
        return out

    model.generate = spy
    r = GenerationDiversityTask(config={"seed": 0, "num_prompts": 2, "n_samples_per_prompt": 8,
                                        "max_new_tokens": 16}).evaluate(model, tok, None)
    expected = [row[:next((i for i, t in enumerate(row) if t in stops), len(row))]
                for row in captured]
    hit = [row for row in captured if any(t in set(gen_stops) for t in row)]
    assert hit, "no row stopped on a generation-config stop id; the test proves nothing"
    total = sum(len(c) for c in expected)
    assert r["_meta_distinct_1_subword"] == pytest.approx(
        len({t for c in expected for t in c}) / total, rel=1e-12)
    assert r["_meta_stop_ids"] == sorted(stops)
    assert r["_meta_n_completions_cut_at_stop"] == sum(
        any(t in stops for t in row) for row in captured)


def test_model_generation_config_does_not_apply():
    """A model that ships sampling defaults (repetition_penalty 1.3 moved
    distinct_2 from 0.845 to 0.912 on a study model) gives the same result as
    one that ships none: the task samples under its own fresh config."""
    tok = fd.tokenizer()
    cfg = {"seed": 0, "num_prompts": 2, "n_samples_per_prompt": 8, "max_new_tokens": 16}
    plain = GenerationDiversityTask(config=dict(cfg)).evaluate(fd.tiny_gpt2(), tok, None)
    model = fd.tiny_gpt2()
    model.generation_config.repetition_penalty = 1.3
    model.generation_config.no_repeat_ngram_size = 2
    model.generation_config.top_k = 5
    shipped = GenerationDiversityTask(config=dict(cfg)).evaluate(model, tok, None)
    for key in GenerationDiversityTask.HEADLINE_KEYS:
        assert shipped[key] == plain[key], key
    assert plain["_meta_ignored_generation_config"] == {}
    assert shipped["_meta_ignored_generation_config"] == {
        "no_repeat_ngram_size": 2, "repetition_penalty": 1.3, "top_k": 5}
