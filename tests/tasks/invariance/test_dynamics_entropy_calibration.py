"""SIZE_INVARIANT of dynamics_entropy_calibration: identity blocks leave the
logits bit-identical, so at a fixed seed the samples, the entropies and the
log losses are identical too."""
from __future__ import annotations

import pytest

import _family_d as fd

inv = fd.inv

from blme.cache import load_long_documents  # noqa: E402
from blme.tasks.dynamics.entropy_calibration import EntropyCalibrationTask  # noqa: E402

_CFG = {"num_samples": 3, "prompt_len": 16, "max_new_tokens": 24, "min_response_tokens": 24,
        "min_bytes": 0, "use_dataset": True, "batch_size": 2, "n_bootstrap": 20}


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_headline_invariant(arch):
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=2)
    tok = fd.tokenizer()
    docs = load_long_documents(3, split="validation")
    ids = [tok(t, return_tensors="pt")["input_ids"] for t in fd.TEXTS[:3]]
    deep = inv.insert_identity_blocks(model, [0, 1], verify_on=ids)
    base = EntropyCalibrationTask(_CFG).evaluate(model, tok, docs)
    ins = EntropyCalibrationTask(_CFG).evaluate(deep, tok, docs)
    assert base["_meta_n_examples"] == 3
    inv.assert_invariant(base, ins, sorted(EntropyCalibrationTask.SIZE_INVARIANT), rtol=0.0)


def test_model_generation_config_does_not_apply():
    """Pure sampling at temperature 1 under the task's own config: a model that
    ships decoding defaults (suppress_tokens, repetition_penalty,
    no_repeat_ngram_size) draws the same samples as one that ships none. A
    passed GenerationConfig alone does not do it: generate() fills its unset
    fields from model.generation_config."""
    tok = fd.tokenizer()
    docs = load_long_documents(3, split="validation")
    plain = EntropyCalibrationTask(_CFG).evaluate(fd.tiny_gpt2(), tok, docs)
    model = fd.tiny_gpt2()
    suppressed = list(range(100, 600))
    model.generation_config.suppress_tokens = suppressed
    model.generation_config.repetition_penalty = 1.3
    model.generation_config.no_repeat_ngram_size = 2
    shipped = EntropyCalibrationTask(_CFG).evaluate(model, tok, docs)
    for key in EntropyCalibrationTask.HEADLINE_KEYS + (
            "_meta_mean_generation_entropy", "_meta_n_generated_tokens"):
        assert shipped[key] == plain[key], key
    assert plain["_meta_ignored_generation_config"] == {}
    assert shipped["_meta_ignored_generation_config"] == {
        "no_repeat_ngram_size": 2, "repetition_penalty": 1.3,
        "suppress_tokens": repr(suppressed)}
    # The model's config is restored after the call.
    assert model.generation_config.suppress_tokens == suppressed
    assert model.generation_config.repetition_penalty == 1.3
