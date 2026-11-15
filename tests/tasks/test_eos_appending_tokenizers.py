"""Tasks that tokenise privately read the last CONTENT token: a tokenizer that
appends EOS (OLMo-1B-hf, even under truncation) gives the same result as one
that does not. Without the strip, dynamics_interpolation on OLMo-1B read the
EOS state (convexity_gap -0.008 instead of +0.541), and the generation tasks
continued an ended text (generation_diversity mean_self_bleu 0.069 instead of
0.034)."""
import math
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from blme.core import _register_all_tasks  # noqa: E402
from blme.registry import get_task  # noqa: E402

CASES = {
    "dynamics_interpolation": ({"num_samples": 5, "num_pairs": 10}, "convexity_gap"),
    "dynamics_perturbation_growth": ({"num_samples": 3}, "growth_log_ratio_relative"),
    "dynamics_gradient_flow": ({"num_samples": 3}, "gradient_flow_log_ratio_in_out"),
    "geometry_representation_sensitivity": ({"num_samples": 3}, "S_unit"),
    "geometry_positional_decay": ({"num_samples": 3}, "positional_decay_nosink_write_weighted"),
    "geometry_effective_depth": ({"num_samples": 3}, "total_block_influence"),
    "geometry_prediction_alignment": ({"num_samples": 3, "use_cache": False},
                                      "_meta_prediction_alignment_raw_mean"),
    # Seeded sampling: the same prompt ids give the same completions.
    "dynamics_generation_diversity": (
        {"num_prompts": 3, "n_samples_per_prompt": 4, "max_new_tokens": 12, "seed": 0},
        "mean_sampling_entropy_first_quarter"),
    "dynamics_coe": ({"num_samples": 3, "max_new_tokens": 8}, "coe_r"),
    "dynamics_sharpness": ({"num_samples": 3, "n_hutchinson": 4, "n_power_iter": 4},
                           "baseline_loss"),
    "geometry_neural_collapse": ({"min_per_class": 2}, "nc1_cdnv_galanti"),
    "geometry_cknna": ({"num_samples": 12, "topk": 3, "use_cache": False},
                       "mean_offdiag_cknna"),
}
# Tasks that need more than the six default texts.
N_TEXTS = {"geometry_neural_collapse": 12, "geometry_cknna": 12}
# Tasks that record how many appended tokens they removed.
COUNTS_STRIPPED = ("dynamics_generation_diversity", "dynamics_coe", "dynamics_sharpness",
                   "geometry_neural_collapse")


def eos_appending(tok):
    """A copy of ``tok`` whose post-processor appends EOS to every encoding, as
    OLMo-1B-hf's tokenizer does (also under truncation)."""
    import copy
    from tokenizers.processors import TemplateProcessing
    t = copy.deepcopy(tok)
    eos, eid = t.eos_token, t.eos_token_id
    t._tokenizer.post_processor = TemplateProcessing(
        single=f"$A {eos}", pair=f"$A {eos} $B {eos}", special_tokens=[(eos, eid)])
    return t


def test_the_fixture_appends_eos():
    tok = fd.tokenizer()
    ids = eos_appending(tok)("The cat sat.", return_tensors="pt", truncation=True,
                             max_length=16)["input_ids"][0]
    assert int(ids[-1]) == tok.eos_token_id and len(ids) == len(tok("The cat sat.")["input_ids"]) + 1


@pytest.mark.parametrize("task", sorted(CASES))
def test_an_appended_eos_does_not_change_the_result(task):
    _register_all_tasks()
    cfg, key = CASES[task]
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    data = fd.dataset(N_TEXTS.get(task, 6))
    cls = get_task(task)
    plain = cls(dict(cfg)).evaluate(model, tok, data)
    eos = cls(dict(cfg)).evaluate(model, eos_appending(tok), data)
    assert "error" not in plain, plain
    a, b = plain[key], eos[key]
    assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, rel=1e-6, abs=1e-9), (a, b)
    if task in COUNTS_STRIPPED:
        assert plain["_meta_n_trailing_special_stripped"] == 0
        assert eos["_meta_n_trailing_special_stripped"] > 0


@pytest.mark.parametrize("task,cfg", [
    ("dynamics_generation_diversity", {"num_prompts": 2, "n_samples_per_prompt": 2,
                                       "max_new_tokens": 4, "seed": 0}),
    ("dynamics_coe", {"num_samples": 2, "max_new_tokens": 4}),
])
def test_generate_receives_the_same_prompt_ids(task, cfg, monkeypatch):
    """The prompt handed to generate() does not end in the appended EOS."""
    _register_all_tasks()
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    real, seen = model.generate, []

    def spy(*args, **kwargs):
        seen.append(kwargs["input_ids"][0].tolist())
        return real(*args, **kwargs)

    monkeypatch.setattr(model, "generate", spy)
    get_task(task)(dict(cfg)).evaluate(model, tok, fd.dataset(2))
    plain = list(seen)
    seen.clear()
    get_task(task)(dict(cfg)).evaluate(model, eos_appending(tok), fd.dataset(2))
    assert plain and seen == plain
    assert all(ids[-1] != tok.eos_token_id for ids in seen)
