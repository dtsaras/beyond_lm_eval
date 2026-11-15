"""Tasks that tokenise privately read the last CONTENT token: a tokenizer that
appends EOS (OLMo-1B-hf, even under truncation) gives the same result as one
that does not. Without the strip, dynamics_interpolation on OLMo-1B read the
EOS state (convexity_gap -0.008 instead of +0.541)."""
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
    "dynamics_perturbation_growth": ({"num_samples": 3}, "growth_log_ratio"),
    "dynamics_gradient_flow": ({"num_samples": 3}, "gradient_flow_log_ratio_in_out"),
    "geometry_representation_sensitivity": ({"num_samples": 3}, "representation_sensitivity"),
    "geometry_positional_decay": ({"num_samples": 3}, "positional_decay_nosink_write_weighted"),
    "geometry_effective_depth": ({"num_samples": 3}, "total_block_influence"),
    "geometry_prediction_alignment": ({"num_samples": 3, "use_cache": False},
                                      "_meta_prediction_alignment_raw_mean"),
}


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
    data = fd.dataset(6)
    cls = get_task(task)
    plain = cls(dict(cfg)).evaluate(model, tok, data)
    eos = cls(dict(cfg)).evaluate(model, eos_appending(tok), data)
    assert "error" not in plain, plain
    a, b = plain[key], eos[key]
    assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, rel=1e-6, abs=1e-9), (a, b)
