"""SIZE_INVARIANT of geometry_predictive_information: identity blocks after
blocks 0 and 1 duplicate existing hidden states, so the profile gains copies
of existing entries and its peak and last entry are unchanged, while the
peak's normalised depth moves."""
from __future__ import annotations

import pytest

import _family_d as fd

inv = fd.inv

from blme.cache import load_default_corpus  # noqa: E402
from blme.tasks.geometry.predictive_information import PredictiveInformationTask  # noqa: E402


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_headline_invariant(arch):
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=3)
    tok = fd.tokenizer()
    data = [{"text": d["text"]} for d in load_default_corpus(80)]
    ids = [tok(t, return_tensors="pt")["input_ids"] for t in fd.TEXTS[:3]]
    deep = inv.insert_identity_blocks(model, [0, 1], verify_on=ids)
    cfg = {"num_samples": 80, "max_length": 64}
    base = PredictiveInformationTask(cfg).evaluate(model, tok, data)
    ins = PredictiveInformationTask(cfg).evaluate(deep, tok, data)
    assert len(ins["predictive_info_profile"]) == len(base["predictive_info_profile"]) + 2
    inv.assert_invariant(base, ins, sorted(PredictiveInformationTask.SIZE_INVARIANT), rtol=0.0)
    assert set(ins["predictive_info_profile"]) == set(base["predictive_info_profile"])
