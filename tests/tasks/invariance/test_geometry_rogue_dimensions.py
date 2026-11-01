"""SIZE_INVARIANT of geometry_rogue_dimensions: identity blocks duplicate
existing hidden states, so the profiles gain copies of existing entries and
their maxima and last entries are unchanged."""
from __future__ import annotations

import pytest

import _family_d as fd

inv = fd.inv

from blme.cache import load_long_documents  # noqa: E402
from blme.tasks.geometry.rogue_dimensions import RogueDimensionsTask  # noqa: E402


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_headline_invariant(arch):
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=3)
    tok = fd.tokenizer()
    docs = load_long_documents(4, split="validation")
    ids = [tok(t, return_tensors="pt")["input_ids"] for t in fd.TEXTS[:3]]
    deep = inv.insert_identity_blocks(model, [0, 1, 1], verify_on=ids)
    cfg = {"num_samples": 4, "context_length": 32, "use_dataset": True}
    base = RogueDimensionsTask(cfg).evaluate(model, tok, docs)
    ins = RogueDimensionsTask(cfg).evaluate(deep, tok, docs)
    assert len(ins["rogue_top1_share_profile"]) == len(base["rogue_top1_share_profile"]) + 3
    inv.assert_invariant(base, ins, sorted(RogueDimensionsTask.SIZE_INVARIANT), rtol=0.0)
