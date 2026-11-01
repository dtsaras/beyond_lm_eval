"""SIZE_INVARIANT of geometry_delta_hyperbolicity: identity blocks leave the
input embeddings untouched and duplicate inner hidden states, so the
embedding value, the last layer and the minimum over layers are unchanged."""
from __future__ import annotations

import pytest

import _family_d as fd

inv = fd.inv

from blme.cache import load_default_corpus  # noqa: E402
from blme.tasks.geometry.delta_hyperbolicity import DeltaHyperbolicityTask  # noqa: E402


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_headline_invariant(arch):
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=3)
    tok = fd.tokenizer()
    data = load_default_corpus(40)
    ids = [tok(t, return_tensors="pt")["input_ids"] for t in fd.TEXTS[:3]]
    deep = inv.insert_identity_blocks(model, [0, 1], verify_on=ids)
    cfg = {"num_samples": 5, "n_points": 24, "min_bytes": 300, "use_dataset": True}
    base = DeltaHyperbolicityTask(cfg).evaluate(model, tok, data)
    ins = DeltaHyperbolicityTask(cfg).evaluate(deep, tok, data)
    assert len(ins["delta_rel_profile"]) == len(base["delta_rel_profile"]) + 2
    inv.assert_invariant(base, ins, sorted(DeltaHyperbolicityTask.SIZE_INVARIANT), rtol=0.0)
