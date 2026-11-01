"""SIZE_INVARIANT of consistency_icl_score: both scores read only the model's
output distribution, which inserted identity blocks leave bit-identical."""
from __future__ import annotations

import pytest

import _family_d as fd

inv = fd.inv

from blme.cache import load_long_documents  # noqa: E402
from blme.tasks.consistency.icl_score import ICLScoreTask  # noqa: E402


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_headline_invariant(arch):
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=2)
    tok = fd.tokenizer()
    docs = load_long_documents(3, split="validation")
    ids = [tok(t, return_tensors="pt")["input_ids"] for t in fd.TEXTS[:3]]
    deep = inv.insert_identity_blocks(model, [0, 0, 1], verify_on=ids)
    cfg = {"num_samples": 3, "use_long_documents": False, "early_byte": 100, "late_byte": 800}
    run = lambda m: ICLScoreTask(cfg).evaluate(m, tok, docs)
    base, ins = run(model), run(deep)
    assert base["_meta_n_contexts"] == 3
    inv.assert_invariant(base, ins, sorted(ICLScoreTask.SIZE_INVARIANT), rtol=0.0)
