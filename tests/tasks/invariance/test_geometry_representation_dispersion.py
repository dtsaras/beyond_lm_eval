"""SIZE_INVARIANT of geometry_representation_dispersion: the final hidden state
of every chunk is bit-identical after inserting identity blocks, so the mean
pairwise cosine distance is too."""
from __future__ import annotations

import pytest

import _family_d as fd

inv = fd.inv

from blme.tasks.geometry.dispersion import RepresentationDispersionTask, wikitext_validation_rows  # noqa: E402


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_headline_invariant(arch):
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=2)
    tok = fd.tokenizer()
    rows = [{"text": t} for t in wikitext_validation_rows()[:200]]
    ids = [tok(t, return_tensors="pt")["input_ids"] for t in fd.TEXTS[:3]]
    deep = inv.insert_identity_blocks(model, [0, 1, 1], verify_on=ids)
    cfg = {"chunk_len": 32, "n_chunks": 24, "use_dataset": True}
    base = RepresentationDispersionTask(cfg).evaluate(model, tok, rows)
    ins = RepresentationDispersionTask(cfg).evaluate(deep, tok, rows)
    assert base["_meta_n_chunks"] == 24
    inv.assert_invariant(base, ins, sorted(RepresentationDispersionTask.SIZE_INVARIANT), rtol=0.0)
