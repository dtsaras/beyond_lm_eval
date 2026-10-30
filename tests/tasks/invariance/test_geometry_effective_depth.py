"""SIZE_INVARIANT of geometry_effective_depth: sum and max of Block Influence.

An identity block maps x to x, so its BI = 1 - cos(x, x) is 0 (to float32
rounding of a unit vector's squared norm): the sum over blocks and the
maximum are unchanged by inserting one, while the mean is diluted by L0 / L
and effective_depth_90 (k90 / L, a capacity ratio) falls by design.
"""
from __future__ import annotations

import pytest

import _family_d as fd

inv = fd.inv

from blme.tasks.geometry.effective_depth import EffectiveDepthTask  # noqa: E402


def _run(model):
    return EffectiveDepthTask({"num_samples": 4}).evaluate(model, fd.tokenizer(), fd.dataset(4))


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_depth_additive_keys_under_identity_blocks(arch):
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=3)
    tok = fd.tokenizer()
    ids = [tok(t, return_tensors="pt")["input_ids"] for t in fd.TEXTS[:3]]
    deep = inv.insert_identity_blocks(model, [0, 1, 1, 2], verify_on=ids)
    base, ins = _run(model), _run(deep)
    assert ins["_meta_n_layers_profiled"] == base["_meta_n_layers_profiled"] + 4
    inv.assert_invariant(base, ins, sorted(EffectiveDepthTask.SIZE_INVARIANT), rtol=1e-6)
    # Known negatives: the depth-diluted mean, and the capacity ratio.
    n0, n1 = base["_meta_n_layers_profiled"], ins["_meta_n_layers_profiled"]
    assert ins["_meta_mean_block_influence"] == pytest.approx(
        base["_meta_mean_block_influence"] * n0 / n1, rel=1e-5)
    inv.assert_moves(base, ins, "_meta_mean_block_influence", min_rel=0.1)
    # Appended identity blocks leave k90 in place, so the ratio falls by L0 / L1.
    tail = inv.insert_identity_blocks(model, [n0 - 1, n0 - 1], verify_on=ids)
    app = _run(tail)
    assert app["effective_depth_90"] == pytest.approx(
        base["effective_depth_90"] * n0 / (n0 + 2), rel=1e-12)
