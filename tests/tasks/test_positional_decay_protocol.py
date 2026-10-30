"""geometry_positional_decay: fp32 at every size (P5), sink-excluded headline
(decision 3b), write-norm-weighted depth aggregation (P4)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))
import _family_d as fd  # noqa: E402

from blme.tasks.geometry.positional_decay import PositionalAttentionDecayTask  # noqa: E402


def _bf16_llama():
    m = fd.tiny_llama(n_layers=2).to(torch.bfloat16)
    for mod in m.modules():                      # bf16 buffers too (RoPE inv_freq)
        for name, buf in list(mod._buffers.items()):
            if buf is not None and buf.is_floating_point():
                mod._buffers[name] = buf.to(torch.bfloat16)
    return m


def _run(model, n=3, **cfg):
    return PositionalAttentionDecayTask({"num_samples": n, **cfg}).evaluate(
        model, fd.tokenizer(), fd.dataset(n))


def test_fp32_at_every_model_size():
    """The old gate left every model above 2e9 parameters in bf16. One extra
    fp32 entry in model.parameters() makes a tiny bf16 model count 3e9."""
    import itertools

    class _Count:                                  # numel only; never computed with
        dtype = torch.float32

        def numel(self):
            return 3_000_000_000

        def is_floating_point(self):
            return True

    m = _bf16_llama()
    real = m.parameters
    m.parameters = lambda recurse=True: itertools.chain(real(recurse), [_Count()])
    assert sum(p.numel() for p in m.parameters()) > 2e9
    res = _run(m)
    assert res["_meta_promoted_to_fp32"] is True
    assert res["_meta_compute_dtype"] == "float32"


def test_buffers_are_handed_back_in_their_dtype():
    """model.float() promoted the bf16 buffers too and only the parameters were
    restored, so later tasks saw an fp32 RoPE table."""
    m = _bf16_llama()
    before = {n: b.dtype for n, b in m.named_buffers()}
    assert torch.bfloat16 in before.values()
    _run(m)
    assert {n: b.dtype for n, b in m.named_buffers()} == before
    assert {p.dtype for p in m.parameters()} == {torch.bfloat16}
