"""dynamics_gradient_flow: a block called with hidden_states= is captured like
a positional call, and a block called with neither fails loudly instead of
reading as a zero gradient norm."""
import inspect
import math
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from blme.tasks.dynamics.gradient_flow import GradientFlowTask  # noqa: E402


def _all_keywords(block):
    """Call ``block`` with every argument bound by name."""
    sig = inspect.signature(block.forward)

    def call(*a, **k):
        kw = {}
        for name, val in sig.bind(*a, **k).arguments.items():
            if sig.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
                kw.update(val)
            else:
                kw[name] = val
        return block(**kw)
    return call


class _KeywordCalls(torch.nn.ModuleList):
    """The decoder iterates over callers that pass every argument, hidden_states
    included, by keyword (as some remote-code models do); indexing still
    returns the blocks the task hooks."""

    def __iter__(self):
        return (_all_keywords(b) for b in super().__iter__())


def _run(model):
    return GradientFlowTask(config={"num_samples": 4}).evaluate(model, fd.tokenizer(), fd.dataset(4))


def test_keyword_called_blocks_give_the_positional_profile():
    """GPT-2 collects hidden_states itself, so its loop can call blocks by
    keyword (Llama's output-capturing hook in transformers 5 reads args[0])."""
    ref = _run(fd.tiny_gpt2())
    kw_model = fd.tiny_gpt2()
    kw_model.transformer.h = _KeywordCalls(list(kw_model.transformer.h))
    got = _run(kw_model)
    assert all(v > 0 for v in got["gradient_norm_per_layer"]), got["gradient_norm_per_layer"]
    assert got["gradient_norm_per_layer"] == ref["gradient_norm_per_layer"]
    assert math.isfinite(got["gradient_flow_log_ratio_in_out"])
    assert got["gradient_flow_log_ratio_in_out"] == ref["gradient_flow_log_ratio_in_out"]


def test_block_input_reads_positional_then_keyword_and_raises_otherwise():
    from blme.tasks.dynamics.gradient_flow import _block_input
    x, y = torch.zeros(1), torch.ones(1)
    m = torch.nn.Identity()
    assert _block_input(m, (x,), {"hidden_states": y}) is x
    assert _block_input(m, (), {"hidden_states": y}) is y
    with pytest.raises(RuntimeError, match="neither a positional input nor hidden_states="):
        _block_input(m, (), {"x": y})
