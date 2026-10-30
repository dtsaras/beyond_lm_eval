"""Precision oracle, dispatch half: a task's numbers do not depend on the
dtype its hidden states are stored in.

The shared cache stores float32, but ``collect_hidden_states`` keeps the
model's dtype, so on a bf16 checkpoint a private route hands bf16 tensors to
the task. The same bf16-exact values are injected stored as bf16, float32 and
float64; every non-_meta_ number must be identical. A bf16 ``.numpy()`` once
crashed procrustes_linearity and trajectory_curvature here.
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import invariance as inv  # noqa: E402
from test_invariance_harness import _DRIVER_CONFIG, TEXTS  # noqa: E402

STORAGE = {"bf16": torch.bfloat16, "fp64": torch.float64}


@pytest.fixture(scope="module")
def bf16_exact_states():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "hf-internal-testing/tiny-random-gpt2"
    model = AutoModelForCausalLM.from_pretrained(name).eval()
    states = inv.states_from_model(model, AutoTokenizer.from_pretrained(name), TEXTS)
    return states.map(lambda x: torch.as_tensor(x).to(torch.bfloat16).to(torch.float64))


def _numbers(d, prefix=""):
    for k, v in d.items():
        if k.startswith("_meta_"):
            continue
        if isinstance(v, dict):
            yield from _numbers(v, f"{prefix}{k}.")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield prefix + k, float(v)
        elif isinstance(v, (list, tuple)) and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            for i, x in enumerate(v):
                yield f"{prefix}{k}[{i}]", float(x)


def _same(a, b):
    return a == b or (math.isnan(a) and math.isnan(b))


@pytest.mark.parametrize("task", sorted(inv.STATE_DRIVABLE_TASKS))
def test_outputs_do_not_depend_on_storage_dtype(task, bf16_exact_states):
    cfg = _DRIVER_CONFIG.get(task)
    ref = dict(_numbers(inv.run_task_on_states(task, bf16_exact_states, cfg,
                                               store_dtype=torch.float32)))
    assert ref, f"{task} returned no numbers"
    for label, dtype in STORAGE.items():
        got = dict(_numbers(inv.run_task_on_states(task, bf16_exact_states, cfg,
                                                   store_dtype=dtype)))
        assert set(got) == set(ref), f"{task} [{label}]: keys differ {sorted(set(got) ^ set(ref))[:5]}"
        moved = [f"{k}: {ref[k]!r} -> {got[k]!r}" for k in ref if not _same(ref[k], got[k])]
        assert not moved, f"{task} [{label} storage] moved:\n" + "\n".join(moved[:10])
