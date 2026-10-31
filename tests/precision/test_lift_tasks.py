"""Every state-drivable paper-core task lifts to float64 under certificate.

The same bf16-exact states run through the task as shipped (float32 storage)
and lifted (float64 storage, every narrowing redirected, ripser replaced by
its GUDHI reference). The certificate proves the lifted run is the shipped
algorithm entirely in float64, so shipped - lifted is the analysis'
arithmetic error; it is reported, and judged against between-model spread
in the precision study, not here.
"""
import json
import os
import sys

import pytest
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
import invariance as inv  # noqa: E402
from precision import lift as L  # noqa: E402
from test_invariance_harness import _DRIVER_CONFIG, TEXTS  # noqa: E402

with open(os.path.join(HERE, "..", "fixtures", "paper_core_tasks.json")) as fh:
    CORE = set(json.load(fh))
TASKS = sorted(t for t in inv.STATE_DRIVABLE_TASKS if t in CORE)


@pytest.fixture(scope="module")
def bf16_exact_states():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "hf-internal-testing/tiny-random-gpt2"
    model = AutoModelForCausalLM.from_pretrained(name).eval()
    states = inv.states_from_model(model, AutoTokenizer.from_pretrained(name), TEXTS)
    return states.map(lambda x: torch.as_tensor(x).to(torch.bfloat16).to(torch.float32))


def test_the_population():
    assert len(TASKS) == 20


@pytest.mark.parametrize("task", TASKS)
def test_the_task_lifts_under_certificate(task, bf16_exact_states):
    cfg = _DRIVER_CONFIG.get(task)
    shipped, lifted, cert = L.shipped_and_lifted(
        lambda: inv.run_task_on_states(task, bf16_exact_states, cfg, store_dtype=torch.float32),
        lambda: inv.run_task_on_states(task, bf16_exact_states, cfg, store_dtype=torch.float64))
    assert cert.ok, cert
    assert set(shipped) == set(lifted)
