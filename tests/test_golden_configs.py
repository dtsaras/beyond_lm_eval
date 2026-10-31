"""The golden snapshot's sample-count cap only ever lowers a task's N.

It used to set ``num_samples`` to the cap for every task, so 38 tasks whose
default is below it ran on more samples than they ship with (coe 5 -> 24) and
13 that do not read the key were handed one.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "golden"))
import golden_snapshot as gs  # noqa: E402
from blme.core import _register_all_tasks  # noqa: E402
from blme.registry import _TASK_REGISTRY  # noqa: E402
from blme.tasks.config_loader import resolve_task_config  # noqa: E402

_register_all_tasks()
TASKS = sorted(_TASK_REGISTRY)
CAP = 24


def _n(task, cfg):
    return resolve_task_config(task, cfg).get("num_samples")


@pytest.mark.parametrize("task", TASKS)
def test_the_cap_lowers_and_never_raises(task):
    default = _n(task, {})
    got = _n(task, gs.task_configs_for(task, "shipped", CAP, "cache"))
    assert got == (None if default is None else min(int(default), CAP))


@pytest.mark.parametrize("task", TASKS)
def test_no_cap_passes_the_defaults(task):
    assert gs.task_configs_for(task, "shipped", None, "cache") == {}


def test_the_population_straddles_the_cap():
    # Both branches above are exercised: tasks below, above and without N.
    defaults = [_n(t, {}) for t in TASKS]
    assert any(d is not None and d < CAP for d in defaults)
    assert any(d is not None and d > CAP for d in defaults)
    assert any(d is None for d in defaults)
