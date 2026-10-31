"""A task leaves the global RNGs (random, numpy, torch) as it found them.

categories and refusal reseeded the global numpy / Python stream and
generation_diversity and self_consistency the global torch one, so every task
drawing from those after them in the same evaluate() got a fixed stream
whatever the run seed. They now draw from local streams that reproduce the
same values, or fork the global one around generate().
"""
import random

import numpy as np
import pytest
import torch

from test_seed_offset import RUNNERS, _texts, _tiny


def _direct(task, texts, base=None):
    """``evaluate()`` called directly: the test drivers seed the global RNGs."""
    def run(extra):
        from blme.core import _register_all_tasks
        from blme.registry import get_task
        from blme.tasks.config_loader import resolve_task_config
        _register_all_tasks()
        model, tok = _tiny()
        cfg = resolve_task_config(task, {**(base or {}), **extra, "use_cache": False})
        with torch.no_grad():
            return get_task(task)(config=cfg).evaluate(
                model, tok, dataset=[{"text": t} for t in texts])
    return run


CASES = {
    "geometry_categories": _direct("geometry_categories", _texts(2)),
    "repe_refusal_direction": RUNNERS["repe_refusal_direction"],
    "dynamics_generation_diversity": _direct(
        "dynamics_generation_diversity", _texts(2),
        {"num_prompts": 2, "n_completions": 4, "max_new_tokens": 6, "self_bleu_sample_size": 4}),
    "consistency_self_consistency": _direct(
        "consistency_self_consistency", _texts(2),
        {"num_prompts": 2, "n_samples_per_prompt": 3, "max_new_tokens": 4}),
}


def _states():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


@pytest.mark.parametrize("task", sorted(CASES))
def test_the_global_rng_is_left_alone(task):
    _tiny()  # built (and seeded) once, outside the measured window
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    before = _states()
    res = CASES[task]({})
    assert isinstance(res, dict) and "error" not in res, res
    after = _states()
    assert after[0] == before[0], "python random moved"
    assert all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
               for a, b in zip(after[1], before[1])), "numpy global RNG moved"
    assert torch.equal(after[2], before[2]), "torch global RNG moved"
