"""geometry_matrix_entropy: the cache window gate, the gated float64 row
normalisation, key hygiene and the size-invariant raw entropy."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from _family_c import TEXTS, inv, load

TASK = "geometry_matrix_entropy"


@pytest.fixture(scope="module")
def tiny():
    return load("hf-internal-testing/tiny-random-gpt2")


def test_cache_route_honours_max_length(tiny):
    """The shared cache tokenises at 512; with max_length 8 (which truncates
    every TEXTS document) the task must self-collect at 8 tokens and match
    its own no-cache run, not read the 512-token cache."""
    model, tok = tiny
    cfg = {"max_length": 8}
    via_cache = inv.run_task_on_model(TASK, model, tok, TEXTS, cfg, use_cache=True)
    no_cache = inv.run_task_on_model(TASK, model, tok, TEXTS, {**cfg, "use_cache": False})
    assert via_cache["matrix_entropy"] == no_cache["matrix_entropy"]
    assert via_cache["_meta_matrix_entropy_source"] == "self_collected"
    assert via_cache["_meta_matrix_entropy_max_length"] == 8
    full = inv.run_task_on_model(TASK, model, tok, TEXTS, {}, use_cache=True)
    assert full["_meta_matrix_entropy_source"] == "shared_cache"
    assert full["matrix_entropy"] != via_cache["matrix_entropy"]
