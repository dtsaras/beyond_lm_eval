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


def test_rows_spanning_50_orders_survive_row_normalisation():
    """Wei et al. normalise every row, so the centred row-normalised
    covariance of the fixture is exactly I_2 / 2 and H / ln 2 = 1 (both
    released implementations return 1.0). The global power-of-two rescale
    and float32 cast used to flush the 1e-25 rows, leaving two collinear rows
    and 9.97e-29. The float64 row path runs only when rows were lost."""
    from adversarial_fixtures import wide_row_magnitude_span
    from blme.tasks.geometry.matrix_entropy import _matrix_entropy
    for label, X, want in wide_row_magnitude_span():
        got = _matrix_entropy(torch.tensor(X, dtype=torch.float32))
        assert got["entropy_normalized"] == pytest.approx(want, rel=1e-6), label
        assert got["n_rows_rescued"] == 2
    ordinary = _matrix_entropy(torch.randn(24, 16, generator=torch.Generator().manual_seed(0)))
    assert ordinary["n_rows_rescued"] == 0
