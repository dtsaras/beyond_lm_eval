"""geometry_representation_sensitivity: the head-linearity guard fails closed.
A guard that cannot run gives NaN headlines with a reason, never the closed
form on an uncertified head."""
import math
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from blme.tasks.geometry.information_geometry import RepresentationSensitivityTask  # noqa: E402


def test_unrunnable_guard_gives_nan_with_reason_through_evaluate():
    """A softcap value the guard cannot parse made the old guard's broad except
    return 0.0 (certified linear) and the closed form ran anyway."""
    model = fd.tiny_gpt2()
    model.config.final_logit_softcapping = "tanh"
    r = RepresentationSensitivityTask(config={"num_samples": 3}).evaluate(
        model, fd.tokenizer(), fd.dataset(3))
    assert "error" not in r, r
    assert math.isnan(r["representation_sensitivity"]), r["representation_sensitivity"]
    assert math.isnan(r["S_unit"])
    assert "linearity test could not run (ValueError" in r["_meta_head_guard_reason"]


def test_guard_returns_nan_and_reason_on_a_tensor_error():
    W = torch.randn(11, 4, generator=torch.Generator().manual_seed(0))
    h = torch.randn(5, 4, generator=torch.Generator().manual_seed(1))
    model = type("M", (), {"config": type("C", (), {})()})()
    bad = torch.nn.Linear(4, 7)      # bias of the wrong length for W
    rel, why = RepresentationSensitivityTask._head_linearity(model, bad, W, h)
    assert math.isnan(rel) and why.startswith("linearity test could not run (RuntimeError")
    good = torch.nn.Linear(4, 11)
    rel, why = RepresentationSensitivityTask._head_linearity(model, good, W, h)
    assert why is None and 0.0 <= rel < 1e-5


def test_linear_head_is_unchanged():
    r = RepresentationSensitivityTask(config={"num_samples": 3}).evaluate(
        fd.tiny_gpt2(), fd.tokenizer(), fd.dataset(3))
    assert math.isfinite(r["representation_sensitivity"]) and "_meta_head_guard_reason" not in r
