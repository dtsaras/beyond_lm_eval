"""Value tests for two confirmatory keys that had none.

- geometry_spectral.avg_alpha_hat: WeightWatcher's alpha_weighted
  (alpha * log10(lambda_max)) averaged over the same layers, from the live
  weightwatcher package (0.7.7).
- dynamics_sharpness.hutchinson_trace_per_param: the trace estimate divided by
  the number of parameters in the curvature scope.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "invariance"))

import _family_d as fd  # noqa: E402
from blme.tasks.dynamics.sharpness import LossSharpnessTask, _select_parameters  # noqa: E402
from blme.tasks.geometry.spectral import WeightSpectralTask  # noqa: E402


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_avg_alpha_hat_is_weightwatchers_mean_alpha_weighted(arch):
    ww = pytest.importorskip("weightwatcher")
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=2)
    r = WeightSpectralTask({}).evaluate(model, None, None)
    df = ww.WeightWatcher(model=model).analyze()
    df = df[df["layer_type"].astype(str).str.lower() != "embedding"]
    assert len(df) == r["_meta_n_modules_scanned"]
    assert r["avg_alpha_official"] == pytest.approx(float(df["alpha"].mean()), rel=1e-12)
    assert r["avg_alpha_hat"] == pytest.approx(float(df["alpha_weighted"].mean()), rel=1e-12)


def test_trace_per_param_is_the_trace_over_the_scope_size():
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    cfg = {"num_samples": 2, "n_hutchinson": 3, "n_power_iter": 3, "seed": 0}
    r = LossSharpnessTask(cfg).evaluate(model, tok, fd.dataset(2))
    n = sum(p.numel() for p in _select_parameters(model, r["param_scope"]))
    assert r["n_params"] == n
    assert r["hutchinson_trace_per_param"] == r["hutchinson_trace_estimate"] / n
