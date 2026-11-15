"""geometry_lm_head_free_energy where the float32 spectrum loses the sign of
the smallest eigenvalue.

GPU smoke (gpt2 in bfloat16 on an RTX 3090): the headline was NaN with no
reason while the float64 value was 1.7264 and the smallest float64 eigenvalue
0.0204. The float32 rounding of W.T @ W (eps32 * largest eigenvalue = 0.026
for gpt2) exceeds that eigenvalue, so its sign follows the BLAS/LAPACK in
use. The float32 analysis now runs on the CPU, and a float32 spectrum with a
non-positive eigenvalue over a float64 spectrum that is positive beyond its
rounding reports the float64 value, flagged; every NaN carries a reason.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from blme.tasks.common import get_lm_head  # noqa: E402
from blme.tasks.geometry import lm_head_free_energy as FE  # noqa: E402

REASON = "_meta_lm_head_bfe_w_nan_reason"
FLAG = "_meta_lm_head_bfe_w_from_float64"


def _run(model):
    return FE.LMHeadFreeEnergyTask({}).evaluate(model, None, None)


def _with_head(W):
    """Tiny Llama whose unembedding is ``W`` (vocab, d)."""
    model = fd.tiny_llama(n_layers=2)
    head = get_lm_head(model).weight
    assert tuple(head.shape) == tuple(W.shape)
    with torch.no_grad():
        head.copy_(W)
    return model


def _ill_conditioned(vocab=1000, d=32, lo=-5.0, seed=0):
    """Singular values 1 .. 10**lo: eigenvalues of W.T @ W down to 10**(2 lo),
    below the float32 rounding of the Gram (1.2e-7) and far above the float64
    one (7e-15)."""
    g = torch.Generator().manual_seed(seed)
    u = torch.linalg.qr(torch.randn(vocab, d, generator=g, dtype=torch.float64))[0]
    v = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))[0]
    s = torch.logspace(0, lo, d, dtype=torch.float64)
    return ((u * s) @ v.T).to(torch.float32)


def test_float32_sign_loss_reports_the_float64_value(monkeypatch):
    """The GPU condition, simulated: the float32 spectrum comes back with a
    non-positive smallest eigenvalue, the float64 one positive."""
    model = fd.tiny_llama(n_layers=2)
    real = FE.head_eigenvalues

    def fake(W, dtype=torch.float32, *a, **k):
        e = real(W, dtype, *a, **k)
        if dtype == torch.float32:
            e = e.copy()
            e[0] = -abs(e[0]) * 1e-3
        return e

    clean = _run(model)
    assert math.isfinite(clean["lm_head_bfe_w"]) and REASON not in clean
    monkeypatch.setattr(FE, "head_eigenvalues", fake)
    r = _run(model)
    assert math.isfinite(r["lm_head_bfe_w"]), r
    assert r[FLAG] is True and REASON not in r and clean[FLAG] is False
    assert r["lm_head_bfe_w"] == r["_meta_lm_head_bfe_w_float64"] == clean["_meta_lm_head_bfe_w_float64"]
    assert r["_meta_lm_head_bfe_lambda"] == pytest.approx(clean["_meta_lm_head_bfe_lambda"], rel=1e-5)
    assert r["_meta_lm_head_min_eigenvalue_float32"] < 0 < r["_meta_lm_head_min_eigenvalue_float64"]


def test_ill_conditioned_head_is_not_nan():
    """No patching: a full-rank head whose smallest eigenvalues are under the
    float32 rounding of the Gram."""
    model = _with_head(_ill_conditioned())
    W = get_lm_head(model).weight.detach()
    e32 = torch.linalg.eigvalsh(W.T @ W).numpy()                 # the release's spectrum
    e64 = torch.linalg.eigvalsh(W.double().T @ W.double()).numpy()
    assert e32.min() <= 0 < 1e-11 < e64.min(), (e32.min(), e64.min())
    r = _run(model)
    assert math.isfinite(r["lm_head_bfe_w"]), r
    assert r[FLAG] is True and REASON not in r
    assert r["lm_head_bfe_w"] == r["_meta_lm_head_bfe_w_float64"]
    am_gm = e64.mean() / math.exp(np.log(e64).mean())
    assert r["lm_head_bfe_w"] == pytest.approx(
        0.5 * math.log(am_gm) + 0.5 + 0.5 * math.log(2 * math.pi), rel=1e-9)


def test_rank_deficient_head_is_nan_with_a_reason():
    """A zero direction is not rounding: float64 does not rescue it."""
    W = _ill_conditioned(lo=-1.0)
    W[:, 3] = 0.0
    r = _run(_with_head(W))
    assert math.isnan(r["lm_head_bfe_w"])
    assert "rank-deficient" in r.get(REASON, ""), r
    assert r[FLAG] is False
    assert math.isnan(r["_meta_lm_head_bfe_lambda"])


def test_float64_rounding_is_not_taken_for_full_rank(monkeypatch):
    """A zero eigenvalue that float64 returns as +1e-13 stays NaN."""
    model = fd.tiny_llama(n_layers=2)
    real = FE.head_eigenvalues

    def fake(W, dtype=torch.float32, *a, **k):
        e = real(W, dtype, *a, **k).copy()
        e[0] = -1e-6 if dtype == torch.float32 else 1e-13 * e[-1] / max(len(e), 1) * 1e-3
        return e

    monkeypatch.setattr(FE, "head_eigenvalues", fake)
    r = _run(model)
    assert math.isnan(r["lm_head_bfe_w"])
    assert "rank-deficient" in r.get(REASON, ""), r
    assert r[FLAG] is False


def test_non_finite_spectrum_is_nan_with_a_reason(monkeypatch):
    """A non-finite float32 eigenvalue is not a lost sign: no float64 rescue."""
    model = fd.tiny_llama(n_layers=2)
    real = FE.head_eigenvalues

    def fake(W, dtype=torch.float32, *a, **k):
        e = real(W, dtype, *a, **k).copy()
        if dtype == torch.float32:
            e[-1] = float("inf")
        return e

    monkeypatch.setattr(FE, "head_eigenvalues", fake)
    r = _run(model)
    assert math.isnan(r["lm_head_bfe_w"])
    assert "non-finite" in r.get(REASON, ""), r
    assert r[FLAG] is False


def test_finite_float32_value_is_the_release_value():
    """Where float32 is finite the headline is the release's float32 value,
    not the float64 one."""
    model = fd.tiny_llama(n_layers=2)
    W = get_lm_head(model).weight.detach().to(torch.float32)
    energy, lam = FE.bfe_w(torch.linalg.eigvalsh(W.T @ W).numpy())
    r = _run(model)
    assert r["lm_head_bfe_w"] == energy and r["_meta_lm_head_bfe_lambda"] == lam
    assert r["lm_head_bfe_w"] != r["_meta_lm_head_bfe_w_float64"]
    assert REASON not in r and r[FLAG] is False


def test_gpt2_bfloat16_on_cpu_is_finite():
    """The smoke's model on the CPU: float32 keeps the sign there."""
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "gpt2", dtype=torch.bfloat16, local_files_only=True).eval()
    except Exception as exc:  # not cached
        pytest.skip(f"gpt2 not available offline: {exc}")
    r = _run(model)
    assert r["_meta_head_shape"] == [50257, 768]
    assert r["_meta_lm_head_bfe_w_float64"] == pytest.approx(1.7263792492616, rel=1e-9)
    assert r["_meta_lm_head_min_eigenvalue_float64"] == pytest.approx(0.0204009110, rel=1e-6)
    assert REASON not in r and math.isfinite(r["lm_head_bfe_w"])
    assert r["lm_head_bfe_w"] == pytest.approx(1.7264, abs=3e-4)   # 1.72640-1.72677 over thread counts
    if r[FLAG]:
        assert r["lm_head_bfe_w"] == r["_meta_lm_head_bfe_w_float64"]
        assert r["_meta_lm_head_min_eigenvalue_float32"] <= 0
    else:
        assert r["_meta_lm_head_min_eigenvalue_float32"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gpu_weights_give_the_cpu_value(dtype):
    model = _with_head(_ill_conditioned(lo=-2.0)).to(dtype)
    cpu = _run(model)
    gpu = _run(model.to("cuda"))
    assert math.isfinite(cpu["lm_head_bfe_w"]) and cpu[FLAG] is False
    assert gpu["lm_head_bfe_w"] == cpu["lm_head_bfe_w"]
    assert gpu["_meta_lm_head_bfe_lambda"] == cpu["_meta_lm_head_bfe_lambda"]
    assert gpu["_meta_lm_head_bfe_w_float64"] == pytest.approx(cpu["_meta_lm_head_bfe_w_float64"], rel=1e-9)
