"""Parity: geometry_lm_head_free_energy vs the RELEASED heavytail code
(josephwilsonmaths/heavytail@80f7f80): ``util/pythia_ww.py::bfe_w`` and
``util/qwen_benchmarks.py::compute_free_energy``, compiled from the clone's
source (their modules import plotting and benchmark loaders at top level), on
the spectrum ``util/pythia_spectra.py`` computes (eigvalsh of W.T @ W in
float32). Skips when the clone is absent (fails under BLME_REQUIRE_REFS=1).
"""
import ast
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "invariance"))

import _family_d as fd  # noqa: E402
from blme.tasks.common import get_lm_head  # noqa: E402
from blme.tasks.geometry import lm_head_free_energy as FE  # noqa: E402

_ROOT = os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))
_REPO = os.path.join(_ROOT, "heavytail")


def _missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


@pytest.fixture(scope="module")
def ref():
    if not os.path.isdir(_REPO):
        _missing(_REPO)
    ns = {"np": np, "torch": torch}
    for fname, name in (("util/pythia_ww.py", "bfe_w"), ("util/qwen_benchmarks.py", "compute_free_energy")):
        path = os.path.join(_REPO, fname)
        tree = ast.parse(open(path).read())
        defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        assert len(defs) == 1
        exec(compile(ast.Module(body=defs, type_ignores=[]), path, "exec"), ns)
    return ns


def test_sources_pin_the_cloned_commit():
    if not os.path.isdir(_REPO):
        _missing(_REPO)
    head = subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert ("@" + head) in FE.LMHeadFreeEnergyTask.SOURCES["code"]


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_evaluate_matches_the_release(ref, arch):
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=2)
    r = FE.LMHeadFreeEnergyTask({}).evaluate(model, None, None)
    weight = get_lm_head(model).weight.detach().to(torch.float32)
    eigvals = torch.linalg.eigvalsh(weight.T @ weight).cpu().numpy()      # pythia_spectra.py
    energy, lam = ref["bfe_w"](eigvals, 0.0)
    lam_q, energy_q = ref["compute_free_energy"](torch.from_numpy(eigvals))
    assert r["lm_head_bfe_w"] == pytest.approx(energy, rel=1e-12)
    assert r["_meta_lm_head_bfe_lambda"] == pytest.approx(lam, rel=1e-12)
    assert energy_q == pytest.approx(energy, rel=1e-5) and lam_q == pytest.approx(lam, rel=1e-5)


def test_closed_form_and_invariances():
    """BFE-W = 1/2 log(AM/GM) + 1/2 + 1/2 log(2 pi); unchanged by rescaling
    and rotating the head; NaN for a rank-deficient head (zero eigenvalue)."""
    g = torch.Generator().manual_seed(0)
    W = torch.randn(500, 24, generator=g, dtype=torch.float64) * torch.linspace(0.1, 3, 24, dtype=torch.float64)
    e = FE.head_eigenvalues(W, torch.float64)
    energy, _ = FE.bfe_w(e)
    am_gm = e.mean() / math.exp(np.log(e).mean())
    assert energy == pytest.approx(0.5 * math.log(am_gm) + 0.5 + 0.5 * math.log(2 * math.pi), rel=1e-12)
    q = torch.linalg.qr(torch.randn(24, 24, generator=g, dtype=torch.float64))[0]
    assert FE.bfe_w(FE.head_eigenvalues(37.0 * W @ q, torch.float64))[0] == pytest.approx(energy, rel=1e-10)
    padded = torch.cat([W, torch.zeros(500, 2, dtype=torch.float64)], dim=1)
    assert math.isnan(FE.bfe_w(FE.head_eigenvalues(padded, torch.float64))[0])
    # A flat spectrum is the minimum: AM = GM.
    flat = FE.bfe_w(np.ones(24))[0]
    assert flat == pytest.approx(0.5 + 0.5 * math.log(2 * math.pi)) and energy > flat


def test_chunked_float64_gram_matches_one_product():
    W = torch.randn(1000, 16, generator=torch.Generator().manual_seed(1))
    full = torch.linalg.eigvalsh(W.double().T @ W.double()).numpy()
    np.testing.assert_allclose(FE.head_eigenvalues(W, torch.float64, chunk=7), full, rtol=1e-12)


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_headline_invariant_under_identity_blocks(arch):
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=2)
    tok = fd.tokenizer()
    ids = [tok(t, return_tensors="pt")["input_ids"] for t in fd.TEXTS[:3]]
    deep = fd.inv.insert_identity_blocks(model, [0, 1], verify_on=ids)
    base = FE.LMHeadFreeEnergyTask({}).evaluate(model, tok, None)
    ins = FE.LMHeadFreeEnergyTask({}).evaluate(deep, tok, None)
    fd.inv.assert_invariant(base, ins, sorted(FE.LMHeadFreeEnergyTask.SIZE_INVARIANT), rtol=0.0)
