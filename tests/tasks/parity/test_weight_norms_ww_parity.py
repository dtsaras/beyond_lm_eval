"""Parity: geometry_weight_norms vs the RELEASED WeightWatcher
(CalculatedContent/WeightWatcher@a5940f7), loaded from the clone under a
private package name, through evaluate().

WeightWatcher's ``apply_norm_metrics`` (weightwatcher.py:5525-5543) reads the
eigenvalues of W^T W: ``norm`` = sum(evals) = ||W||_F^2, ``spectral_norm`` =
max(evals) = sigma_max^2 and ``stable_rank`` = norm / spectral_norm. On every
2-D matrix of every block (HF Conv1D in tiny-random-gpt2, split Linear in a
tiny Llama, fused QKV in a tiny GPT-NeoX), BLME's four headline keys must
equal the per-block means of sqrt(norm), sqrt(spectral_norm) and stable_rank
from the release's ``analyze()``, averaged over blocks, and one minus the
coefficient of variation of the per-block Frobenius means. The release
computes in float32, hence rel 1e-5. Skips without the clone (fails under
BLME_REQUIRE_REFS=1).
"""
import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

pytest.importorskip("pandas")
pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "invariance"))

import _family_d as fd  # noqa: E402
from dynamics_fixtures import tiny_neox  # noqa: E402
from blme.tasks.common import get_layers  # noqa: E402
from blme.tasks.geometry import weight_norms as WN  # noqa: E402

_REPO = os.path.join(os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos")),
                     "WeightWatcher")
_PKG = os.path.join(_REPO, "weightwatcher")
_NAME = "_blme_parity_ww_norms_ref"


@pytest.fixture(scope="module")
def WeightWatcher():
    if not os.path.isfile(os.path.join(_PKG, "weightwatcher.py")):
        if os.environ.get("BLME_REQUIRE_REFS") == "1":
            pytest.fail(f"reference not available: {_REPO}")
        pytest.skip(f"reference not available: {_REPO}")
    spec = importlib.util.spec_from_file_location(_NAME, os.path.join(_PKG, "__init__.py"),
                                                  submodule_search_locations=[_PKG])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = pkg
    try:
        # RMT_Util picks its SVD and weight dtype by torch.cuda.is_available().
        with warnings.catch_warnings(), mock.patch.object(torch.cuda, "is_available", lambda: False), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            spec.loader.exec_module(pkg)
        yield pkg.WeightWatcher
    finally:
        for n in [n for n in sys.modules if n == _NAME or n.startswith(_NAME + ".")]:
            sys.modules.pop(n, None)


def test_sources_pin_the_cloned_commit(WeightWatcher):
    head = subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert ("CalculatedContent/WeightWatcher@" + head) in WN.WeightNormProfileTask.SOURCES["code"]


def _block_modules(model):
    """(named_modules position, name, block index) of every module in a block
    that owns a 2-D weight."""
    blocks = {id(b): i for i, b in enumerate(get_layers(model))}
    owner = {}
    for b in get_layers(model):
        for m in b.modules():
            owner[id(m)] = blocks[id(b)]
    out = []
    for pos, (name, m) in enumerate(model.named_modules()):
        w = getattr(m, "weight", None)
        if id(m) in owner and isinstance(w, torch.nn.Parameter) and w.ndim == 2:
            out.append((pos, name, owner[id(m)]))
    return out


@pytest.mark.parametrize("build", [fd.tiny_gpt2, lambda: fd.tiny_llama(n_layers=3),
                                   lambda: tiny_neox(n_layers=3)],
                         ids=["gpt2_conv1d", "llama", "neox_fused_qkv"])
def test_evaluate_equals_release_norm_metrics(WeightWatcher, build):
    model = build()
    res = WN.WeightNormProfileTask({}).evaluate(model, None, None)
    mods = _block_modules(model)
    n_params = sum(1 for b in get_layers(model) for p in b.parameters() if p.ndim >= 2)
    assert len(mods) == n_params
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = WeightWatcher(model=model).analyze(layers=[p for p, _, _ in mods],
                                                svd_method="accurate")
    assert list(df["longname"]) == [n for _, n, _ in mods]
    block = np.array([b for _, _, b in mods])
    L = int(block.max()) + 1
    frob = np.sqrt(df["norm"].to_numpy(float))
    spec = np.sqrt(df["spectral_norm"].to_numpy(float))
    srank = df["stable_rank"].to_numpy(float)
    per = {k: np.array([v[block == i].mean() for i in range(L)])
           for k, v in (("frob", frob), ("spec", spec), ("srank", srank))}
    assert len(res["frobenius_norm_per_layer"]) == L
    np.testing.assert_allclose(res["frobenius_norm_per_layer"], per["frob"], rtol=1e-5)
    np.testing.assert_allclose(res["spectral_norm_per_layer"], per["spec"], rtol=1e-5)
    np.testing.assert_allclose(res["stable_rank_per_layer"], per["srank"], rtol=1e-5)
    assert res["mean_frobenius_norm"] == pytest.approx(per["frob"].mean(), rel=1e-5)
    assert res["mean_spectral_norm"] == pytest.approx(per["spec"].mean(), rel=1e-5)
    assert res["mean_stable_rank"] == pytest.approx(per["srank"].mean(), rel=1e-5)
    cv = per["frob"].std() / per["frob"].mean()
    assert res["norm_uniformity"] == pytest.approx(max(0.0, 1.0 - cv), rel=1e-5, abs=1e-7)
