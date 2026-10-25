"""Bit-exact parity with released references, asserted mechanically.

WHY THIS FILE EXISTS
--------------------
Several tasks derive their verdict from agreeing with a released implementation
to 0.0, and the single most repeated maintainer error in this campaign is
"improving" their precision and silently destroying that agreement. On
2026-09-08 it happened twice in one hour:

  * `repe_task_vectors` — promoting the class means to double moved
    max|v_BLME - official ClusterMeanRepReader| from 0.000e+00 to 1.907e-06;
  * `geometry_schatten` — promoting `_normalise_matrix` moved
    max|BLME - official MNN.py| from 0.000e+00 to 1.192e-07,

the second AFTER the first had been diagnosed, fixed, and written up. Where a
task's value rests on reference parity, a strictly MORE ACCURATE computation is
a regression, and the correct shape is float32-first: promote only where the
float32 path has already failed, which is the regime the reference cannot
handle either.

Writing that rule in a ledger did not prevent the second occurrence. An
assertion does.

Each test skips if its reference is unavailable, so this file is safe in a
minimal environment — but it must FAIL, not skip, when the reference is present
and the number has moved.
"""
from __future__ import annotations

import importlib.util
import os

import pytest
import torch

_SCRATCH = os.environ.get(
    "BLME_REF_DIR",
    "/tmp/claude-1736197890/-home-dtsaras-projects-beyond-lm-eval/"
    "7d99fdd4-90f5-40ae-993c-0193a9a5f41b/scratchpad",
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or not os.path.exists(path):
        pytest.skip(f"reference not available: {path}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"reference failed to load: {type(exc).__name__}: {exc}")
    return mod


def test_schatten_mnn_is_bit_exact_against_released_reference():
    """`geometry_schatten` == MLGroupJLU/MatrixNuclearNorm on ordinary data.

    Guards the exact regression of 2026-09-08: an unconditional `.double()` in
    `_normalise_matrix` took this from 0.0 to 1.192e-07.
    """
    M = _load(os.path.join(_SCRATCH, "MNN_official.py"), "mnn_official_guard")
    if not hasattr(M, "matrix_nuclear_norm") or not hasattr(M, "normalize"):
        pytest.skip("official MNN.py lacks the expected entry points")
    from blme.tasks.geometry.schatten import (_matrix_nuclear_norm_fast,
                                              _normalise_matrix)

    gen = torch.Generator().manual_seed(7)
    worst = 0.0
    for _ in range(50):
        A = torch.randn(16, 32, generator=gen)
        blme = float(_matrix_nuclear_norm_fast(_normalise_matrix(A))) / 16.0
        ref = float(M.matrix_nuclear_norm(M.normalize(A.clone()))) / 16.0
        worst = max(worst, abs(blme - ref))
    assert worst == 0.0, (
        f"schatten lost bit-exact parity with the released MNN.py "
        f"(max |delta| = {worst:.3e}). If this followed a precision change, the "
        f"promotion must be gated on the float32 path having already failed.")


def test_schatten_still_survives_the_mixed_magnitude_case():
    """The other half: parity must not be bought back by re-breaking the guard."""
    from blme.tasks.geometry.schatten import (_normalise_matrix,
                                              _per_sentence_measurements)
    Z = torch.tensor([[1e37, 1e-13, 3.7e-14],
                      [1e37, -1e-13, -3.7e-14]], dtype=torch.float32)
    out = _per_sentence_measurements(_normalise_matrix(Z))
    got = out.get("row_normalized_matrix_nuclear_norm")
    assert got == pytest.approx(0.9085408449172974, abs=1e-9), (
        f"mixed-magnitude MNN/T = {got!r}; the executed official value is "
        f"0.9085408449172974 and the analytic value 0.9085407984878452")


def test_spectral_matches_live_weightwatcher_on_small_n():
    """`geometry_spectral`'s ported small-N branch == live WWFit."""
    wp = pytest.importorskip("weightwatcher.WW_powerlaw")
    import numpy as np

    from blme.tasks.geometry.spectral import _pl_mle_alpha_ks
    for n in (8, 9, 10, 12, 15, 19, 20):
        W = np.diag(np.arange(n, 0, -1).astype(np.float64))
        ev = np.sort(np.linalg.svd(W, compute_uv=False) ** 2)
        blme = _pl_mle_alpha_ks(ev)[0]
        ref = float(wp.pl_fit(data=ev, pl_package="ww").alpha)
        assert blme == pytest.approx(ref, rel=1e-9), f"N={n}: {blme} vs WWFit {ref}"


def test_magnitude_matches_live_magnipy():
    """`geometry_magnitude` == magnipy, including the collinear fixture that
    motivated deleting its centring."""
    mc = pytest.importorskip("magnipy.magnitude.compute")
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    from blme.tasks.geometry.magnitude import _magnitude, _pairwise_distance_matrix
    X = np.array([[0.0], [1.0], [9.999999562023526e24]])
    blme = _magnitude(_pairwise_distance_matrix(X), 1.0)
    D = squareform(pdist(X))
    ref = float(np.sum(mc.compute_magnitude_until_convergence(
        D, ts=[1.0], method="cholesky", perturb_singularities=True)[0]))
    assert blme == pytest.approx(ref, rel=1e-12), (
        f"magnitude {blme!r} vs live magnipy {ref!r} on the collinear fixture")
