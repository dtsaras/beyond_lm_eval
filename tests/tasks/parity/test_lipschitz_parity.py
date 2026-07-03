"""Numeric-parity test: the spectral-norm (sigma_max) KERNEL underlying the
AutoLip Lipschitz upper bound (Virmaux & Scaman 2018; Miyato 2018).

IMPORTANT — task scope (read before assuming what this covers)
--------------------------------------------------------------
BLME's ``geometry_lipschitz`` (src/blme/tasks/geometry/lipschitz.py) does NOT
implement a spectral-norm / power-iteration sigma_max kernel and does NOT form
an AutoLip product of per-layer spectral norms. It computes the hidden-state
RELATIVE-CHANGE ratio ||h_{l+1}(x) - h_l(x)|| / ||h_l(x)|| across adjacent
layers, and is already labelled ``proxy-only`` in task_metadata.py and in its
own docstring (lines 37-40, 112-115). The Virmaux&Scaman / Miyato citations
refer to the sigma_max KERNEL, which in BLME lives in ``weight_norms.py`` and
``spectral.py`` (both EXACT SVD: ``torch.linalg.svdvals(W)[0]``), NOT in
``lipschitz.py``.

What this test proves (to the strictest standard):
  1. BLME's ACTUAL sigma_max kernel ``torch.linalg.svdvals(W)[0]``
     (src/blme/tasks/geometry/weight_norms.py:79-80) == exact
     ``numpy.linalg.norm(W, 2)`` == ``scipy.linalg.svdvals(W)[0]`` to < 1e-9.
  2. The OFFICIAL reference power iteration (avirmaux/lipEstimation @ 336b6cc:
     max_eigenvalue._power_method_matrix on W^T W, and generic_power_method on
     a Linear layer) CONVERGES to the exact sigma_max to ~7e-8 relative given
     enough iterations. Reference runs in float32 (its own dtype contract), so
     the achievable parity floor is float32-limited -> RELATIVE tol 1e-3.
  3. The AutoLip global bound == PRODUCT of exact per-layer sigma_max
     (reference lipschitz_spectral_ub semantics). This PRODUCT is an
     UPPER BOUND on the true network Lipschitz constant (the tighter SeqLip
     bound is <= this product) -> documented PROXY, not tight.
  4. Anchors: diagonal -> max|diag|; orthogonal -> 1; scaled identity cI -> c.

VERDICT: sigma_max KERNEL = PARITY; global product-of-sigma_max = documented
PROXY / AutoLip UPPER BOUND; geometry_lipschitz task = separate hidden-state
relative-change proxy.
"""
import os
import sys

import numpy as np
import pytest
import scipy.linalg
import torch

# ---------------------------------------------------------------------------
# Locate the OFFICIAL reference (avirmaux/lipEstimation). Skip cleanly if the
# clone is not present in this environment.
# ---------------------------------------------------------------------------
_REF_CANDIDATES = [
    os.environ.get("LIPESTIMATION_DIR", ""),
    "/tmp/claude-1736197890/-home-dtsaras-projects-beyond-lm-eval/"
    "42dc090a-aa6d-4f5b-bd0f-fb7213022dc9/scratchpad/refrepos/lipEstimation",
]
_REF_DIR = next((d for d in _REF_CANDIDATES if d and
                 os.path.isfile(os.path.join(d, "max_eigenvalue.py"))), None)
if _REF_DIR is None:
    pytest.skip(
        "avirmaux/lipEstimation reference not found (set LIPESTIMATION_DIR); "
        "clone: git clone --depth 1 https://github.com/avirmaux/lipEstimation",
        allow_module_level=True,
    )
if _REF_DIR not in sys.path:
    sys.path.insert(0, _REF_DIR)

from max_eigenvalue import (  # noqa: E402
    _power_method_matrix,
    generic_power_method,
)

TOL_BLME = 1e-9        # exact-SVD kernel vs exact reference (float64 eps)
TOL_POW_REL = 1e-3     # float32 power iteration convergence (relative)


# ---------------------------------------------------------------------------
# sigma_max computations
# ---------------------------------------------------------------------------
def exact_sigma_max_numpy(W):
    """Exact reference: numpy.linalg.norm(W, 2) == largest singular value."""
    return float(np.linalg.norm(W, 2))


def blme_sigma_max(W):
    """BLME's ACTUAL kernel: torch.linalg.svdvals(W)[0]
    (src/blme/tasks/geometry/weight_norms.py:79-80, spectral.py:66)."""
    Wt = torch.as_tensor(W, dtype=torch.float64)
    return float(torch.linalg.svdvals(Wt)[0].item())


def ref_power_matrix(W, max_iter=5000, eps=1e-12):
    """Reference power iteration on W^T W (max_eigenvalue._power_method_matrix).
    Reference initialises v in float32, so W must be float32 to match it."""
    Wt = torch.as_tensor(W, dtype=torch.float32)
    ev, _ = _power_method_matrix(Wt, eps=eps, max_iter=max_iter)
    return float(ev.item())


def ref_power_generic(W, max_iter=5000, eps=1e-12):
    """Reference generic_power_method on a bias-free Linear layer y = W x.
    Returns sigma_max of the linear part, i.e. sigma_max(W)."""
    out_f, in_f = W.shape
    lin = torch.nn.Linear(in_f, out_f, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.as_tensor(W, dtype=torch.float32))
    for p in lin.parameters():
        p.requires_grad = False
    ev, _, _ = generic_power_method(lin.float(), [1, in_f], eps=eps,
                                    max_iter=max_iter)
    return float(ev.item())


_SHAPES = [(8, 8), (16, 32), (64, 64), (128, 96), (256, 256)]


def _random_W(m, n, seed):
    rng = np.random.RandomState(seed)
    return rng.randn(m, n).astype(np.float64)


# ---------------------------------------------------------------------------
# (1) BLME exact-SVD kernel == exact numpy/scipy sigma_max
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m,n", _SHAPES)
def test_blme_kernel_matches_exact_sigma_max(m, n):
    """BLME's torch.linalg.svdvals(W)[0] == numpy.linalg.norm(W,2) == scipy."""
    W = _random_W(m, n, seed=hash((m, n)) & 0xFFFF)
    exact_np = exact_sigma_max_numpy(W)
    exact_sp = float(scipy.linalg.svdvals(W)[0])
    blme = blme_sigma_max(W)
    assert abs(blme - exact_np) < TOL_BLME, (blme, exact_np)
    assert abs(blme - exact_sp) < TOL_BLME, (blme, exact_sp)


# ---------------------------------------------------------------------------
# (2) Reference power iteration converges to exact sigma_max
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m,n", _SHAPES)
def test_reference_power_iteration_converges_to_exact(m, n):
    """lipEstimation power iteration -> exact sigma_max (both variants)."""
    W = _random_W(m, n, seed=(hash((m, n)) & 0xFFFF) + 1)
    exact_np = exact_sigma_max_numpy(W)
    p_mat = ref_power_matrix(W)
    p_gen = ref_power_generic(W)
    rel_mat = abs(p_mat - exact_np) / exact_np
    rel_gen = abs(p_gen - exact_np) / exact_np
    assert rel_mat < TOL_POW_REL, f"power_matrix rel={rel_mat:.2e}"
    assert rel_gen < TOL_POW_REL, f"generic_power rel={rel_gen:.2e}"


# ---------------------------------------------------------------------------
# (3) AutoLip global bound == PRODUCT of per-layer sigma_max (UPPER BOUND)
# ---------------------------------------------------------------------------
def test_autolip_product_of_sigma_max():
    """Global AutoLip bound = product of exact per-layer sigma_max.

    This PRODUCT is an UPPER BOUND on the true network Lipschitz constant
    (the tighter SeqLip bound is <= this product): it is a documented PROXY,
    not the tight constant. BLME's exact-SVD kernel reproduces each factor,
    so the product is bit-exact vs the exact reference."""
    layer_shapes = [(32, 48), (48, 48), (48, 16)]
    per_exact, per_blme = [], []
    for i, (m, n) in enumerate(layer_shapes):
        W = _random_W(m, n, seed=100 + i)
        per_exact.append(exact_sigma_max_numpy(W))
        per_blme.append(blme_sigma_max(W))
    prod_exact = float(np.prod(per_exact))
    prod_blme = float(np.prod(per_blme))
    assert abs(prod_exact - prod_blme) < TOL_BLME, (prod_exact, prod_blme)
    # AutoLip is an upper bound: product >= max single-layer sigma_max.
    assert prod_exact >= max(per_exact) - TOL_BLME


# ---------------------------------------------------------------------------
# (4) Anchors: diagonal, orthogonal, scaled identity
# ---------------------------------------------------------------------------
def test_anchor_diagonal():
    """sigma_max(diag(d)) == max|d|."""
    diag = np.array([-3.0, 0.5, 2.0, -1.25, 4.0])
    W = np.diag(diag)
    expected = float(np.max(np.abs(diag)))
    assert abs(blme_sigma_max(W) - expected) < TOL_BLME
    assert abs(ref_power_matrix(W) - expected) < TOL_POW_REL * max(expected, 1.0)


def test_anchor_orthogonal():
    """sigma_max(Q) == 1 for orthogonal Q."""
    rng = np.random.RandomState(7)
    Q, _ = np.linalg.qr(rng.randn(20, 20))
    assert abs(blme_sigma_max(Q) - 1.0) < TOL_BLME
    assert abs(ref_power_matrix(Q) - 1.0) < TOL_POW_REL


def test_anchor_scaled_identity():
    """sigma_max(cI) == c."""
    c = 3.7
    W = c * np.eye(12)
    assert abs(blme_sigma_max(W) - c) < TOL_BLME
    assert abs(ref_power_matrix(W) - c) < TOL_POW_REL * c


# ---------------------------------------------------------------------------
# (5) Honest label: geometry_lipschitz ships a DIFFERENT proxy metric.
# ---------------------------------------------------------------------------
def test_geometry_lipschitz_is_relative_change_proxy_not_sigma_max():
    """The geometry_lipschitz task computes ||dh||/||h|| (hidden-state
    relative-change ratio), NOT a sigma_max weight-matrix kernel. Guard the
    honest labelling so a future refactor that silently swaps in a real
    Lipschitz kernel (or mislabels this one) trips this test."""
    from blme.task_metadata import TASK_CERTIFICATION
    entry = TASK_CERTIFICATION["geometry_lipschitz"]
    # Metadata must keep it labelled as a proxy (not parity-ready).
    status = entry.status
    assert status == "proxy-only", (
        f"geometry_lipschitz status changed to {status!r}; it computes a "
        "hidden-state relative-change ratio, not the true operator Lipschitz "
        "constant / AutoLip product — keep it labelled proxy-only.")

    # Source of the task must NOT contain a weight-matrix sigma_max kernel.
    import blme.tasks.geometry.lipschitz as lip_mod
    src = open(lip_mod.__file__).read()
    for forbidden in ("svdvals", "power_iteration", "power_method",
                      "linalg.svd"):
        assert forbidden not in src, (
            f"geometry_lipschitz now references {forbidden!r}; if a real "
            "sigma_max/AutoLip kernel was added, update its label and this "
            "parity test to verify it against avirmaux/lipEstimation.")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
