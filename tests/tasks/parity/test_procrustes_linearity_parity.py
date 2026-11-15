"""Numeric-parity test: BLME geometry_procrustes_linearity vs the OFFICIAL
LLM-Microscope reference (Razzhigaev et al. 2024, arXiv:2405.12250).

TASK: geometry_procrustes_linearity
BLME: src/blme/tasks/geometry/procrustes_linearity.py::_procrustes_similarity.
      The headline profile uses ``rank_aware=True`` (Moore-Penrose cutoff at
      numpy's pinv tolerance S_0 max(N, D) eps); the reference's every-positive
      1 / S rule (``rank_aware=False``) feeds only the
      ``_meta_reference_rule_*`` keys.

Reference (OFFICIAL): AIRI-Institute/LLM-Microscope commit
b6db939b2696845ce1f88cf69cddc41e808eea17, pip package llm-microscope==0.0.7.
The metric is ``procrustes_similarity`` from LLM_microscope.ipynb cell 3 ==
llm_microscope/functions.py:86-115: an UNCONSTRAINED least-squares map
A = X^+ Y on centred, Frobenius-normalised clouds, sim = 1 - ||X A - Y||_F^2.
It is transcribed below; tests/tasks/parity/test_procrustes_linearity_values.py
runs the notebook's own cell through evaluate().

The bar pinned here:
  (a) full-rank N >> D: both branches equal the reference (in fact 0.0);
  (b) anchors (orthogonal map, rescale -> exactly 1; independent -> low)
      hold for both branches;
  (c) a centred N < D cloud is rank-deficient (rank N - 1): the headline
      branch is the minimum-norm least-squares fit, an orthogonal projection,
      so an exactly linear Y scores 1 to rounding, against an independent
      numpy-SVD projection; the reference rule inverts the null singular value;
  (d) an ill-conditioned but full-rank cloud has no singular value below the
      cutoff, so the two branches and the torch reference coincide.
"""

import json
from pathlib import Path

import numpy as np
import pytest

# The REAL BLME helper — src/blme is exercised, not a copy.
from blme.tasks.geometry.procrustes_linearity import _procrustes_similarity

torch = pytest.importorskip("torch")


FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures/reference_parity/parity/procrustes_linearity.json"
)


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# OFFICIAL reference, transcribed VERBATIM from LLM-Microscope @ b6db939
# (LLM_microscope.ipynb cell 3 == llm_microscope/functions.py:86-115).
# Only cosmetic: computed in float64 to isolate the algorithm from dtype noise
# (the repo runs it on whatever dtype the input tensors carry).
# ---------------------------------------------------------------------------
def _ref_get_est_svd(X, Y):
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    A_estimation = Vh.T * (1 / S)[None, ...] @ U.T @ Y  # Y=XA
    Y_est = X @ A_estimation
    return Y_est


def _ref_procrustes_similarity(x, y):
    with torch.no_grad():
        X = x - x.mean(dim=0, keepdim=True)
        Y = y - y.mean(dim=0, keepdim=True)
        X = X / X.norm()
        Y = Y / Y.norm()
        Y_estimation = _ref_get_est_svd(X, Y)
        y_error = (Y_estimation - Y).square().sum()
        sim = float(1 - y_error)
    return sim


def _ref(X, Y) -> float:
    return _ref_procrustes_similarity(
        torch.from_numpy(np.asarray(X, dtype=np.float64)),
        torch.from_numpy(np.asarray(Y, dtype=np.float64)),
    )


# ===========================================================================
# (a) BLME == OFFICIAL reference, bit-exact, on a well-conditioned N>>D pair.
# ===========================================================================
def test_blme_matches_official_reference_primary():
    rng = np.random.default_rng(0)
    N, D = 200, 16
    X = rng.standard_normal((N, D))
    A_true = rng.standard_normal((D, D))
    Y = X @ A_true + 0.30 * rng.standard_normal((N, D))

    off = _ref(X, Y)
    blme = _procrustes_similarity(X, Y)
    head, rank = _procrustes_similarity(X, Y, rank_aware=True, return_rank=True)

    assert abs(off - blme) < 1e-6
    assert off == pytest.approx(blme, abs=1e-12)
    assert rank == D and head == pytest.approx(off, abs=1e-12)

    fx = _fixture()["primary"]
    assert blme == pytest.approx(fx["blme"], abs=1e-9)
    assert off == pytest.approx(fx["official"], abs=1e-9)
    assert abs(off - blme) <= fx["tol"]


# ===========================================================================
# (b) Anchors.
# ===========================================================================
def test_anchor_orthogonal_map_is_exactly_one():
    """Y = X @ Q with Q random ORTHOGONAL => Y in row space of X => sim == 1.0."""
    rng = np.random.default_rng(0)
    N, D = 200, 16
    X = rng.standard_normal((N, D))
    G = rng.standard_normal((D, D))
    Q, Rm = np.linalg.qr(G)
    Q = Q * np.sign(np.diag(Rm))
    Y = X @ Q

    blme = _procrustes_similarity(X, Y)
    off = _ref(X, Y)

    assert abs(blme - 1.0) < 1e-9
    assert abs(off - 1.0) < 1e-9
    assert abs(_procrustes_similarity(X, Y, rank_aware=True) - 1.0) < 1e-9
    assert abs(off - blme) < 1e-6

    fx = _fixture()["anchor_orthogonal"]
    assert fx["eq_one"] is True
    assert blme == pytest.approx(fx["blme"], abs=1e-9)


def test_anchor_independent_is_low():
    """Independent random Y => low linearity, and still matches the reference."""
    rng = np.random.default_rng(0)
    N, D = 200, 16
    X = rng.standard_normal((N, D))
    _ = rng.standard_normal((D, D))                # consume RNG to match verify order
    _ = X @ _ + 0.30 * rng.standard_normal((N, D))
    _ = np.linalg.qr(rng.standard_normal((D, D)))  # orthogonal-anchor draw
    Y_ind = rng.standard_normal((N, D))

    blme = _procrustes_similarity(X, Y_ind)
    off = _ref(X, Y_ind)

    assert blme < 0.5
    assert abs(off - blme) < 1e-6
    assert _procrustes_similarity(X, Y_ind, rank_aware=True) == pytest.approx(off, abs=1e-12)

    fx = _fixture()["anchor_independent"]
    assert fx["below_half"] is True
    assert blme == pytest.approx(fx["blme"], abs=1e-9)


def test_anchor_scale_invariance_is_exactly_one():
    """Y = 2*X: Frobenius-normalization strips the global scale => sim == 1.0.

    Documents that the linearity score is scale-invariant (the identity map
    sends normalized X onto normalized Y exactly).
    """
    rng = np.random.default_rng(0)
    N, D = 200, 16
    X = rng.standard_normal((N, D))
    Y = 2.0 * X

    blme = _procrustes_similarity(X, Y)
    off = _ref(X, Y)

    assert abs(blme - 1.0) < 1e-9
    assert abs(off - 1.0) < 1e-9
    assert abs(_procrustes_similarity(X, Y, rank_aware=True) - 1.0) < 1e-9

    fx = _fixture()["anchor_scaled_2x"]
    assert fx["eq_one"] is True


# ===========================================================================
# (c) N < D: the centred X has rank N - 1.
# ===========================================================================
def _numpy_projection_sim(X, Y):
    """1 - ||P_X Y - Y||_F^2 on centred, normalised clouds, P_X from numpy's
    SVD at the same cutoff (an independent minimum-norm least squares)."""
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    Xc, Yc = Xc / np.linalg.norm(Xc), Yc / np.linalg.norm(Yc)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    keep = S > S[0] * max(Xc.shape) * np.finfo(np.float64).eps
    Ur = U[:, keep]
    return 1.0 - float(np.square(Ur @ (Ur.T @ Yc) - Yc).sum()), int(keep.sum())


def test_wide_matrix_headline_is_the_minimum_norm_fit():
    rng = np.random.default_rng(0)
    Nw, Dw = 12, 40
    Xw = rng.standard_normal((Nw, Dw))
    Aw = rng.standard_normal((Dw, Dw))
    Yw = Xw @ Aw

    head, rank = _procrustes_similarity(Xw, Yw, rank_aware=True, return_rank=True)
    want, want_rank = _numpy_projection_sim(Xw, Yw)
    assert rank == want_rank == Nw - 1
    assert head == pytest.approx(want, abs=1e-12)
    assert head == pytest.approx(1.0, abs=1e-12)

    # The reference rule on the same pair (_meta_reference_rule_*).
    blme = _procrustes_similarity(Xw, Yw)
    off = _ref(Xw, Yw)
    assert off == pytest.approx(blme, abs=1e-12)
    fx = _fixture()["anchor_wide_NleD"]
    assert blme == pytest.approx(fx["blme"], abs=1e-9)
    assert off == pytest.approx(fx["official"], abs=1e-9)


def test_rank_deficient_independent_target_headline_is_the_projection():
    """An independent Y on a rank-deficient X: the headline is the share of
    Y in the column space of X, as numpy's projection computes it."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((30, 5)) @ rng.standard_normal((5, 20))
    Y = rng.standard_normal((30, 20))
    head, rank = _procrustes_similarity(X, Y, rank_aware=True, return_rank=True)
    want, want_rank = _numpy_projection_sim(X, Y)
    assert rank == want_rank == 5
    assert head == pytest.approx(want, abs=1e-12)
    assert 0.0 < head < 0.5


# ===========================================================================
# (d) Conditioning caveat: BLME tracks the torch reference bit-exactly even
#     on an ill-conditioned cloud. The absolute value is a conditioning
#     artifact (module CONDITIONING CAVEAT); the port is torch-based so it
#     reproduces the reference rather than a differently-resolved null space.
# ===========================================================================
def test_conditioning_ill_conditioned_still_bit_exact():
    rng = np.random.default_rng(11)
    # Low-effective-rank cloud with a near-epsilon spectral tail: rank-8
    # signal embedded in D=120 plus a tiny perturbation.
    N, r, D = 300, 8, 120
    core = rng.standard_normal((N, r)) @ rng.standard_normal((r, D))
    X = core + 1e-7 * rng.standard_normal((N, D))
    Y = X @ rng.standard_normal((D, D)) + 0.05 * rng.standard_normal((N, D))

    blme = _procrustes_similarity(X, Y)
    off = _ref(X, Y)

    head, rank = _procrustes_similarity(X, Y, rank_aware=True, return_rank=True)

    # Whatever the (artifact-laden) value is, BLME == the torch reference.
    assert abs(off - blme) < 1e-6
    assert off == pytest.approx(blme, abs=1e-9)
    # Every singular value clears the cutoff, so the headline branch agrees.
    assert rank == D and head == pytest.approx(blme, abs=1e-9)


def test_degenerate_inputs_return_nan():
    """Constant / mismatched / too-small clouds return NaN, never crash."""
    X = np.ones((10, 4))                 # constant => zero norm after centering
    Y = np.random.default_rng(1).standard_normal((10, 4))
    assert np.isnan(_procrustes_similarity(X, Y))
    assert np.isnan(_procrustes_similarity(Y, X))                 # Y-side constant
    assert np.isnan(_procrustes_similarity(Y, Y[:, :3]))          # shape mismatch
    assert np.isnan(_procrustes_similarity(Y[:1], Y[:1]))         # N<2
    for a, b in ((X, Y), (Y, X), (Y, Y[:, :3]), (Y[:1], Y[:1])):
        assert np.isnan(_procrustes_similarity(a, b, rank_aware=True))

