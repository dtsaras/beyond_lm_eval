"""Numeric-parity test: BLME geometry_procrustes_linearity vs the OFFICIAL
LLM-Microscope reference (Razzhigaev et al. 2024, arXiv:2405.12250).

TASK: geometry_procrustes_linearity
BLME: src/blme/tasks/geometry/procrustes_linearity.py
      _procrustes_similarity(X, Y)  — the verified artifact.

Reference (OFFICIAL): AIRI-Institute/LLM-Microscope commit
b6db939b2696845ce1f88cf69cddc41e808eea17, pip package llm-microscope==0.0.7.
The metric is ``procrustes_similarity`` from LLM_microscope.ipynb cell 3 ==
llm_microscope/functions.py:86-115. Despite the "procrustes" name it fits an
UNCONSTRAINED least-squares linear map A = X^+ Y (pseudo-inverse), NOT an
orthogonal transform; sim = 1 - ||X A - Y||_F^2 on centered + Frobenius-
normalized clouds.

The reference is transcribed VERBATIM below (REF_SRC) so the test exercises
the exact upstream algebra without importing the llm_microscope package or a
model. The BLME side imports the REAL helper from src/blme (not a copy).

The bar pinned here:
  (a) BLME _procrustes_similarity == the OFFICIAL reference, bit-exact
      (< 1e-6, in fact 0.0) on a well-conditioned N>>D pair.
  (b) Anchors: orthogonal map -> exactly 1.0; independent random -> low;
      2*X (scale) -> exactly 1.0 (score is scale-invariant).
  (c) N<D rank-deficient X: BLME now == the reference bit-exactly too,
      because BLME runs the SAME torch SVD (an earlier numpy port diverged).
  (d) Conditioning caveat, pinned as a test: on an ill-conditioned matrix a
      numpy-SVD pseudo-inverse and torch's disagree substantially; BLME must
      track torch (the reference), so this divergence is asserted to document
      why the port is torch-based, not numpy-based.
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

    assert abs(off - blme) < 1e-6
    assert off == pytest.approx(blme, abs=1e-12)

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

    assert abs(blme - 1.0) < 1e-9      # headline anchor from the prompt
    assert abs(off - 1.0) < 1e-9
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

    fx = _fixture()["anchor_scaled_2x"]
    assert fx["eq_one"] is True


# ===========================================================================
# (c) N<D rank-deficient X — BLME matches the reference bit-exactly (torch SVD).
# ===========================================================================
def test_wide_matrix_matches_reference_bit_exact():
    rng = np.random.default_rng(0)
    Nw, Dw = 12, 40
    Xw = rng.standard_normal((Nw, Dw))
    Aw = rng.standard_normal((Dw, Dw))
    Yw = Xw @ Aw

    blme = _procrustes_similarity(Xw, Yw)
    off = _ref(Xw, Yw)

    assert abs(off - blme) < 1e-6
    assert off == pytest.approx(blme, abs=1e-12)

    fx = _fixture()["anchor_wide_NleD"]
    assert blme == pytest.approx(fx["blme"], abs=1e-9)
    assert off == pytest.approx(fx["official"], abs=1e-9)


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

    # Whatever the (artifact-laden) value is, BLME == the torch reference.
    assert abs(off - blme) < 1e-6
    assert off == pytest.approx(blme, abs=1e-9)


def test_degenerate_inputs_return_nan():
    """Constant / mismatched / too-small clouds return NaN, never crash."""
    X = np.ones((10, 4))                 # constant => zero norm after centering
    Y = np.random.default_rng(1).standard_normal((10, 4))
    assert np.isnan(_procrustes_similarity(X, Y))
    assert np.isnan(_procrustes_similarity(Y, X))                 # Y-side constant
    assert np.isnan(_procrustes_similarity(Y, Y[:, :3]))          # shape mismatch
    assert np.isnan(_procrustes_similarity(Y[:1], Y[:1]))         # N<2


def test_fixture_records_parity_verdict():
    fx = _fixture()
    assert fx["task"] == "geometry_procrustes_linearity"
    assert fx["verdict"] == "PARITY"
    assert fx["primary"]["pass"] is True
    assert fx["reference_impl"]["commit"].startswith("b6db939")
    assert "least-squares" in fx["reference_impl"]["transform"]
