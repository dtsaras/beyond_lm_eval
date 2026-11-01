"""Parity: geometry_alpha_req vs the RELEASED reptrix code (BARL-SSL/reptrix@dbf53c5).

``utils.get_eigenspectrum`` (scikit-learn PCA explained-variance ratios),
``alpha.get_powerlaw`` (the weighted power-law fit) and ``rankme.get_rank``
are run on the same inputs as BLME's kernels, and the task's real
``evaluate()`` is compared with ``alpha.get_alpha`` / ``rankme.get_rankme`` on
the cloud it scores. Skips when the clone is absent (fails under
BLME_REQUIRE_REFS=1).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import adversarial_fixtures as adv  # noqa: E402
import invariance as inv  # noqa: E402
from blme.tasks.geometry import alpha_req as A  # noqa: E402

_REPO = os.path.join(os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos")),
                     "reptrix")


@pytest.fixture(scope="module")
def ref():
    if not os.path.isdir(os.path.join(_REPO, "reptrix")):
        if os.environ.get("BLME_REQUIRE_REFS") == "1":
            pytest.fail(f"reference not available: {_REPO}")
        pytest.skip(f"reference not available: {_REPO}")
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, _REPO)
    try:
        from reptrix import alpha, rankme, utils
    finally:
        sys.path.remove(_REPO)
    return alpha, rankme, utils


def test_sources_pin_the_cloned_commit(ref):
    import subprocess
    head = subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert A.AlphaReQTask.SOURCES["code"].endswith("@" + head)


def _clouds():
    """Power-law clouds wider and narrower than N, and an isotropic one."""
    rng = np.random.default_rng(0)
    for n, d, a in ((400, 256, 1.0), (256, 400, 1.5), (300, 300, 0.7)):
        scale = np.arange(1, d + 1, dtype=np.float64) ** (-a / 2)
        yield f"N={n} D={d} alpha={a}", rng.standard_normal((n, d)) * scale
    yield "isotropic N=500 D=200", rng.standard_normal((500, 200))


@pytest.mark.parametrize("label,X", list(_clouds()), ids=lambda v: v if isinstance(v, str) else "")
def test_eigenspectrum_alpha_and_rankme_match_reptrix(ref, label, X):
    alpha, rankme, utils = ref
    eig = A._eigenspectrum(X)
    eig_ref = utils.get_eigenspectrum(activations_np=X, max_eigenvals=2048)
    np.testing.assert_allclose(eig, eig_ref, rtol=1e-9, atol=1e-15)
    a, r2 = A._powerlaw_alpha(eig_ref)
    a_ref, _, _, r2_ref = alpha.get_powerlaw(eigen=eig_ref, trange=np.arange(10, 100))
    assert a == pytest.approx(float(a_ref), rel=1e-12, abs=1e-12)
    assert r2 == pytest.approx(float(r2_ref), rel=1e-12, abs=1e-12)
    assert A._rankme(eig_ref) == pytest.approx(float(rankme.get_rank(eig_ref)), rel=1e-12)


@pytest.mark.parametrize("label,eig", list(adv.degenerate_spectra()),
                         ids=[lab for lab, _ in adv.degenerate_spectra()])
def test_rankme_matches_reptrix_on_degenerate_spectra(ref, label, eig):
    _, rankme, _ = ref
    eig = np.sort(np.asarray(eig, dtype=np.float64))[::-1]
    assert A._rankme(eig) == pytest.approx(float(rankme.get_rank(eig)), rel=1e-12)
    # alpha needs ranks 11-100: every degenerate spectrum is too short or zero there.
    assert np.isnan(A._powerlaw_alpha(eig)[0])


def test_exact_power_law_is_recovered():
    """Known positive: on lambda_i = i^-alpha the weighted fit is exact."""
    for a in (0.5, 1.0, 2.3):
        eig = np.arange(1, 501, dtype=np.float64) ** -a
        assert A._powerlaw_alpha(eig / eig.sum())[0] == pytest.approx(a, rel=1e-10)


def test_evaluate_matches_get_alpha_and_get_rankme(ref):
    """The task's real evaluate() on injected states against reptrix on the
    cloud it scores: the last row of each document at the output state,
    stored as float32 by the cache."""
    alpha, rankme, _ = ref
    rng = np.random.default_rng(1)
    n, d, layers = 160, 192, 3
    scale = np.arange(1, d + 1) ** -0.6
    lengths = rng.integers(3, 7, size=n)
    docs = {l: [torch.tensor(rng.standard_normal((int(t), d)) * scale, dtype=torch.float32)
                for t in lengths] for l in range(layers)}
    out = inv.run_task_on_states("geometry_alpha_req", docs)
    X = torch.stack([doc[-1] for doc in docs[layers - 1]]).double()
    a_ref = alpha.get_alpha(X, max_eigenvals=2048)[0]
    assert out["alpha_req"] == pytest.approx(float(a_ref), rel=1e-9)
    assert out["rankme"] == pytest.approx(float(rankme.get_rankme(X)), rel=1e-9)
    assert out["_meta_n_points"] == n and out["_meta_n_eigenvalues"] == n
