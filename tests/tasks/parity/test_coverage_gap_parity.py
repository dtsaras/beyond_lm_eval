"""Executed reference parity for the six tasks that had none.

A coverage sweep over all 51 paper-core tasks found 45 with a parity or
reference test naming them and 6 without, so their implementation faithfulness
rested on code reading alone. Each reference below is derived independently
from the cited definition, NOT transcribed from the BLME implementation.
"""
import math

import numpy as np
import pytest


# --------------------------------------------------------------------------
# topology_persistence_entropy — Atienza et al. Def 3.1 / giotto-tda
# --------------------------------------------------------------------------
def test_persistence_entropy_matches_scipy_entropy_base2():
    """PE = -sum p_i log2 p_i with p_i = l_i / sum(l).

    giotto-tda computes this as scipy.stats.entropy(lifespans, base=2), which
    normalises internally -- an independent route to the same number.
    """
    scipy_stats = pytest.importorskip("scipy.stats")
    from blme.tasks.topology.persistence_entropy import _persistence_entropy

    rng = np.random.default_rng(0)
    for _ in range(20):
        lifespans = rng.exponential(1.0, size=rng.integers(2, 40))
        got = _persistence_entropy(list(lifespans))
        ref = float(scipy_stats.entropy(lifespans, base=2))
        assert got == pytest.approx(ref, rel=1e-12), f"{got!r} != {ref!r}"


def test_persistence_entropy_is_scale_invariant():
    """p_i is a ratio, so a common positive rescale cannot move PE."""
    from blme.tasks.topology.persistence_entropy import _persistence_entropy
    lifespans = list(np.random.default_rng(1).exponential(1.0, size=25))
    base = _persistence_entropy(lifespans)
    for s in (1e-14, 1e-6, 1e6, 1e14):
        assert _persistence_entropy([x * s for x in lifespans]) == \
            pytest.approx(base, rel=1e-12), f"PE moved at scale {s}"


def test_persistence_entropy_extremes():
    """Uniform lifespans maximise PE at log2(m); one dominant bar drives it to 0."""
    from blme.tasks.topology.persistence_entropy import _persistence_entropy
    for m in (2, 8, 64):
        assert _persistence_entropy([1.0] * m) == pytest.approx(math.log2(m), rel=1e-12)
    assert _persistence_entropy([1.0, 1e-18]) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# geometry_mp_bulk_deviation — Marchenko-Pastur, analytic
# --------------------------------------------------------------------------
def test_mp_bulk_edge_matches_analytic_law():
    from blme.tasks.geometry.rmt_bulk import _mp_bulk_edge
    for gamma in (0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 10.0):
        assert _mp_bulk_edge(gamma) == pytest.approx(
            (1.0 + math.sqrt(gamma)) ** 2, rel=1e-15)


def test_mp_bulk_edge_bounds_a_real_wishart_spectrum():
    """The largest eigenvalue of a white Wishart converges to (1+sqrt(gamma))^2.

    This checks the edge against SAMPLED random matrices, not just against a
    restatement of its own formula.
    """
    from blme.tasks.geometry.rmt_bulk import _mp_bulk_edge
    rng = np.random.default_rng(0)
    for N, D in ((4000, 400), (4000, 1000), (2000, 1000)):
        X = rng.standard_normal((N, D))
        ev = np.linalg.eigvalsh(X.T @ X / N)
        edge = _mp_bulk_edge(D / N)
        assert ev.max() <= edge * 1.06, f"gamma={D/N}: max {ev.max()} vs edge {edge}"
        assert ev.max() >= edge * 0.90, f"gamma={D/N}: max {ev.max()} vs edge {edge}"


# --------------------------------------------------------------------------
# dynamics_generation_diversity — Li et al. 2016 Distinct-n
# --------------------------------------------------------------------------
def _ref_distinct_n(tokens, n):
    """Li et al. 2016 Sec 5.2: distinct n-gram TYPES / total tokens."""
    if len(tokens) < n:
        return 0.0
    grams = {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}
    return len(grams) / len(tokens)


def _ref_distinct_n_corpus(completions, n):
    """Collection-level: union of types over the collection / total tokens."""
    types, total = set(), 0
    for c in completions:
        total += len(c)
        for i in range(len(c) - n + 1):
            types.add(tuple(c[i:i + n]))
    return len(types) / total if total else 0.0


def test_distinct_n_matches_independent_reference():
    from blme.tasks.dynamics.generation_diversity import _distinct_n
    rng = np.random.default_rng(0)
    for _ in range(30):
        toks = list(rng.integers(0, 12, size=int(rng.integers(1, 60))))
        for n in (1, 2, 3):
            assert _distinct_n(toks, n) == pytest.approx(
                _ref_distinct_n(toks, n), rel=1e-15)


def test_distinct_n_corpus_is_a_collection_ratio_not_a_mean():
    """Ten duplicate [1,2] completions: 0.1 as a collection ratio, 1.0 as a mean."""
    from blme.tasks.dynamics.generation_diversity import _distinct_n_corpus
    comps = [[1, 2]] * 10
    assert _distinct_n_corpus(comps, 1) == pytest.approx(
        _ref_distinct_n_corpus(comps, 1), rel=1e-15)
    assert _distinct_n_corpus(comps, 1) == pytest.approx(0.1, rel=1e-12)
    assert np.mean([_ref_distinct_n(c, 1) for c in comps]) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# geometry_contextualization — Ethayarajh 2019 IntraSim
# --------------------------------------------------------------------------
def _ref_intrasim(V):
    """IntraSim_l(s) = mean_i cos(v_i, mean_j v_j), computed directly."""
    V = np.asarray(V, dtype=np.float64)
    c = V.mean(axis=0)
    nc = np.linalg.norm(c)
    if nc == 0:
        return float("nan")
    cos = [float(v @ c / (np.linalg.norm(v) * nc)) for v in V
           if np.linalg.norm(v) > 0]
    return float(np.mean(cos)) if cos else float("nan")


def test_intrasim_matches_independent_reference():
    from blme.tasks.geometry.contextualization import _intra_sentence_mean_cosine
    rng = np.random.default_rng(0)
    for _ in range(20):
        V = rng.standard_normal((int(rng.integers(2, 30)), 16))
        assert _intra_sentence_mean_cosine(V) == pytest.approx(
            _ref_intrasim(V), rel=1e-10)


def test_intrasim_is_invariant_to_positive_rescaling():
    """Ethayarajh Defs 1-2 are exactly invariant under a common rescale."""
    from blme.tasks.geometry.contextualization import _intra_sentence_mean_cosine
    V = np.random.default_rng(3).standard_normal((12, 16))
    base = _intra_sentence_mean_cosine(V)
    for s in (1e-30, 1e-8, 1e8, 1e25, 2.406159650500376e38):
        got = _intra_sentence_mean_cosine(V * s)
        assert got == pytest.approx(base, rel=1e-9), f"IntraSim moved at {s}: {got}"


# --------------------------------------------------------------------------
# dynamics_coe — Wang et al. 2025, Alsace08/Chain-of-Embedding score.py
# --------------------------------------------------------------------------
def _ref_coe(chain):
    """CoE-R (Eq. 5) and CoE-C (Eq. 6/7) in the official normalized/normalized form.

    Mag_i = ||h_{i+1} - h_i||, Ang_i = angle(h_i, h_{i+1}),
    Z_Mag = ||h_L - h_0||, Z_Ang = angle(h_0, h_L),
    CoE-R = mean(Mag_i/Z_Mag) - mean(Ang_i/Z_Ang),
    CoE-C = |mean( (Mag_i/Z_Mag) * exp(i * Ang_i/Z_Ang) )|.
    """
    H = [np.asarray(c, dtype=np.float64).ravel() for c in chain]
    def ang(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0)))
    L = len(H) - 1
    mags = [float(np.linalg.norm(H[i + 1] - H[i])) for i in range(L)]
    angs = [ang(H[i], H[i + 1]) for i in range(L)]
    z_mag = float(np.linalg.norm(H[-1] - H[0]))
    z_ang = ang(H[0], H[-1])
    mn = [m / z_mag for m in mags]
    an = [a / z_ang for a in angs]
    x = np.mean([m * math.cos(a) for m, a in zip(mn, an)])
    y = np.mean([m * math.sin(a) for m, a in zip(mn, an)])
    return float(np.mean(mn) - np.mean(an)), float(math.hypot(x, y))


def test_coe_matches_independent_reference():
    torch = pytest.importorskip("torch")
    from blme.tasks.dynamics.coe import _coe_from_chain
    rng = np.random.default_rng(0)
    for _ in range(15):
        L = int(rng.integers(2, 10))
        chain = [torch.from_numpy(rng.standard_normal(24).astype(np.float64))
                 for _ in range(L + 1)]
        got = _coe_from_chain(chain)
        r_ref, c_ref = _ref_coe([c.numpy() for c in chain])
        assert got["coe_r"] == pytest.approx(r_ref, rel=1e-9), \
            f"coe_r {got['coe_r']!r} != {r_ref!r}"
        assert got["coe_c"] == pytest.approx(c_ref, rel=1e-9), \
            f"coe_c {got['coe_c']!r} != {c_ref!r}"


def test_coe_is_dimensionless_under_rescaling():
    """Every CoE quantity is a ratio to Z_Mag / Z_Ang, so a rescale cancels."""
    torch = pytest.importorskip("torch")
    from blme.tasks.dynamics.coe import _coe_from_chain
    rng = np.random.default_rng(5)
    chain = [torch.from_numpy(rng.standard_normal(24).astype(np.float64))
             for _ in range(6)]
    base = _coe_from_chain(chain)
    for s in (1e-30, 1e-8, 1e8, 1e25, 2.406159650500376e38):
        got = _coe_from_chain([c * s for c in chain])
        for key in ("coe_r", "coe_c"):
            assert got[key] == pytest.approx(base[key], rel=1e-9), \
                f"{key} moved at scale {s}: {got[key]!r} vs {base[key]!r}"


# --------------------------------------------------------------------------
# topology_persistence_landscape — GUDHI representations.Landscape
# --------------------------------------------------------------------------
def test_persistence_landscape_matches_bubenik_and_gudhi_up_to_convention():
    """lambda_k(t) is the k-th largest tent function (Bubenik 2015).

    BLME implements the ORIGINAL definition,
    lambda_k(t) = k-max_i max(0, min(t - b_i, d_i - t)),
    and matches it to 0.0 exactly.

    GUDHI's `representations.Landscape` uses the ROTATED-frame convention,
    which is exactly sqrt(2) times that -- measured ratio 1.414214 on every
    nonzero sample, max|gudhi - sqrt2*bubenik| = 1.67e-16. Neither is wrong;
    they are different conventions for the same object. It is pinned here
    because anyone comparing a BLME landscape number against a GUDHI-derived
    one must account for the factor.
    """
    gudhi = pytest.importorskip("gudhi.representations")
    from blme.tasks.topology.persistence_landscape import _compute_landscape
    rng = np.random.default_rng(0)
    births = rng.uniform(0, 1, 12)
    deaths = births + rng.uniform(0.01, 0.6, 12)
    dgm = np.c_[births, deaths]

    k, res = 3, 100
    lo, hi = float(births.min()), float(deaths.max())
    grid = np.linspace(lo, hi, res)
    tents = np.array([[max(0.0, min(t - b, d - t)) for t in grid]
                      for b, d in dgm])
    bubenik = np.sort(tents, axis=0)[::-1][:k]

    got = np.asarray(_compute_landscape(dgm, n_landscapes=k, n_points=res))
    assert got.shape == (k, res), got.shape
    assert np.abs(got - bubenik).max() == 0.0, (
        f"BLME departs from the Bubenik definition by "
        f"{np.abs(got - bubenik).max():.3e}")

    ref = gudhi.Landscape(num_landscapes=k, resolution=res,
                          sample_range=[lo, hi]).fit_transform([dgm])[0]
    assert np.allclose(ref.reshape(k, res), np.sqrt(2) * bubenik, atol=1e-12), (
        "GUDHI is no longer sqrt(2) x Bubenik; the convention note is stale")


def test_persistence_landscape_scale_equivariance():
    """Landscapes carry units: scaling the diagram by s scales lambda by s."""
    from blme.tasks.topology.persistence_landscape import _compute_landscape
    rng = np.random.default_rng(2)
    b = rng.uniform(0, 1, 10); d = b + rng.uniform(0.05, 0.5, 10)
    dgm = np.c_[b, d]
    base = np.asarray(_compute_landscape(dgm, n_landscapes=3, n_points=64))
    for s in (1e-6, 1e-3, 1e3, 1e6):
        got = np.asarray(_compute_landscape(dgm * s, n_landscapes=3, n_points=64))
        assert np.allclose(got, base * s, rtol=1e-9), f"not equivariant at {s}"
