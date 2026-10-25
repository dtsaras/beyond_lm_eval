"""Fixture spaces that this campaign's own verification kept missing.

Three distinct validation failures on 2026-09-07, all of which produced a
confident "verified" claim:

1. FIXTURES THAT DESTROY THEIR PAYLOAD (five occurrences). `(P + 1e6).astype(
   float32)` quantises a unit-scale cloud onto a 0.0625 lattice, so the task is
   handed garbage and correctly reports garbage. `on_grid_translation` builds
   translations that are exactly representable instead.

2. LADDERS THAT STOP SHORT. `dynamics_stability`'s self-identity axiom held at
   every rung through 1e38 and broke at 2.406e38, because a DERIVED quantity
   (the row norm, ~3x the row max) overflowed while the values themselves were
   representable. `scale_ladder` therefore ends at the largest value whose
   REDUCTIONS still fit, not at a round number.

3. FIXTURE SPACES THAT STRUCTURALLY EXCLUDE THE DEFECT. `geometry_spectral` was
   validated on 120 random spectra from full-rank Gaussians -- which never
   contain exact zeros -- and the defect was in a rank-deficient spectrum. More
   draws of the same kind would never have found it. `degenerate_spectra` and
   `wide_dynamic_range` construct the awkward cases deliberately.

Import these rather than hand-rolling a fixture; a hand-rolled one tends to be
drawn from the space where the code already works.
"""
from __future__ import annotations

import numpy as np

FLOAT32_MAX = 3.4028234663852886e38


def scale_ladder(reduction_depth: int = 2, n_terms: int = 1024, dtype=np.float32):
    """Multiplicative rungs ending at the last value whose REDUCTIONS survive.

    `reduction_depth` is how many times the data is multiplied by itself
    downstream (1 = plain sum, 2 = squared norm, 3 = Gram-of-Gram) and
    `n_terms` is how many of those products get ADDED. Both matter: the first
    version of this helper accounted only for the power and its own
    self-check failed, because `x**2` fitting does not mean `sum of d of them`
    fits. That is precisely the `dynamics_stability` defect -- the row max was
    representable and the row norm, a sum over coordinates, was not -- so a
    ladder that gets this wrong reproduces the bug it exists to catch.
    """
    budget = float(np.finfo(dtype).max) / max(1, n_terms)
    top = budget ** (1.0 / max(1, reduction_depth))
    return [1e-30, 1e-25, 1e-13, 1.0, 1e13, 1e25, top / 4.0, top]


def on_grid_translation(X, offset: float, dtype=np.float32):
    """Translate so the result is EXACTLY representable -- no payload lost.

    Returns `(baseline, translated, ok)`.

    **Compare against `baseline`, NOT against the original X.** This function
    SNAPS the cloud to the ULP grid at `offset` before translating, so the
    translated array is a faithful shift of the SNAPPED cloud, not of the one
    you passed in. Comparing a task's output on `X` against its output on
    `translated` therefore compares two different point sets and manufactures a
    difference that is not the code's fault.

    That is not hypothetical: a round-11 reviewer using the two-value version of
    this helper measured an apparent 14.3% homology shift and nH1 24 -> 1 from
    exactly this mistake, and only caught it because they re-derived the
    reference on the snapped cloud. The helper written to prevent fixture traps
    had a fixture trap in its signature. Returning the baseline makes the
    correct comparison the easy one.

    `ok` is False when the offset cannot carry this data at all, in which case a
    test should skip rather than blame the code.
    """
    X = np.asarray(X, dtype=np.float64)
    ulp = float(np.spacing(dtype(abs(offset)))) if offset else 0.0
    Xs = np.round(X / ulp) * ulp if ulp > 0 else X
    T = (Xs + offset).astype(dtype)
    ok = np.array_equal(T.astype(np.float64) - offset, Xs)
    return Xs.astype(dtype), T, bool(ok)


def degenerate_spectra():
    """Spectra a full-rank Gaussian never produces. Yields (label, eigenvalues)."""
    yield "two positives + 62 exact zeros", np.sort(np.array([1.0, 4.0] + [0.0] * 62))
    yield "single positive", np.array([1.0])
    yield "all identical", np.full(20, 3.0)
    yield "duplicated pairs", np.repeat(np.array([1.0, 2.0, 3.0]), 4)
    yield "one dominant + tiny tail", np.sort(np.r_[1e12, np.full(19, 1e-12)])
    yield "exact zeros interleaved", np.sort(np.r_[np.zeros(10), np.arange(1, 11.0)])


def wide_dynamic_range():
    """Clouds where centring destroys structure. Yields (label, X).

    The first two are DIRECTION A (centring itself is lossy: the ULP of the
    centroid exceeds the data's own gaps). The third is DIRECTION B -- centring
    in float64 is perfectly safe there, and the loss happens only if the cloud
    is narrowed to float32 BEFORE centring. `centring_is_safe` deliberately
    reports it as safe, because in float64 it is; use it with a dtype-order
    check, not on its own.
    """
    yield "collinear 0/1/1e25", np.array([[0.0], [1.0], [9.999999562023526e24]])
    yield "two lattices 2^-80 and 1/32", np.concatenate(
        [[j * 2.0 ** -80 for j in range(21)], [1.0 + j / 32 for j in range(21)]]
    )[:, None]
    d = np.array([4.0] + [1.0] * 15) / 64.0
    yield "1e6 + tiny diag", np.vstack([np.diag(d), -np.diag(d)]) + 1e6


def centring_is_safe(X):
    """True when subtracting the mean will not destroy resolution.

    The ULP of the centroid becomes the resolution floor of every coordinate,
    so compare it against the smallest nonzero gap actually present. Fires on
    the `magnitude` and `lid` fixtures and stays silent on healthy data,
    gpt2-shaped hidden states, and a legitimately translated cloud.
    """
    X = np.asarray(X, dtype=np.float64)
    ulp = float(np.max(np.spacing(np.abs(X.mean(axis=0)))))
    gaps = np.diff(np.unique(X))
    gaps = gaps[gaps > 0]
    return (True if gaps.size == 0 else bool(ulp < float(gaps.min())))


def narrowing_is_safe(X, dtype=np.float32):
    """DIRECTION B: would casting to `dtype` destroy structure before centring?

    Separate from `centring_is_safe` because the two failure directions are
    independent: `geometry_svd`'s `1e6 + diag([4,1..1]/64)` centres perfectly in
    float64 (so `centring_is_safe` says True, correctly) yet loses 5 distinct
    values down to 3 the moment it is narrowed, taking effective rank from
    7.411061418596 to exactly 1.0.
    """
    X = np.asarray(X, dtype=np.float64)
    return int(np.unique(X.astype(dtype)).size) == int(np.unique(X).size)
