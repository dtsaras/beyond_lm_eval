"""`centre_and_rescale` — the one safe order, pinned.

Seven tasks were found carrying the same preparation defect on 2026-09-07, and
every one of them had already been "fixed" for a scale bug at least once. The
three failure modes interact, so a fix for any one reintroduces another; these
tests pin all three at once, which is the only way to stop the cycle.

The fixtures are the ones that actually caught the defects in the field, not
invented ones.
"""
import math

import numpy as np
import pytest
import torch

from blme.tasks.geometry.utils import centre_and_rescale, safe_svdvals


# The `matrix_entropy` fixture: exactly representable, so any movement under a
# common offset is the implementation's, not the fixture's.
_WEI = [[-0.125, -0.375], [-0.3125, -0.1875], [-0.5, 0.125], [-0.25, -0.3125]]


def assert_payload_survives(original, transformed, dtype=np.float32):
    """A ladder fixture must prove its OWN input is still faithful.

    Four fixtures in one session (2026-09-07) reported a task defect that was
    really fixture destruction: `(P + 1e6).astype(np.float32)` quantises a
    unit-scale cloud onto a 0.0625 grid, because float32's resolution at 1e6 is
    0.0625. The task then correctly reports garbage from garbage, and the test
    blames the task.

    Overflow is the easy direction and is already asserted elsewhere
    (`isfinite`). This is the QUANTISATION direction: after the transform, the
    fixture's points must still be distinguishable from one another at roughly
    the precision the untransformed cloud had.
    """
    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(transformed, dtype=dtype).astype(np.float64)
    assert np.isfinite(b).all(), "fixture overflowed; it tests the harness"
    # centred spreads must be preserved: the transform is meant to be an offset
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean(axis=0, keepdims=True)
    denom = np.abs(ac).max()
    if denom == 0:
        return
    rel = np.abs(ac - bc).max() / denom
    assert rel < 1e-3, (
        f"fixture lost its payload to quantisation (rel {rel:.2e}); the test "
        f"would be measuring the dtype, not the code")


def test_the_payload_guard_catches_a_destroyed_fixture():
    """Guard on the guard: the check must actually fire on the bad fixture."""
    rng = np.random.default_rng(11)
    P = rng.standard_normal((60, 5))
    assert_payload_survives(P, P + 1e2)            # survivable offset
    with pytest.raises(AssertionError, match="quantisation"):
        assert_payload_survives(P, P + 1e6)        # the shape that fooled me 4x


@pytest.mark.parametrize("offset", [0.0, 1e2, 1e6, 1e8, 1e12])
def test_centring_is_exact_under_a_common_offset(offset):
    """Failure mode 1: cast-then-centre quantises the offset away.

    In float32 the payload of `1e6 + 0.125` is already gone, so subtracting the
    mean afterwards cannot recover it. `matrix_entropy` moved -80.17% at +1e6
    and NaN at +1e8 for a statistic Wei et al. define to be translation
    invariant. Centred values must be BIT-IDENTICAL across offsets.
    """
    base, _ = centre_and_rescale(torch.tensor(_WEI, dtype=torch.float64))
    got, _ = centre_and_rescale(torch.tensor(_WEI, dtype=torch.float64) + offset)
    assert torch.equal(base, got), (
        f"offset {offset:g} changed the centred data: {base.tolist()} vs {got.tolist()}")


@pytest.mark.parametrize("scale", [1.0, 1e-30, 1e-13, 1e25, 1e38,
                                   2.406159650500376e38])
def test_subtraction_does_not_overflow_at_the_top_of_float32(scale):
    """Failure mode 2: centring in float32 overflows in the SUBTRACTION.

    A unit-max cloud scaled to 2.4e38 has centred entries beyond 3.4e38, so
    `X - mean` returns inf. In `collapse` this survived wiring `safe_svdvals`,
    because the overflow was one line ABOVE the SVD.
    """
    rng = np.random.default_rng(3)
    X = rng.standard_normal((120, 16))
    X = (X / np.abs(X).max()).astype(np.float32)          # representable at every rung
    Xs = (X.astype(np.float64) * scale).astype(np.float32)
    assert np.isfinite(Xs).all(), "fixture itself overflowed; the test would be vacuous"
    Z, _ = centre_and_rescale(Xs)
    assert np.isfinite(Z).all(), f"centring produced non-finite values at scale {scale:g}"


def test_effective_rank_is_scale_invariant_through_the_helper():
    """Roy-Vetterli Property 2: erank(cA) == erank(A), end to end.

    This is the property `collapse` violated, returning NaN / 5.213422550 / NaN
    at 2.406159650500376e38 and reporting collapse_ratio 1.0 instead of
    0.511027294.
    """
    def erank(M):
        Z, _ = centre_and_rescale(M)
        S = np.asarray(safe_svdvals(Z), dtype=np.float64)
        lam = S ** 2
        total = lam.sum()
        if not np.isfinite(total) or total <= 0:
            return float("nan")
        p = lam[lam > 0] / total
        return float(np.exp(-(p * np.log(p)).sum()))

    rng = np.random.default_rng(3)
    X = rng.standard_normal((120, 16))
    X = (X / np.abs(X).max()).astype(np.float32)
    base = erank(X)
    assert np.isfinite(base)
    for s in (1e-30, 1e-13, 1e25, 1e38, 2.406159650500376e38):
        got = erank((X.astype(np.float64) * s).astype(np.float32))
        assert got == pytest.approx(base, rel=1e-8), f"erank moved at scale {s:g}"


@pytest.mark.parametrize("scale", [1e-30, 1e-13, 1e25, 1e38])
def test_exponent_lets_an_absolute_scale_caller_undo_the_rescale(scale):
    """Scale-FREE callers ignore `exp2`; absolute-scale callers must not.

    Returning it is what makes the helper usable by both, instead of forcing a
    second, unrescaled code path -- which is how `homology` acquired a defect
    that inflated every normalised summary by the rescale factor.
    """
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((40, 6)) / 4.0)
    X = (X / np.abs(X).max() * scale)
    Z, exp2 = centre_and_rescale(X, centre=False)
    recovered = np.ldexp(np.abs(Z).max(), exp2)
    assert recovered == pytest.approx(float(np.abs(X).max()), rel=1e-6)


def test_ordinary_data_takes_no_rescale_at_all():
    """The gate must be a no-op in the ordinary range.

    That is what makes this safe to adopt everywhere: inside 1e-6..1e6 the
    helper only promotes and centres, so no shipped number can move because a
    call site started using it.
    """
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 8))
    Z, exp2 = centre_and_rescale(X)
    assert exp2 == 0
    assert np.allclose(Z, X - X.mean(axis=0, keepdims=True), rtol=0, atol=0)


def test_power_of_two_rescale_is_exact_not_a_division():
    """`ldexp`, not `/ max`. A division introduces rounding, which made whether
    the guard fired observable in the result (codex round 5, 2026-09-05)."""
    rng = np.random.default_rng(2)
    X = (rng.standard_normal((30, 5)) * 1e20)
    Z, exp2 = centre_and_rescale(X, centre=False)
    # every element must be an exact power-of-two shift of the original
    back = np.ldexp(Z, exp2)
    assert np.array_equal(back, np.asarray(X, dtype=np.float64))
