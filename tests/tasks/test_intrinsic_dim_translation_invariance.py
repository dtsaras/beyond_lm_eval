"""Two-NN depends only on distance RATIOS, so a lossless translation cannot move it.

The rescale in `_compute_id` used to divide by an arbitrary coordinate max,
with a comment claiming that "only shifts exponents". That holds only for a
power-of-two divisor: dividing by an arbitrary max re-rounds every coordinate,
and Two-NN excludes duplicate and equal-radius queries by EXACT comparison,
so the rounding silently changed which queries were excluded.
"""
import numpy as np
import pytest

from blme.tasks.geometry.intrinsic_dim import IntrinsicDimensionTask


def _lattice():
    """500 distinct rows of a 33x33 integer lattice / 16 -- exact in float32."""
    G = np.stack(np.meshgrid(np.arange(-16, 17), np.arange(-16, 17)),
                 axis=-1).reshape(-1, 2)
    idx = np.random.default_rng(0).choice(len(G), 500, replace=False)
    return G[idx].astype(np.float32) / 16


# The released pdist + Facco/Ansuini value, constant over every offset.
_RELEASED = 2.170755946319


def test_integer_translations_do_not_move_the_estimate():
    """Was 2.170755946319 -> 3.765359458874 (+73.46%) at offset 2 alone."""
    fn = IntrinsicDimensionTask(config={})._compute_id
    P = _lattice()
    base = fn(P)["intrinsic_dimension"]
    assert base == pytest.approx(_RELEASED, rel=1e-11)

    worst, worst_off = 0.0, None
    for off in range(65):
        T = (P + off).astype(np.float32)
        # FIXTURE INTEGRITY: the translation must itself be lossless, or the
        # test is measuring the fixture rather than the task.
        assert len(np.unique(T, axis=0)) == 500
        v = fn(T)["intrinsic_dimension"]
        rel = abs(v - base) / abs(base)
        if rel > worst:
            worst, worst_off = rel, off
    assert worst < 1e-9, (
        f"translation moved the estimate by {worst:.3e} at offset {worst_off}")


def test_rescale_divisor_is_a_power_of_two():
    """A cloud whose max is not a power of two must not be re-rounded."""
    fn = IntrinsicDimensionTask(config={})._compute_id
    P = _lattice()
    # max == 1 (a power of two: division is a no-op) vs max == 3 (not one).
    assert np.abs(P).max() == 1.0
    assert np.abs(P + 2).max() == 3.0
    assert fn(P)["intrinsic_dimension"] == pytest.approx(
        fn((P + 2).astype(np.float32))["intrinsic_dimension"], rel=1e-12)


@pytest.mark.parametrize("scale", [1e-30, 1e-25, 1e-22, 1.0, 1e13, 1e25,
                                   2.406159650500376e38])
def test_scale_ladder_still_holds(scale):
    """The rescale exists to stop sklearn underflowing; that must survive."""
    fn = IntrinsicDimensionTask(config={})._compute_id
    X = np.random.default_rng(0).standard_normal((500, 32))
    X = X / np.abs(X).max()
    assert fn(X * scale)["intrinsic_dimension"] == pytest.approx(
        fn(X)["intrinsic_dimension"], rel=1e-12)
