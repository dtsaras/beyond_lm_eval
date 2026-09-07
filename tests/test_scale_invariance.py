"""SCALE INVARIANCE is a contract, and this file is where it is enforced.

Rounds 2-4 of the 2026-09 adversarial campaign were dominated by a single
defect class, found again and again in new places:

    a metric whose DEFINITION is invariant to a positive rescaling of the
    representation, implemented with ARITHMETIC that is not.

Two mechanisms produce it. An **absolute epsilon** (``norm + 1e-10``,
``std > 1e-12``, ``sd <= 1e-12``) hard-codes a magnitude into a scale-free
formula. And every **norm squares its input**, so a float32 norm underflows to
0.0 -- or overflows to inf -- long before the data itself does.

`blme.tasks.geometry.utils` exports `safe_unit_rows`, `safe_unit_rows_torch`
and `safe_norm` for the second mechanism: they factor out the largest magnitude
first, which is exact in binary floating point because it only shifts
exponents. This file pins the resulting contract for every kernel that has it,
so the next person to touch these files finds out immediately.

Every case below states the exact value the metric must hold, and the value it
returned BEFORE the fix, so a regression is recognisable rather than merely
red.
"""
import numpy as np
import pytest
import torch

# The top of this list matters. `SCALES` used to stop at 1e20, which cannot
# reach the boundary where a power-of-two guard breaks: `2.0 ** round(log2(g))`
# overflows float32 to `inf` for any g >= 2^127.5 = 2.4062e38 -- the TOP 29% OF
# THE FLOAT32 RANGE -- and that bug shipped in three files unnoticed because no
# test probed there (independent Opus review, 2026-09-05).
SCALES = (1e0, 1e-6, 1e-13, 1e-23, 1e-30, 1e6, 1e20, 1e30, 2.406159650500376e38)


def _at_scale(X, s):
    """``X`` rescaled so its LARGEST MAGNITUDE is exactly ``s``.

    Multiplying a fixture by a large scale overflows the fixture itself -- a
    matrix with entries up to 5 times 2.4e38 is 1.2e39, past float32's maximum,
    which tests the multiplication rather than the metric. Normalising first
    puts the input exactly at the requested magnitude, which is what these
    cases mean to probe.
    """
    X = np.asarray(X)
    m = float(np.max(np.abs(X)))
    if m == 0:
        return X
    return (X / m * np.asarray(s, dtype=X.dtype)).astype(X.dtype)


# Tolerance note. A kernel that consumes float32 STORAGE cannot be invariant
# to better than float32 rounding: multiplying a float32 array by a scale
# changes its low bits, so an SVD of the rescaled array differs at ~1e-8
# relative (float32 eps is 1.2e-7). `rel=1e-6` is therefore the honest bar for
# those, and `1e-9` for kernels that work in float64 throughout. Both are far
# tighter than every defect this file guards, which were 3% to 100% errors.
def _const(vals, name, rel=1e-9):
    vals = [float(v) for v in vals]
    base = vals[0]
    for v in vals[1:]:
        assert np.isfinite(v), f"{name}: non-finite at some scale: {vals}"
        assert abs(v - base) <= rel * max(abs(base), 1e-300), (
            f"{name} is not scale-invariant: {vals}"
        )


def test_isotropy_participation_ratio_and_anisotropy():
    """`(sum L)^2 / sum(L^2)` and Ethayarajh Eq. 4 are both ratios.

    Before: participation ratio 4.924101 at scale 1 but 4.812010 at 1e-4
    (a fixed +1e-12 in the denominator), and the exact mean cosine 0.288888889
    read 0.145793468 at 3e-23 (a float32 norm underflow).
    """
    from blme.tasks.geometry.isotropy import _svd_metrics_for_layer
    rng = np.random.default_rng(0)
    X = (rng.normal(size=(40, 6)) * np.array([5., 4., 3., 2., 1., .5])).astype(np.float32)
    prs, cos = [], []
    for s in SCALES:
        m = _svd_metrics_for_layer(_at_scale(X, s))
        prs.append(m["participation_ratio"])
        cos.append(m["avg_cosine_similarity"])
    _const(prs, "participation_ratio", rel=1e-6)      # float32 storage
    _const(cos, "avg_cosine_similarity", rel=1e-6)


def test_isoscore_is_scale_invariant():
    """Rudman Eq. 4.1 is a ratio of spectral moments.

    Before: -12.7% at 3e-22, +30.0% at 1e-22, NaN at 3e-23 and at 1e19.
    """
    from blme.tasks.geometry.isotropy import _isoscore
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8)) * np.array([4., 3., 2., 1., .5, .2, .1, .05])
    _const([_isoscore(_at_scale(X, s)) for s in SCALES], "isoscore")


def test_lid_matches_official_and_is_scale_invariant():
    """Ma et al. Eq. (4) enters distances only through the ratio d_i/d_k.

    Before: an absolute ``d_k < 1e-10`` floor dropped the whole cloud out of
    the estimator at scale 1e-11, where official ``mle_batch`` still returned
    6.99074513291813.
    """
    from blme.tasks.geometry.lid import _lid_mle
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    d0 = np.sort(np.linalg.norm(X[0] - X[1:], axis=1))[:20]
    vals = []
    for s in SCALES:
        d = np.sort(np.linalg.norm(_at_scale(X[0] - X[1:], s), axis=1))[:20]
        vals.append(_lid_mle(d, 20))
    _const(vals, "lid_mle")
    official = -20 / np.sum(np.log(d0[:20] / d0[19]))
    assert vals[0] == pytest.approx(official, rel=1e-12)


def test_matrix_entropy_is_scale_invariant():
    """Def. 4.1 row-normalises, so the entropy cannot depend on scale.

    Before: 1.000000003 at eps 1.1e-12 collapsing to 9.97e-29 at 1.0e-12, and
    0.482747648 at 3e-23 (float32 row-norm underflow).
    """
    from blme.tasks.geometry.matrix_entropy import _matrix_entropy
    Z = torch.eye(4, dtype=torch.float32)
    _const([_matrix_entropy(torch.as_tensor(_at_scale(Z.numpy(), s)))["entropy"]
            for s in SCALES], "matrix_entropy", rel=1e-6)   # float32 storage


def test_linear_cka_is_scale_and_translation_invariant():
    """Kornblith CKA is invariant to isotropic rescaling AND to a common shift.

    Before: 0.132094427492 -> 4.41881768414e-5 at scale 1e-5 (a +1e-12
    denominator), and 24.17% error at offset 1e3 (Gram formed before centring).
    """
    from blme.tasks.geometry.mutual_info import _normalized_linear_hsic as cka
    rng = np.random.default_rng(0)
    X = torch.tensor(rng.normal(size=(60, 10)))
    Y = torch.tensor(rng.normal(size=(60, 10)) * .6 + rng.normal(size=(60, 10)) * .1)
    _const([cka(torch.as_tensor(_at_scale(X.numpy(), s)),
                 torch.as_tensor(_at_scale(Y.numpy(), s))) for s in SCALES],
           "cka(scale)")
    _const([cka(X + o, Y + o) for o in (0.0, 1e3, 1e4, 1e6)], "cka(offset)", rel=1e-8)


def test_rankme_and_stable_rank_are_scale_invariant():
    """RankMe (Garrido Eq. 2) and stable rank are both spectral ratios.

    Before: RankMe on 1e-20 * I_2 read 1.000000368 -- a rank-ONE answer for a
    perfectly isotropic spectrum -- because eps was in the denominator rather
    than added to the ratio. Stable rank of diag(10..1) read 4.000000 at 1e-23
    against an exact 3.850000.
    """
    from blme.tasks.geometry.schatten import _rankme
    _const([_rankme(np.array([s, s], dtype=np.float64)) for s in SCALES], "rankme")
    S = np.arange(10, 0, -1, dtype=np.float64)
    srs = [float(np.sum((_at_scale(S, s) / _at_scale(S, s)[0]) ** 2)) for s in SCALES]
    _const(srs, "stable_rank")
    assert srs[0] == pytest.approx(3.85, rel=1e-12)


def test_pl_mle_alpha_is_scale_invariant():
    """The PL-MLE alpha depends only on ratios lambda_i / x_min.

    Before: an absolute ``lam > 1e-12`` cut silently changed the tail
    population below that magnitude.
    """
    from blme.tasks.geometry.spectral import _pl_mle_alpha_ks
    S = np.arange(10, 0, -1, dtype=np.float64)
    _const([_pl_mle_alpha_ks(_at_scale(S, s) ** 2)[0] for s in SCALES],
           "alpha_official")


def test_block_influence_of_an_identity_block_is_zero_at_every_scale():
    """BI = 1 - cos(h, h) = 0 exactly, whatever the magnitude of h.

    Before: torch's default cosine eps is an ABSOLUTE 1e-8, so rescaling one
    identity state from norm 5.477 to 5.477e-9 moved BI to 0.699999988; and the
    norm's own arithmetic underflowed at 1e-23, giving -0.070435762.
    """
    from blme.tasks.geometry.effective_depth import _block_influence
    rng = np.random.default_rng(0)
    h = torch.tensor(rng.normal(size=(1, 6, 12)), dtype=torch.float32)
    for s in SCALES:
        x = torch.as_tensor(_at_scale(h.numpy(), s)).float()
        # float32 eps is 1.19e-7, and cos(v, v) is not exactly 1 in float32 at
        # every magnitude, so the honest bar here is the storage precision.
        assert _block_influence(x, x) == pytest.approx(0.0, abs=1e-6), f"scale {s}"


def test_magnitude_distances_are_translation_invariant():
    """Magnitude is a functional of the distance matrix, which is
    translation-invariant.

    Before: ``||x||^2 + ||y||^2 - 2 x.y`` catastrophically cancels once the
    cloud sits far from the origin -- a 1e8 shift manufactured 6 false ZERO
    pairs on a 40-point cloud.
    """
    from blme.tasks.geometry.magnitude import _pairwise_distance_matrix as pdm
    rng = np.random.default_rng(343)
    X = rng.normal(size=(40, 9))
    u = rng.normal(size=9)
    u /= np.linalg.norm(u)
    D0 = pdm(X)
    for shift in (1e4, 1e6, 1e8):
        D = pdm(X + shift * u)
        assert int((D == 0).sum()) == len(X), f"false zero pairs at shift {shift}"
        assert np.abs(D - D0).max() < 1e-6 * max(1.0, shift * 1e-8)


def test_pow2_rescale_survives_both_float32_boundaries():
    """The magnitude guards rescale by a power of two; that must never overflow.

    `2.0 ** round(log2(gmax))` is wrong at BOTH ends and shipped in three
    files: it reaches 2**128 for any gmax >= 2^127.5 = 2.4062e38 (the top 29%
    of the float32 range), which is `inf`, so the divisor became inf and the
    data became 0 -- MNN/T read 0.5773502588 at 2.406159650500376e38 and
    **0.0 at the very next float32**, a silently finite 100% error. Using
    `floor` instead underflows at the bottom: at gmax = 1e-45 it asks for
    2**-150, which is 0 (independent Opus review, 2026-09-05).
    """
    from blme.tasks.geometry.utils import pow2_rescale
    for a in (3.4e38, 2.406159650500376e38, 2.4e38, 1.0, 1e-38, 1e-45):
        X = torch.tensor([[a, a / 2]], dtype=torch.float32)
        Y = pow2_rescale(X)
        assert bool(torch.isfinite(Y).all()), f"non-finite at {a}"
        assert float(Y.abs().max()) > 0.0, f"flushed to zero at {a}"


def test_matrix_entropy_holds_across_the_whole_float32_range():
    """Including past the boundary where the old guard overflowed."""
    from blme.tasks.geometry.matrix_entropy import _matrix_entropy
    X = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.]],
                     dtype=torch.float32)
    base = _matrix_entropy(X)["entropy"]
    for a in (1e-38, 1e-20, 1e20, 1e30, 2.4e38, 2.406159650500376e38, 3.4e38):
        got = _matrix_entropy(X * a)["entropy"]
        assert got == pytest.approx(base, rel=1e-6), f"scale {a}: {got} vs {base}"


def test_pl_mle_ks_is_not_data_independent_at_the_shortest_tail():
    """A KS criterion that is CONSTANT cannot select anything.

    A round-4 reviewer reported that the released WWFit uses a bias-corrected
    ``(n-1)/sum-log`` Hill form below 20 tail points. I implemented it without
    checking against live WeightWatcher, and it silently destroyed the
    estimator: with the ``(n-1)`` numerator the KS statistic at ``n_tail = 2``
    collapses to the DATA-INDEPENDENT constant ``1 - e^-1 - 0.5 =
    0.132120558828558``, so the scan picked a 2-point tail on essentially every
    matrix. Against live WeightWatcher 0.7.7 the plain-``n`` form matches to
    mean 3.634e-15 while ``(n-1)`` was off by mean 0.266 / max 6.64 over gpt2's
    48 modules (independent Opus review, 2026-09-05).

    This test pins the property that made the bug detectable at all: the KS
    value at the shortest admissible tail must DEPEND ON THE DATA.
    """
    from blme.tasks.geometry.spectral import _pl_mle_alpha_ks
    ks_seen = set()
    for tail in ([1.0, 10.0], [1.0, 1e6], [3.0, 3.0001], [2.0, 100.0]):
        t = np.asarray(tail)
        x, k = t[0], t.size
        d = float(np.log(t / x).sum())
        alpha = 1.0 + k / d                     # plain n, as shipped
        emp = np.arange(k) / k
        ks_seen.add(round(float(np.max(np.abs(emp - (1.0 - (x / t) ** (alpha - 1.0))))), 12))
    # The plain-n KS at n_tail=2 is itself a constant (0.364664716763387), but
    # it is LARGE, so a 2-point tail never wins against a real fit. That is the
    # property to check end to end.
    rng = np.random.default_rng(0)
    for _ in range(3):
        lam = rng.pareto(2.5, 200) + 1.0
        alpha, xmin, ks, n_tail = _pl_mle_alpha_ks(lam)
        assert n_tail > 100, f"KS selected a {n_tail}-point tail on planted Pareto data"
        assert 2.5 < alpha < 4.5, f"alpha {alpha} does not recover the planted 3.5"


def test_isoscore_is_translation_invariant():
    """IsoScore centres the cloud, so a common OFFSET must cancel exactly.

    This case was MISSING from the contract file, and its absence let a real
    defect through: `_isoscore` subtracted a float32 mean from float32 data
    before promoting, and a float32 cloud translated by +1e6 read
    0.001783276461 against released IsoScore 2.0.1's 0.995837092400 -- a
    99.82% error (independent Opus review, 2026-09-05). Promoting BEFORE the
    mean restores it.

    The bar is set by the INPUT's own precision: at +1e8 the float32 spacing is
    about 8, which destroys the cloud before the estimator sees it, so that
    offset is outside the representable domain and is not asserted.
    """
    from blme.tasks.geometry.isotropy import _isoscore
    rng = np.random.default_rng(0)
    X = (rng.normal(size=(200, 8))
         * np.array([4., 3., 2., 1., .5, .2, .1, .05])).astype(np.float32)
    base = _isoscore(X)
    for off in (1e2, 1e4, 1e6):
        got = _isoscore((X + np.float32(off)).astype(np.float32))
        assert got == pytest.approx(base, rel=1e-3), f"offset {off}: {got} vs {base}"


def test_safe_norm_beats_the_naive_norm_where_it_matters():
    """The helper itself: exact where ``x.norm()`` flushes to zero.

    Before: a gradient with 8 coordinates at 1.250000039e-23 reported norm 0.0
    against a true 3.535534017e-23, and two gradients at -5.000000097707407e-25
    gave norm 0.0 and turned SAM sharpness into NaN.
    """
    from blme.tasks.geometry.utils import safe_norm
    for a in (1e-23, 5e-25, 1e-30, 1e-38):
        g = torch.full((8,), a, dtype=torch.float32)
        assert safe_norm(g) == pytest.approx(a * np.sqrt(8), rel=1e-6)
        assert float(g.norm()) == 0.0, "the naive form is expected to underflow here"
    assert safe_norm(torch.zeros(5)) == 0.0


def test_safe_unit_rows_is_exact_at_every_scale():
    """Row normalisation must return the SAME unit vector at any magnitude."""
    from blme.tasks.geometry.utils import safe_unit_rows, safe_unit_rows_torch
    v = np.array([[3.0, 4.0]], dtype=np.float32)
    for a in (1.0, 1e-13, 3e-23, 1e-30, 1e-38):
        got = safe_unit_rows(v * np.float32(a))[0]
        assert got == pytest.approx([0.6, 0.8], rel=1e-6), f"numpy at {a}"
        gott = safe_unit_rows_torch(torch.tensor(v * np.float32(a)))[0]
        assert gott.tolist() == pytest.approx([0.6, 0.8], rel=1e-6), f"torch at {a}"
    # An exactly zero row has no direction and stays zero.
    assert safe_unit_rows(np.zeros((1, 3), dtype=np.float32))[0].tolist() == [0, 0, 0]


def test_mean_squared_overlap_is_scale_invariant_and_never_goes_negative():
    """Superposition MSO: the kernel that reached a "provably impossible" state.

    `_mean_squared_overlap` is a mean of SQUARED COSINES, so it is invariant to
    a positive rescaling of the whole matrix and is bounded below by zero. The
    float32 implementation was neither. Measured on a 3000x64 Gaussian
    (red-team review, 2026-09-05):

      scale 1e-25  ->  **nan**    (sum of squares 6.4e-49 underflows float32's
                                   1.4e-45 subnormal floor, so `W.norm(dim=1)`
                                   reports 0.0 for EVERY row, all 3000 are
                                   dropped as "zero rows", and V < 2)
      scale 1e19   ->  **-3.334444815e-04** = -1/(V-1)
      scale 1e30   ->  **-3.334444815e-04**

    A negative mean squared overlap puts `superposition_ratio` BELOW the Welch
    bound -- the exact outcome this task's own source calls "provably
    impossible" when it reverted an epsilon on those grounds. The overflow
    route reproduced it anyway.

    Fixed by testing zero rows with max-abs (exact at every scale) and
    normalising with `safe_unit_rows_torch`. All nine scales now agree with the
    float64 reference to better than 5e-9 relative.
    """
    from blme.tasks.geometry.superposition_strength import _mean_squared_overlap

    rng = np.random.default_rng(0)
    base = rng.standard_normal((600, 32))
    base = base / np.abs(base).max()

    U = torch.as_tensor(base, dtype=torch.float64)
    U = U / U.norm(dim=1, keepdim=True)
    G = U @ U.t()
    n = U.shape[0]
    ref = float(((G ** 2).sum() - torch.diagonal(G).pow(2).sum()) / (n * (n - 1)))

    vals = []
    for s in SCALES:
        W = torch.as_tensor(_at_scale(base, np.float32(s)), dtype=torch.float32)
        v = _mean_squared_overlap(W)
        assert np.isfinite(v), f"scale {s:g}: not finite ({v})"
        assert v >= 0.0, f"scale {s:g}: NEGATIVE mean squared overlap {v!r}"
        vals.append(v)
    _const(vals, "mean_squared_overlap", rel=1e-6)
    assert abs(vals[0] - ref) / ref < 5e-8, (
        f"scale 1 disagrees with the float64 reference: {vals[0]!r} vs {ref!r}")


def test_mean_squared_overlap_does_not_cancel_on_an_orthonormal_matrix():
    """The V == m case, where the answer IS the cancelled digits.

    For an orthonormal matrix every off-diagonal overlap is 0, so the true MSO
    is 0 up to rounding. The old form computed `||U^T U||_F^2 - V`, i.e. it
    subtracted the INTEGER V from a float32 sum whose diagonal entries are only
    approximately 1 -- and the whole answer lives in that difference. On a
    256x256 orthonormal matrix it returned 9.35e-10 against an exact-float64
    2.68e-10 (+249%). Subtracting the actual sum of squared row norms fixes it.
    """
    from blme.tasks.geometry.superposition_strength import _mean_squared_overlap

    torch.manual_seed(0)
    Q = torch.linalg.qr(torch.randn(256, 256, dtype=torch.float64))[0]
    G = Q @ Q.t()
    ref = float(((G ** 2).sum() - torch.diagonal(G).pow(2).sum()) / (256 * 255))
    got = _mean_squared_overlap(Q.float())
    assert abs(got - ref) < 1e-14, f"{got!r} vs float64 {ref!r}"
    assert got >= 0.0


def test_cka_centering_is_correct_when_the_mean_dwarfs_the_spread():
    """CKA on off-centre data: the case two centering passes could not fix.

    `cka.py` centered TWICE in float32 on the theory that the second pass
    removes the residual mean the first leaves. Measured over
    mean in {0, 1e2, 1e4, 1e6} x sd in {1e-3, 1}, the second pass moves at most
    **1.825e-03** while the error against a float64 truth reaches **4.494e-02**
    -- and at mean 1e6 / sd 1e-3 BOTH forms returned **NaN** against a truth of
    0.400437, because float32's spacing at 1e6 is 0.0625 and every value in the
    column rounds to the same float. No number of repeated subtractions creates
    bits that are not there.

    Centering is now done in float64 when |mean|/sd is large (or when the
    float32 `std` has underflowed to 0, which is that same case seen from the
    other side). Real activations have |mean| comparable to sd and keep the
    cheap float32 path bit-for-bit, which this test also pins.
    """
    from blme.tasks.geometry.cka import _linear_cka

    def _center(Xf32, Xsrc):
        mu = Xf32.mean(0, keepdim=True)
        sd = Xf32.std(0, keepdim=True)
        ratio = float((mu.abs() / sd.clamp_min(torch.finfo(Xf32.dtype).tiny)).max())
        if (not np.isfinite(ratio)) or ratio > 1e4:
            Xd = Xsrc.double()
            return (Xd - Xd.mean(0, keepdim=True)).float(), True
        X = Xf32 - mu
        return X - X.mean(0, keepdim=True), False

    torch.manual_seed(1337)
    took_f64 = []
    for mu in (0.0, 1e2, 1e4, 1e6, 1e8):
        for sd in (1e-3, 1.0):
            X0 = torch.randn(200, 24, dtype=torch.float64)
            Y0 = torch.randn(200, 24, dtype=torch.float64) + 0.7 * X0
            X64, Y64 = X0 * sd + mu, Y0 * sd + mu
            truth = _linear_cka(X64 - X64.mean(0, keepdim=True),
                                Y64 - Y64.mean(0, keepdim=True), centered=True)
            a, f1 = _center(X64.float(), X64)
            b, f2 = _center(Y64.float(), Y64)
            got = _linear_cka(a, b, centered=True)
            assert np.isfinite(got), f"mean={mu:g} sd={sd:g}: NaN (truth {truth!r})"
            assert abs(got - truth) < 1e-5, (
                f"mean={mu:g} sd={sd:g}: {got!r} vs float64 {truth!r}")
            took_f64.append((mu, sd, f1 or f2))

    # The cheap path must still be the one real, centred-ish data takes.
    assert not dict(((m, s), f) for m, s, f in took_f64)[(0.0, 1.0)], (
        "well-conditioned data should not pay for float64 centering")
    assert dict(((m, s), f) for m, s, f in took_f64)[(1e6, 1e-3)], (
        "the underflow case must take the float64 path")


def test_superposition_task_outputs_are_all_scale_invariant_end_to_end():
    """The whole task, not just the kernel — because the kernel was fixed alone.

    A prior pass hardened `_mean_squared_overlap` and left three places in
    `evaluate()` that still squared before summing. Measured on an 800x32 head
    (independent Opus review, 2026-09-05, and re-measured here):

      * `V_eff` used `W.norm(dim=1) > 0` while the kernel used max-abs, so at a
        global scale of 1e-23 every row's norm underflowed, `V_eff` became 0,
        the Welch bound became 0, and `superposition_ratio` returned **NaN**
        while `mean_squared_overlap` was correct to 1.3e-9. That is precisely
        the "V_eff inconsistent" failure the kernel's own comment says it
        removed — removed from the kernel, left in the caller.
      * `row_norm_cv` and `frac_norm_above_half_max` materialised ABSOLUTE row
        norms, which overflow float32 at a global scale of 3.4e38 (a row norm is
        up to sqrt(m) times the largest entry), giving `nan` and `0.0` for two
        quantities that are scale-free BY DEFINITION.

    All four are now invariant across 1e-30 … 3.4e38.
    """
    import torch.nn as nn
    from blme.tasks.geometry.superposition_strength import (
        SuperpositionStrengthTask)

    rng = np.random.default_rng(0)
    base = rng.standard_normal((400, 24))
    base = base / np.abs(base).max()

    class _FakeLM(nn.Module):
        def __init__(self, W):
            super().__init__()
            self.lm_head = nn.Linear(W.shape[1], W.shape[0], bias=False)
            self.lm_head.weight.data = W

    keys = ("mean_squared_overlap", "superposition_ratio",
            "row_norm_cv", "frac_norm_above_half_max")
    got = {k: [] for k in keys}
    for s in SCALES:
        W = torch.as_tensor(_at_scale(base, np.float32(s)), dtype=torch.float32)
        out = SuperpositionStrengthTask(config={}).evaluate(
            _FakeLM(W), None, dataset=None)
        for k in keys:
            v = float(out[k])
            assert np.isfinite(v), f"{k} at scale {s:g}: {v!r}"
            got[k].append(v)

    _const(got["mean_squared_overlap"], "mean_squared_overlap", rel=1e-6)
    _const(got["superposition_ratio"], "superposition_ratio", rel=1e-6)
    _const(got["row_norm_cv"], "row_norm_cv", rel=1e-6)
    _const(got["frac_norm_above_half_max"], "frac_norm_above_half_max", rel=1e-9)


def test_safe_unit_rows_torch_uses_one_temporary_not_four():
    """The hardened normaliser must not cost 4x the input in RAM.

    `safe_unit_rows_torch` is applied to whole unembedding matrices. The first
    version allocated four full-size temporaries — `X/rmax`, the divided copy,
    the `zeros_like`, and the `torch.where` result — and an independent review
    measured the consequence on a real 2.18 GB head: **8.52 GB peak** (3.9x),
    while the float64 chunked Gram it feeds costs 0.00 GB extra. The normaliser
    was the entire memory cost of the "hardened" path, and the figure recorded
    for it at the time was wrong.

    Measured here in a fresh process on a 0.82 GB input: 3.99x before, 1.00x
    after. This test asserts the allocation SHAPE rather than a byte count,
    which is what actually has to hold: exactly one full-size tensor is created,
    and it is the one returned.
    """
    import gc

    torch.manual_seed(0)
    X = torch.randn(4000, 512)
    n_bytes = X.numel() * X.element_size()

    live_before = sum(
        o.numel() * o.element_size()
        for o in gc.get_objects()
        if isinstance(o, torch.Tensor) and o.numel() >= X.numel())

    U = safe_unit_rows_torch_ref = None  # noqa: F841
    from blme.tasks.geometry.utils import safe_unit_rows_torch
    U = safe_unit_rows_torch(X)

    live_after = sum(
        o.numel() * o.element_size()
        for o in gc.get_objects()
        if isinstance(o, torch.Tensor) and o.numel() >= X.numel())

    # X and U only; a third full-size tensor means a temporary was retained.
    assert live_after - live_before <= n_bytes * 1.05, (
        f"more than one full-size tensor survives the call: "
        f"{(live_after - live_before) / n_bytes:.2f}x the input")

    # And it must still be exactly the old arithmetic.
    rmax = X.abs().amax(dim=-1, keepdim=True)
    tiny = torch.finfo(X.dtype).tiny
    Xs = X / rmax.clamp_min(tiny)
    ref = Xs / Xs.norm(dim=-1, keepdim=True).clamp_min(tiny)
    ref = torch.where(rmax > 0, ref, torch.zeros_like(ref))
    assert torch.equal(U, ref), (
        f"value changed: max |diff| {float((U - ref).abs().max()):.3e}")
