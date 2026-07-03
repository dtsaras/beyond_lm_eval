"""Numeric-parity test: BLME WEAT helpers vs. the official sent-bias reference.

Task: consistency_bias_weat (WEAT effect size — Caliskan et al. 2017 Science;
SEAT extension May et al. 2019, arXiv:1903.10561).

BLME helpers under test (src/blme/tasks/consistency/bias.py):
    - _weat_effect_size(X, Y, A, B)   (line ~142)  -> Cohen's d effect size
    - _weat_statistic(X, Y, A, B)     (line ~133)  -> WEAT test statistic s(X,Y,A,B)

Official reference: W4ngatang/sent-bias
    repo:   https://github.com/W4ngatang/sent-bias
    commit: e3559fb669ca4832743b42fee715994c15c7f1af
    file:   sentbias/weat.py
        effect_size(X, Y, A, B, cossims)            line 178
            num = mean_X s(x,A,B) - mean_Y s(y,A,B)
            den = stdev_{X u Y} s(w,A,B), ddof=1
            s(w,A,B) = mean_a cos(w,a) - mean_b cos(w,b)
            cossim(x,y) = dot(x,y)/sqrt(dot(x,x)*dot(y,y))
        s_XYAB / s_XAB / s_wAB (lines 32-79)        WEAT statistic
            s(X,Y,A,B) = sum_X s(x,A,B) - sum_Y s(y,A,B)

The expected OFFICIAL values below were produced by running that exact
reference code on the toy input regenerated here from a fixed seed
(see scratchpad/wave1/bias_weat_verify.py, abs_diff == 0.0). The
permutation p-value is NOT covered: it is stochastic in BLME and the
official non-parametric path relies on the numpy<1.20 `np.int` alias
(removed in numpy 2.x). Effect size and the test statistic are
deterministic closed forms, so they are pinned exactly.
"""
import numpy as np
import pytest

from blme.tasks.consistency.bias import _weat_effect_size, _weat_statistic

# OFFICIAL reference outputs (sent-bias @ e3559fb), toy seed=12345, dim=5, n=4.
OFFICIAL_EFFECT_SIZE = -0.5483337931868121
OFFICIAL_STATISTIC = -0.7437862779422525
TOL = 1e-9


def _build_toy(seed=12345, dim=5, n=4):
    rng = np.random.default_rng(seed)
    X = [rng.standard_normal(dim) for _ in range(n)]
    Y = [rng.standard_normal(dim) for _ in range(n)]
    A = [rng.standard_normal(dim) for _ in range(n)]
    B = [rng.standard_normal(dim) for _ in range(n)]
    return X, Y, A, B


def test_weat_effect_size_matches_sentbias():
    X, Y, A, B = _build_toy()
    blme_d = _weat_effect_size(X, Y, A, B)
    assert blme_d == pytest.approx(OFFICIAL_EFFECT_SIZE, abs=TOL)


def test_weat_statistic_matches_sentbias():
    X, Y, A, B = _build_toy()
    blme_stat = _weat_statistic(X, Y, A, B)
    assert blme_stat == pytest.approx(OFFICIAL_STATISTIC, abs=TOL)
