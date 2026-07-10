"""Campaign-4 B8: parity/analytic tests for the gap-survey metrics.

New tasks: geometry_superposition_strength (Liu-Liu-Gore 2505.10465),
dynamics_fractal (Alabdulmohsin 2402.01825), geometry_effective_depth
(ShortGPT 2403.03853 / Csordas 2505.13898), dynamics_perturbation_growth
(QLE grounding 2503.13530).
Extensions: spectral official PL-MLE alpha (vs LIVE weightwatcher),
MP lower edge (Staats 2410.17770), critical sharpness (Kalra 2601.16979),
scale-stratified PH (Fay 2505.20435).

Every quantity is pinned against an analytic anchor, a live official
package (weightwatcher, nolds), or an independent recomputation.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn


# ===========================================================================
# geometry_superposition_strength
# ===========================================================================
class TestSuperpositionStrength:
    def test_orthonormal_rows_zero_overlap(self):
        from blme.tasks.geometry.superposition_strength import (
            _mean_squared_overlap, _welch_bound)
        W = torch.eye(16)
        assert _mean_squared_overlap(W) == pytest.approx(0.0, abs=1e-12)
        assert _welch_bound(16, 16) == 0.0

    def test_simplex_etf_achieves_welch_bound_exactly(self):
        """V = m+1 unit vectors as a simplex ETF: all pairwise cos = -1/m,
        so mean squared overlap = 1/m^2 == Welch bound (V-m)/(m(V-1))."""
        from blme.tasks.geometry.superposition_strength import (
            _mean_squared_overlap, _welch_bound)
        m = 12
        V = m + 1
        # Construct the simplex: project the V standard basis vectors of
        # R^V onto the hyperplane orthogonal to (1,...,1).
        E = np.eye(V) - np.full((V, V), 1.0 / V)
        Q, _ = np.linalg.qr(E.T)           # basis of the m-dim hyperplane
        S = E @ Q[:, :m]                   # (V, m) simplex vertices
        S = S / np.linalg.norm(S, axis=1, keepdims=True)
        mso = _mean_squared_overlap(torch.tensor(S, dtype=torch.float64))
        welch = _welch_bound(V, m)
        assert welch == pytest.approx(1.0 / m ** 2, rel=1e-12)
        assert mso == pytest.approx(welch, rel=1e-6)

    def test_random_gaussian_rows_give_one_over_m(self):
        """iid Gaussian rows: E[cos^2] = 1/m, so overlap_vs_random ~ 1."""
        from blme.tasks.geometry.superposition_strength import (
            _mean_squared_overlap)
        torch.manual_seed(0)
        m, V = 64, 600
        W = torch.randn(V, m)
        mso = _mean_squared_overlap(W)
        assert mso * m == pytest.approx(1.0, rel=0.1)

    def test_sampled_path_consistent_with_exact(self):
        from blme.tasks.geometry.superposition_strength import (
            _mean_squared_overlap)
        torch.manual_seed(1)
        W = torch.randn(300, 24)
        exact = _mean_squared_overlap(W, max_pairs=10 ** 9)
        sampled = _mean_squared_overlap(W, max_pairs=20_000, seed=0)
        assert sampled == pytest.approx(exact, rel=0.05)

    def test_task_headline_is_mean_centered(self):
        """Task-level protocol pin: LLM unembeddings carry a shared mean
        direction (anisotropy cone) that inflates raw overlaps; Liu et
        al.'s ~1/m Welch scaling only reproduces after mean-centering
        (gpt2: raw 0.075 vs centered 0.0025 ~ 1.9/m). Construct W =
        shared mean + iid Gaussian: the headline (centered) MSO must be
        ~1/m while the raw variant stays an order of magnitude larger."""
        from blme.tasks.geometry.superposition_strength import (
            SuperpositionStrengthTask)

        m, V = 64, 600
        torch.manual_seed(3)
        mean_dir = torch.randn(1, m) * 5.0   # dominant shared component
        W = mean_dir + torch.randn(V, m)

        class FakeLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(m, V, bias=False)
                self.lm_head.weight.data = W

        out = SuperpositionStrengthTask(config={}).evaluate(
            FakeLM(), None, dataset=None)
        assert out["mean_squared_overlap"] * m == pytest.approx(1.0, rel=0.15)
        assert out["mean_squared_overlap_raw"] > 10 * out["mean_squared_overlap"]
        assert out["vocab_size"] == V and out["hidden_dim"] == m


# ===========================================================================
# dynamics_fractal (Hurst / Joseph / Holder)
# ===========================================================================
class TestFractal:
    def test_white_noise_anchors(self):
        """iid Gaussian increments: H ~ 0.5 (R/S), J = 0.5 (sqrt diffusion)."""
        from blme.tasks.dynamics.fractal import _fractal_params
        rng = np.random.default_rng(0)
        z = rng.standard_normal(20_000)
        out = _fractal_params(z)
        assert out["hurst_exponent"] == pytest.approx(0.5, abs=0.07)
        assert out["joseph_exponent"] == pytest.approx(0.5, abs=0.05)

    def test_persistent_series_high_hurst(self):
        """Strongly trending (integrated) increments -> H near 1."""
        from blme.tasks.dynamics.fractal import _rs_hurst
        t = np.linspace(0, 20 * np.pi, 20_000)
        x = np.sin(t / 200.0) + 0.01 * np.random.default_rng(1).standard_normal(t.size)
        x = (x - x.mean()) / x.std()
        assert _rs_hurst(x) > 0.85

    def test_hurst_tracks_nolds_reference(self):
        """Estimator-family agreement with the official nolds.hurst_rs on
        the SAME series (windowing details differ; band 0.1)."""
        nolds = pytest.importorskip("nolds")
        from blme.tasks.dynamics.fractal import _rs_hurst
        rng = np.random.default_rng(2)
        x = rng.standard_normal(8_192)
        blme = _rs_hurst(x)
        ref = float(nolds.hurst_rs(x))
        assert blme == pytest.approx(ref, abs=0.1)

    def test_deterministic(self):
        from blme.tasks.dynamics.fractal import _fractal_params
        rng = np.random.default_rng(3)
        z = rng.standard_normal(4_096)
        a = _fractal_params(z)
        b = _fractal_params(z)
        assert a == b


# ===========================================================================
# geometry_effective_depth
# ===========================================================================
class TestEffectiveDepth:
    def _states(self, transforms, T=50, D=32, seed=0):
        g = torch.Generator().manual_seed(seed)
        h = torch.randn(T, D, generator=g)
        states = [h]
        for f in transforms:
            h = f(h)
            states.append(h)
        return states

    def test_identity_layers_zero_influence(self):
        from blme.tasks.geometry.effective_depth import _depth_profile
        out = _depth_profile(self._states([lambda h: h, lambda h: h]))
        assert out["mean_block_influence"] == pytest.approx(0.0, abs=1e-6)
        assert out["frac_low_influence"] == 1.0

    def test_sign_flip_layer_bi_two(self):
        from blme.tasks.geometry.effective_depth import _depth_profile
        out = _depth_profile(self._states([lambda h: -h]))
        assert out["block_influence_per_layer"][0] == pytest.approx(2.0, abs=1e-6)

    def test_orthogonal_rotation_bi_near_one(self):
        from blme.tasks.geometry.effective_depth import _depth_profile
        D = 64
        Q, _ = torch.linalg.qr(torch.randn(D, D, generator=torch.Generator().manual_seed(1)))
        out = _depth_profile(self._states([lambda h: h @ Q], D=D))
        assert out["block_influence_per_layer"][0] == pytest.approx(1.0, abs=0.15)

    def test_effective_depth_concentrated_first_layer(self):
        """All contribution in layer 1 (later layers identity) ->
        effective_depth_90 = 1/L."""
        from blme.tasks.geometry.effective_depth import _depth_profile
        out = _depth_profile(self._states(
            [lambda h: h * 3.0, lambda h: h, lambda h: h, lambda h: h]))
        assert out["effective_depth_90"] == pytest.approx(0.25, abs=1e-9)


# ===========================================================================
# dynamics_perturbation_growth
# ===========================================================================
class TestPerturbationGrowth:
    def test_linear_gain_stack_exact_slope(self):
        """h_{l+1} = a*h_l propagates a perturbation delta by |a| per layer:
        growth_log_ratio = L * ln(a) EXACTLY, slope = L*ln(a) over [0,1]."""
        from blme.tasks.dynamics.perturbation_growth import _growth_profile
        g = torch.Generator().manual_seed(0)
        v = torch.randn(40, 16, generator=g, dtype=torch.float64)
        delta = 1e-3 * torch.randn(40, 16, generator=g, dtype=torch.float64)
        a, L = 1.7, 5
        clean = [v * (a ** l) for l in range(L + 1)]
        pert = [(v + delta) * (a ** l) for l in range(L + 1)]
        out = _growth_profile(clean, pert)
        # (helper computes in float32 -> rel ~1e-6 precision floor)
        assert out["growth_log_ratio"] == pytest.approx(L * np.log(a), rel=1e-5)
        assert out["growth_slope"] == pytest.approx(L * np.log(a), rel=1e-5)

    def test_contracting_stack_negative_slope(self):
        from blme.tasks.dynamics.perturbation_growth import _growth_profile
        g = torch.Generator().manual_seed(1)
        v = torch.randn(30, 8, generator=g, dtype=torch.float64)
        d = 1e-4 * torch.randn(30, 8, generator=g, dtype=torch.float64)
        clean = [v * (0.5 ** l) for l in range(4)]
        pert = [(v + d) * (0.5 ** l) for l in range(4)]
        out = _growth_profile(clean, pert)
        assert out["growth_slope"] < 0


# ===========================================================================
# spectral: official PL-MLE alpha vs LIVE weightwatcher
# ===========================================================================
class TestOfficialAlpha:
    def test_planted_pareto_recovered(self):
        from blme.tasks.geometry.spectral import _pl_mle_alpha_ks
        rng = np.random.default_rng(0)
        lam = (1.0 - rng.random(5000)) ** (-1.0 / 2.0)   # Pareto, alpha = 3
        a, xmin, ks, ntail = _pl_mle_alpha_ks(lam)
        assert a == pytest.approx(3.0, abs=0.1)
        assert ntail > 1000

    def test_matches_live_weightwatcher(self):
        """BLME's KS-optimal PL-MLE on eigenvalues == the OFFICIAL
        weightwatcher alpha on the same planted-spectrum Linear layer
        (measured agreement ~1e-5)."""
        ww = pytest.importorskip("weightwatcher")
        import logging
        logging.disable(logging.CRITICAL)
        try:
            rng = np.random.default_rng(0)
            n = 300
            q = (1.0 - (np.arange(1, n + 1) - 0.5) / n) ** (-1.0 / 2.5)
            s = np.sort(q)[::-1]
            U, _ = np.linalg.qr(rng.standard_normal((n, n)))
            Vt, _ = np.linalg.qr(rng.standard_normal((n, n)))
            W = (U * s) @ Vt.T
            lin = nn.Linear(n, n, bias=False)
            with torch.no_grad():
                lin.weight.copy_(torch.tensor(W, dtype=torch.float32))

            details = ww.WeightWatcher(model=nn.Sequential(lin)).analyze(
                layers=[], plot=False)
            ref_alpha = float(details["alpha"].iloc[0])

            from blme.tasks.geometry.spectral import _pl_mle_alpha_ks
            evals = np.sort(np.linalg.svd(
                lin.weight.detach().numpy().astype(np.float64),
                compute_uv=False)) ** 2
            blme_alpha, _, _, _ = _pl_mle_alpha_ks(evals)
            assert blme_alpha == pytest.approx(ref_alpha, abs=5e-3)
        finally:
            logging.disable(logging.NOTSET)


# ===========================================================================
# MP lower edge
# ===========================================================================
class TestMPLowerEdge:
    def test_iid_no_lower_outliers(self):
        from blme.tasks.geometry.rmt_bulk import _mp_metrics_core
        rng = np.random.default_rng(0)
        core = _mp_metrics_core(rng.standard_normal((1600, 400)), 0.05)
        assert core["lower_outlier_frac"] == pytest.approx(0.0, abs=0.01)

    def test_correlated_dims_produce_lower_outliers(self):
        """Near-duplicate dimensions push eigenvalues BELOW the bulk (the
        Staats et al. small-singular-value signal)."""
        from blme.tasks.geometry.rmt_bulk import _mp_metrics_core
        rng = np.random.default_rng(0)
        B = rng.standard_normal((1600, 380))
        M = np.concatenate(
            [B, B[:, :20] * 0.9 + 0.1 * rng.standard_normal((1600, 20))],
            axis=1)
        core = _mp_metrics_core(M, 0.05)
        assert core["lower_outlier_frac"] >= 0.04   # ~20/400 planted

    def test_gamma_ge_one_is_nan(self):
        from blme.tasks.geometry.rmt_bulk import _mp_metrics_core
        rng = np.random.default_rng(1)
        core = _mp_metrics_core(rng.standard_normal((100, 300)), 0.05)
        assert np.isnan(core["lower_outlier_frac"])


# ===========================================================================
# critical sharpness
# ===========================================================================
class TestCriticalSharpness:
    @pytest.mark.parametrize("lam", [0.25, 4.0, 37.5])
    def test_quadratic_recovers_curvature_exactly(self, lam):
        """L = 0.5*lam*||theta||^2: raw-GD step is stable iff eta < 2/lam,
        so lambda_c = 2/eta* == lam analytically."""
        from blme.tasks.dynamics.sharpness import _critical_step_size
        theta = nn.Parameter(torch.tensor([3.0, -4.0], dtype=torch.float64))

        def loss():
            return 0.5 * lam * (theta ** 2).sum()

        base = float(loss().item())
        g = torch.autograd.grad(loss(), [theta])
        gn = float(g[0].norm().item())
        before = theta.detach().clone()
        eta, lc = _critical_step_size(loss, [theta], list(g), gn, base,
                                      bisect_iters=40)
        assert lc == pytest.approx(lam, rel=1e-4)
        assert eta == pytest.approx(2.0 / lam, rel=1e-4)
        assert torch.equal(theta.detach(), before), "params not restored"

    def test_zero_gradient_nan(self):
        from blme.tasks.dynamics.sharpness import _critical_step_size
        theta = nn.Parameter(torch.zeros(2, dtype=torch.float64))

        def loss():
            return (theta ** 2).sum()

        eta, lc = _critical_step_size(loss, [theta], [torch.zeros(2)], 0.0, 0.0)
        assert np.isnan(eta) and np.isnan(lc)


# ===========================================================================
# scale-stratified PH summaries
# ===========================================================================
class TestScaleStratifiedPH:
    def test_constructed_diagram_anchors(self):
        from blme.tasks.topology.homology import _scale_stratified_summary
        dgms = [np.array([[0.0, 0.2], [0.0, 0.9]]),
                np.array([[0.05, 0.1], [0.5, 2.0]])]
        out = _scale_stratified_summary(dgms, scale=1.0, frac=0.5)
        assert out["h0_fine_count"] == 1 and out["h0_coarse_count"] == 1
        assert out["h1_fine_count"] == 1 and out["h1_coarse_count"] == 1
        assert out["h1_fine_persistence_frac"] == pytest.approx(
            0.05 / (0.05 + 1.5), abs=1e-12)

    def test_degenerate_scale_empty(self):
        from blme.tasks.topology.homology import _scale_stratified_summary
        assert _scale_stratified_summary([np.zeros((0, 2))], scale=0.0) == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
