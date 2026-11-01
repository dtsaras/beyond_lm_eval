"""``geometry_ww_randomized``: WeightWatcher's self-randomized and
Marchenko-Pastur-fit metrics of every weight matrix.

Papers: Prakash & Martin 2025, "Grokking and Generalization Collapse: Insights
from HTSR theory" (arXiv:2506.04434; outliers of the randomized spectrum,
correlation traps); Martin & Hinrichs 2025, "SETOL: A Semi-Empirical Theory of
(Deep) Learning" (arXiv:2507.17912; the ERG condition det X = 1); Staats, Thamm
& Rosenow 2024, "Small Singular Values Matter" (arXiv:2410.17770; departures
from random-matrix theory at the small end of the spectrum); Martin & Mahoney
2021 (JMLR, arXiv:1810.01075; the MP soft rank).

Code: CalculatedContent/WeightWatcher@a5940f7 (``weightwatcher.py``:
``combined_eigenvalues``, ``random_eigenvalues``, ``apply_random_esd``,
``mp_fit``, ``fit_powerlaw``, ``apply_detX``; ``RMT_Util.py``:
``jensen_shannon_distance``, ``fit_density_with_range``, ``resid_mp``,
``marchenko_pastur_fun``, ``quarter_circle_fun``, ``calc_sigma``,
``rescale_eigenvalues``, ``detX_constraint``; ``WW_powerlaw.WWFit``), as
``WeightWatcher.analyze(randomize=True, mp_fit=True, detX=True)`` runs them.

For a weight matrix W (N x M, N >= M, Q = N / M) with eigenvalues lambda of
W^T W (squared float64 singular values, sorted ascending, M of them):

* ``rand_distance``: the entries of W are shuffled (``W.flatten()``, shuffle,
  reshape) and ``jensen_shannon_distance(lambda, lambda_rand)`` is taken. It
  applies ``scipy.stats.entropy`` to the two sorted eigenvalue VECTORS, each
  normalised to sum 1 over the index i, not to histograms. A shuffle keeps
  ||W||_F, so both vectors have the same sum. ``ww_softrank`` = max lambda_rand
  / max lambda;
* ``mp_fit``: lambda rescaled to mean 1; a linear-kernel KDE (bandwidth 0.1)
  on 1000 points of [0, max + 0.5], points of density below 0.01 dropped; the
  MP density of aspect ratio Q (the quarter-circle law of the singular values
  when Q = 1) is fitted for sigma by ``scipy.optimize.brute`` over
  [0.05, 2) x sigma_0 in steps of 0.01 sigma_0 (sigma_0 from lambda_max), then
  Nelder-Mead. Bulk edges lambda_+- = (sigma (1 +- Q^-1/2))^2; ``num_spikes`` =
  #{lambda > lambda_+ + sqrt(TW)}, TW = Q^-1/2 lambda_+^(2/3) M^(-2/3);
  ``mp_softrank`` = lambda_+ / lambda_max;
* ``apply_detX``: lambda rescaled to mean 1; scanning suffixes from the top,
  ``detX_num`` is the length of the first suffix whose product is below 1;
  ``ERG_gap`` = detX_num - num_pl_spikes, where num_pl_spikes counts lambda >=
  the xmin of WWFit (the KS-optimal power-law fit of lambda > 1e-5);
* the Staats et al. lower edge (BLME): the fraction of lambda below the fitted
  lambda_-, NaN when Q = 1 (lambda_- = 0).

Every step runs the released expressions on the same float64 eigenvalues, so
the values equal the release's to the bit; the KDE, which does not depend on
sigma, is computed once per fit instead of once per sigma.
tests/tasks/parity/test_ww_randomized_parity.py compares each function and
whole ``analyze()`` runs with the clone. The release's own path is followed
without CUDA (with CUDA it reads float16 weights and runs float32 torch SVDs).

The release's ``np.prod`` over an ascending suffix multiplies the smallest
eigenvalues first; once the log of the largest suffix product passes ~745 the
partial product underflows to 0 and the scan stops early. Every 4096 x 4096
q/o projection of Qwen3-8B-Base tested is affected (detX_num off by up to 84),
no gpt2 matrix is; ``detX_num`` and ``ERG_gap`` stay the release's and
``detX_num_logsum`` / ``ERG_gap_logsum`` take each product as a sum of logs.
"""
import logging

import numpy as np
import scipy.linalg
from scipy import optimize, stats
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from .spectral import _HFConv1D, _language_model_modules

logger = logging.getLogger("blme")

# constants.py: DEFAULT_MIN_EVALS, DEFAULT_MAX_EVALS, DEFAULT_MAX_N (layer_supported
# skips M < 8, M > 15000 and N > 50000), EVALS_THRESH (the PL fit's floor).
MIN_EVALS = 8
MAX_EVALS = 15000
MAX_N = 50000
EVALS_THRESH = 1e-5
# apply_random_esd: 5 randomized copies when n_comp < 100.
SMALL_N_COMP = 100
SMALL_N_REPLICAS = 5
# mp_fit: bw = 0.1; resid_mp: THRESH = 0.01, a 1000-point grid.
KDE_BANDWIDTH = 0.1
KDE_DENSITY_FLOOR = 0.01
KDE_GRID = 1000
# WW_powerlaw: SMALL_N_CUTOFF = 20; fit_powerlaw_smallN(k_min=8).
SMALL_N_CUTOFF = 20
SMALL_N_KMIN = 8
DEFAULT_SHUFFLE_SEED = 42


def _esd(W: np.ndarray) -> np.ndarray:
    """``combined_eigenvalues`` of one dense matrix: squared singular values of
    the float64 matrix (scipy gesdd, the release's CPU SVD), sorted ascending."""
    sv = scipy.linalg.svd(W, compute_uv=False)
    sv = np.sort(sv.flatten())[-min(W.shape):]
    return np.sort(sv * sv)


def jensen_shannon_distance(p, q):
    """``RMT_Util.jensen_shannon_distance``: entropy of the vectors themselves."""
    p = np.array(p)
    q = np.array(q)
    m = (p + q) / 2
    divergence = (stats.entropy(p, m) + stats.entropy(q, m)) / 2
    return np.sqrt(divergence)


def _calc_sigma(Q, evals):
    lmax = np.max(evals)
    inv_sqQ = 1.0 / np.sqrt(Q)
    sigma_2 = lmax / np.square(1 + inv_sqQ)
    return np.sqrt(sigma_2)


def _marchenko_pastur_fun(x, Q, sigma=1):
    y = 1 / Q
    b = np.power(sigma * (1 + np.sqrt(1 / Q)), 2)
    a = np.power(sigma * (1 - np.sqrt(1 / Q)), 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.nan_to_num((1 / (2 * np.pi * sigma * sigma * x * y)) * np.sqrt((b - x) * (x - a)))
    return x, val, a, b


def _quarter_circle_fun(x, sigma=1):
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.nan_to_num((1 / (np.pi * sigma * sigma)) * np.sqrt(4 - x ** 2))
    return x, val


def _kde_profile(to_fit, bw=KDE_BANDWIDTH):
    """The sigma-independent half of ``resid_mp``: the KDE on its grid, the
    points above the density floor, and the number below it."""
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(kernel="linear", bandwidth=bw).fit(to_fit.reshape(-1, 1))
    xde = np.linspace(0, np.max(to_fit) + 0.5, KDE_GRID)
    yde = np.exp(kde.score_samples(xde[:, np.newaxis]))
    bad_ids = np.where(np.array(yde) < KDE_DENSITY_FLOOR)[0]
    good_ids = np.where(np.array(yde) > KDE_DENSITY_FLOOR)[0]
    xde = np.array(xde)[good_ids]
    yde = np.array(yde)[good_ids]
    x_peak = xde[np.where(yde == max(yde))][0]
    return xde, yde, x_peak, max(xde), len(bad_ids)


def _resid_mp(p, xde, yde, x_peak, x_top, Q, n_bad):
    """``resid_mp(..., allresid=False)`` on a precomputed KDE profile."""
    sigma = p
    if Q == 1:
        xmp, ymp = _quarter_circle_fun(xde, sigma=sigma)
        resid = ymp - yde
    else:
        xmp, ymp, a, b = _marchenko_pastur_fun(xde, Q=Q, sigma=sigma)
        if (b < x_peak) | (b > x_top) | (a > x_peak):
            resid = np.zeros(len(yde)) + 1000
        else:
            resid = ymp - yde
    # The release appends a zero per dropped point; pairwise summation sees them.
    resid = np.concatenate((resid, np.zeros(n_bad)))
    return np.sum(resid ** 2)


def _fit_density_with_range(evals, Q, bw=KDE_BANDWIDTH):
    """``fit_density_with_range``: (sigma, residual)."""
    sigma_scaling_factor = _calc_sigma(Q, evals)
    sigma_range = (slice(0.05 * sigma_scaling_factor, 2.0 * sigma_scaling_factor,
                         0.01 * sigma_scaling_factor),)
    to_fit = np.sqrt(evals) if Q == 1 else evals
    xde, yde, x_peak, x_top, n_bad = _kde_profile(to_fit, bw)
    brute_output = optimize.brute(_resid_mp, sigma_range,
                                  args=(xde, yde, x_peak, x_top, Q, n_bad), full_output=True)
    return brute_output[0][0], brute_output[1]


def mp_fit(evals, N, M):
    """``WeightWatcher.mp_fit`` with rescale=True (no plots):
    ``{num_spikes, sigma_mp, mp_softrank, bulk_min, bulk_max, W_scale}``;
    bulk_min / bulk_max in the units of ``evals``."""
    Q = N / M
    to_plot = evals.copy()
    Wnorm = np.sqrt(np.sum(evals))
    Wscale = np.sqrt(to_plot.shape[0]) / Wnorm
    to_plot = (Wscale * Wscale) * to_plot
    lambda_max = np.max(to_plot)
    s1, _ = _fit_density_with_range(to_plot, Q, bw=KDE_BANDWIDTH)
    bulk_max = (s1 * (1 + 1 / np.sqrt(Q))) ** 2
    bulk_min = (s1 * (1 - 1 / np.sqrt(Q))) ** 2
    TW = 1 / np.sqrt(Q) * np.power(bulk_max, 2 / 3) * np.power(M, -2 / 3)
    bulk_max_TW = bulk_max + np.sqrt(TW)
    num_spikes = len(to_plot[to_plot > bulk_max_TW])
    mp_softrank = bulk_max / lambda_max
    return {"num_spikes": int(num_spikes), "sigma_mp": float(s1), "mp_softrank": float(mp_softrank),
            "bulk_min": float(bulk_min / (Wscale * Wscale)),
            "bulk_max": float(bulk_max / (Wscale * Wscale)), "W_scale": float(Wscale)}


def wwfit(data):
    """``WW_powerlaw.WWFit(data)`` (power law, no xmin/xmax): ``(xmin, alpha, D)``,
    or None where the constructor raises (fewer than 2 points)."""
    data = np.asarray(np.sort(data), dtype=np.float64)
    N = len(data)
    if N < 2:
        return None
    xmins = data[:-1]
    log_data = np.log(data, dtype=np.float64)
    alphas = np.zeros(N - 1, dtype=np.float64)
    Ds = np.ones(N - 1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if N < SMALL_N_CUTOFF:
            # fit_powerlaw_smallN; its J-objective winner is discarded by the
            # constructor's argmin(Ds), so only alphas and Ds matter.
            any_j = False
            for i, xmin in enumerate(data[:-1]):
                n_int = N - i
                if n_int < SMALL_N_KMIN:
                    continue
                n = float(n_int)
                s = np.sum(log_data[i:]) - n * log_data[i]
                if s <= 1e-12:
                    continue
                alpha_bc = 1.0 + (n - 1.0) / s
                alphas[i] = alpha_bc
                if alpha_bc <= 1.0:
                    continue
                F_fit = 1.0 - (data[i:] / xmin) ** (1.0 - alpha_bc)
                F_emp = np.arange(n_int, dtype=np.float64) / n
                Ds[i] = float(np.max(np.abs(F_emp - F_fit)))
                any_j = True
            if not any_j:
                n = float(N)
                s = np.sum(log_data) - n * log_data[0]
                if s > 1e-12:
                    alpha_bc = 1.0 + (n - 1.0) / s
                    alphas[0] = alpha_bc
                    F_fit = 1.0 - (data / data[0]) ** (1.0 - alpha_bc)
                    F_emp = np.arange(N, dtype=np.float64) / n
                    Ds[0] = float(np.max(np.abs(F_emp - F_fit)))
        else:
            for i, xmin in enumerate(data[:-1]):
                n = float(N - i)
                alpha = 1 + n / (np.sum(log_data[i:]) - n * log_data[i])
                alphas[i] = alpha
                if alpha > 1:
                    Ds[i] = np.max(np.abs(
                        1 - (data[i:] / xmin) ** (-alpha + 1) - np.arange(n) / n))
    i = np.argmin(Ds)
    return float(xmins[i]), float(alphas[i]), float(Ds[i])


def num_pl_spikes(evals):
    """``fit_powerlaw``'s num_pl_spikes: #{lambda >= xmin} for the WWFit of
    lambda > 1e-5; None where the release's fit FAILED (it then reports -1)."""
    fit = wwfit(evals[evals > EVALS_THRESH])
    if fit is None or np.isnan(fit[1]):
        return None
    return int(len(evals[evals >= fit[0]]))


def _rescale_eigenvalues(evals):
    """``RMT_Util.rescale_eigenvalues``: eigenvalues scaled to mean 1."""
    N = len(evals)
    Wnorm = np.sqrt(np.sum(evals))
    Wscale = np.sqrt(N) / Wnorm
    return (Wscale * Wscale) * evals


def detx_constraint(evals):
    """``apply_detX``: ``rescale_eigenvalues`` then ``detX_constraint(rescale=False)``:
    ``(detX_num, detX_idx)``."""
    evals = _rescale_eigenvalues(evals)
    if len(evals) < 2 or not (evals[0] < evals[-1]):
        return 0, 0
    num_evals = len(evals)
    idx = 0
    with np.errstate(over="ignore", under="ignore"):
        for idx in range(len(evals) - 1, 0, -1):
            detX = np.prod(evals[idx:])
            if detX < 1.0:
                num_evals = len(evals) - idx
                break
    return int(num_evals), int(idx)


def detx_num_logsum(evals):
    """detX_num of the same scan with each suffix's product taken as a sum of
    logs (BLME). ``np.prod`` multiplies an ascending suffix smallest first, so
    once the log of the largest suffix product passes ~745 the partial products
    underflow to 0 and the release's scan stops early."""
    evals = _rescale_eigenvalues(evals)
    if len(evals) < 2 or not (evals[0] < evals[-1]):
        return 0
    with np.errstate(divide="ignore"):
        suffix_logs = np.cumsum(np.log(evals[::-1]))[:-1]
    below = np.nonzero(suffix_logs < 0)[0]
    return int(below[0] + 1) if below.size else int(len(evals))


def _ww_matrices(model):
    """``(names, weights, counts)``: the language model's Linear and HF
    Conv1D weights (``geometry_spectral``'s population) that WeightWatcher's
    ``layer_supported`` admits, as stored (a Conv1D weight is (in, out), as the
    release reads it). torch convolution kernels are skipped and counted."""
    try:
        out_emb = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    except Exception:
        out_emb = None
    lm_ids = _language_model_modules(model, out_emb)
    targets = (torch.nn.Linear,) + ((_HFConv1D,) if _HFConv1D is not None else ())
    convs = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
    counts = {"outside_lm": 0, "conv_kernels_skipped": 0, "below_min_evals": 0,
              "above_max_evals": 0, "above_max_n": 0, "head_tagged": 0}
    names, weights = [], []
    for name, module in model.named_modules():
        if not isinstance(module, targets + convs):
            continue
        w = module._parameters.get("weight")
        if w is None:
            continue
        is_head = (module is out_emb) or name.rsplit(".", 1)[-1] in ("lm_head", "embed_out",
                                                                       "output_projection")
        if lm_ids is not None and id(module) not in lm_ids and not is_head:
            counts["outside_lm"] += 1
            continue
        if isinstance(module, convs) or w.dim() != 2:
            counts["conv_kernels_skipped"] += 1
            continue
        N, M = max(w.shape), min(w.shape)
        if M < MIN_EVALS:
            counts["below_min_evals"] += 1
            continue
        if M > MAX_EVALS:
            counts["above_max_evals"] += 1
            continue
        if N > MAX_N:
            counts["above_max_n"] += 1
            continue
        counts["head_tagged"] += int(is_head)
        names.append(name)
        weights.append(w)
    return names, weights, counts


def ww_matrix_metrics(W32: np.ndarray, rng: np.random.RandomState) -> dict:
    """Every per-matrix quantity for one weight matrix, drawing its shuffles from
    ``rng`` in the release's order: ``apply_random_esd``'s first call
    (``SMALL_N_REPLICAS`` copies when M < 100), whose spectra feed only the
    release's plots, ``max_rand_eval`` and randomized MP fit, is drawn and not
    decomposed; its second, single-copy call gives rand_distance and
    ww_softrank."""
    N, M = max(W32.shape), min(W32.shape)
    W = W32.astype(float)
    evals = _esd(W)
    out = {"n_evals": int(evals.size), "Q": float(N / M)}
    replicas = SMALL_N_REPLICAS if M < SMALL_N_COMP else 1
    for _ in range(replicas):
        rng.shuffle(np.empty(W.size, dtype=np.uint8))
    Wrand = W.flatten()
    rng.shuffle(Wrand)
    rand_evals = _esd(Wrand.reshape(W.shape))
    if not np.max(evals) > 0:
        out["reason"] = "all-zero matrix"
        return out
    out["rand_distance"] = float(jensen_shannon_distance(evals, rand_evals))
    out["ww_softrank"] = float(np.max(rand_evals) / np.max(evals))
    try:
        out.update(mp_fit(evals, N, M))
    except Exception as e:  # the release logs it and reports -1 for every MP column
        out["mp_fit_error"] = f"{type(e).__name__}: {e}"
    out["detX_num"] = detx_constraint(evals)[0]
    out["detX_num_logsum"] = detx_num_logsum(evals)
    out["num_pl_spikes"] = num_pl_spikes(evals)
    if out["num_pl_spikes"] is not None:
        out["ERG_gap"] = out["detX_num"] - out["num_pl_spikes"]
        out["ERG_gap_logsum"] = out["detX_num_logsum"] - out["num_pl_spikes"]
    if "bulk_min" in out and N != M:
        out["frac_below_bulk_min"] = float(np.count_nonzero(evals < out["bulk_min"]) / evals.size)
    return out


def _mean(values):
    arr = np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else float("nan")


@register_task("geometry_ww_randomized")
class WWRandomizedTask(DiagnosticTask):
    """WeightWatcher's distance of every weight matrix from its element-shuffled
    copy, its MP-bulk fit and SETOL's ERG gap, averaged over the matrices."""

    TOKEN_POLICY = "not_token_based"
    SOURCES = {
        "paper": ("arXiv:2506.04434 (Prakash & Martin 2025, Grokking and Generalization Collapse: "
                  "Insights from HTSR theory); arXiv:2507.17912 (Martin & Hinrichs 2025, SETOL); "
                  "arXiv:2410.17770 (Staats, Thamm & Rosenow 2024, Small Singular Values Matter); "
                  "arXiv:1810.01075 (Martin & Mahoney 2021, JMLR, MP soft rank)"),
        "code": "CalculatedContent/WeightWatcher@a5940f79e9d5bbc37c82360c6f2f8a3aefe6543a",
        "llm_precedent": ("arXiv:2410.17770 (departures from random-matrix theory in the weight "
                          "matrices of pretrained transformer language models, scored by "
                          "perplexity); arXiv:2507.17912 (SETOL's ERG condition on state-of-the-art "
                          "pretrained models)"),
    }
    ADAPTATION_NOTE = (
        "Each matrix is measured as WeightWatcher.analyze(randomize=True, mp_fit=True, detX=True) "
        "measures it without CUDA (float32 weights, float64 scipy SVDs; with CUDA the release reads "
        "float16 weights and runs float32 SVDs), over the language model's Linear and HF Conv1D "
        "weights that the release's layer_supported admits (8 <= M <= 15000, N <= 50000, so a "
        "vocabulary head above 50000 rows is skipped as the release skips it). "
        "The release's unseeded global np.random shuffle is replaced by np.random.RandomState("
        "shuffle_seed), consumed in the release's order, so analyze() after "
        "np.random.seed(shuffle_seed) shuffles identically; one draw per model gives the nulls. "
        "Where the release reports -1 for a failed MP or power-law fit, and ERG_gap = detX_num + 1 "
        "after a failed power-law fit, BLME reports NaN and counts the failures. "
        "The release's detX_num multiplies float64 suffix products, which underflow on wide "
        "matrices (Qwen3-8B q/o projections: off by up to 84), so a log-domain detX_num_logsum is "
        "kept beside it in _meta_. "
        "frac_below_bulk_min, the fraction of eigenvalues below the fitted MP lower edge, is a BLME "
        "composition after Staats et al. and is NaN for square matrices, whose lower edge is 0. "
        "num_spikes, detX_num and ERG_gap are counts that grow with width, so their fractions of the "
        "M eigenvalues are reported alongside; spike_fraction_mean is the headline one. "
        "No headline is invariant to zero-padding or isometric embedding: both change what a shuffle "
        "of the entries produces."
    )
    HEADLINE_KEYS = ("rand_distance_mean", "ww_softrank_mean", "mp_softrank_mean",
                     "spike_fraction_mean", "frac_below_bulk_min_mean")
    SIZE_INVARIANT = {}

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running WeightWatcher randomized / MP-fit Analysis...")
        seed = int(self.config.get("shuffle_seed", DEFAULT_SHUFFLE_SEED))
        names, weights, counts = _ww_matrices(model)
        if not weights:
            return {"error": "No Linear / Conv1D weight matrices that WeightWatcher analyses"}
        rng = np.random.RandomState(seed)
        rows = []
        with torch.no_grad():
            for w in weights:
                W32 = w.detach().to(device="cpu", dtype=torch.float32).numpy()
                rows.append(ww_matrix_metrics(W32, rng))

        def col(key):
            return [r.get(key) for r in rows]

        n_evals = col("n_evals")
        frac = lambda key: [None if v is None else v / n for v, n in zip(col(key), n_evals)]
        out = {
            "rand_distance_mean": _mean(col("rand_distance")),
            "ww_softrank_mean": _mean(col("ww_softrank")),
            "mp_softrank_mean": _mean(col("mp_softrank")),
            "spike_fraction_mean": _mean(frac("num_spikes")),
            "frac_below_bulk_min_mean": _mean(col("frac_below_bulk_min")),
            "_meta_modules": names,
            "_meta_n_evals": n_evals,
            "_meta_Q": col("Q"),
        }
        for key in ("rand_distance", "ww_softrank", "num_spikes", "mp_softrank", "sigma_mp",
                    "bulk_min", "bulk_max", "detX_num", "detX_num_logsum", "num_pl_spikes",
                    "ERG_gap", "ERG_gap_logsum", "frac_below_bulk_min"):
            out[f"_meta_{key}"] = [float("nan") if v is None else v for v in col(key)]
        out["_meta_num_spikes_mean"] = _mean(col("num_spikes"))
        out["_meta_ERG_gap_mean"] = _mean(col("ERG_gap"))
        out["_meta_ERG_gap_fraction_mean"] = _mean(frac("ERG_gap"))
        out["_meta_ERG_gap_logsum_fraction_mean"] = _mean(frac("ERG_gap_logsum"))
        out["_meta_n_matrices"] = len(rows)
        out["_meta_n_square"] = sum(int(r["Q"] == 1.0) for r in rows)
        out["_meta_n_mp_fit_failed"] = sum(int("mp_fit_error" in r) for r in rows)
        out["_meta_n_pl_fit_failed"] = sum(int(r.get("num_pl_spikes") is None and "reason" not in r)
                                           for r in rows)
        # Matrices whose float64 suffix products under- or overflowed, so the
        # release's detX_num is not the det X = 1 crossing.
        out["_meta_n_detX_float_mismatch"] = sum(
            int(r.get("detX_num") != r.get("detX_num_logsum")) for r in rows)
        out["_meta_n_all_zero_matrices"] = sum(int("reason" in r) for r in rows)
        out["_meta_n_head_tagged"] = counts["head_tagged"]
        out["_meta_n_modules_outside_lm"] = counts["outside_lm"]
        out["_meta_n_conv_kernels_skipped"] = counts["conv_kernels_skipped"]
        out["_meta_n_below_min_evals"] = counts["below_min_evals"]
        out["_meta_n_above_max_evals"] = counts["above_max_evals"]
        out["_meta_n_above_max_N"] = counts["above_max_n"]
        out["_meta_shuffle_seed"] = seed
        errors = [f"{n}: {r['mp_fit_error']}" for n, r in zip(names, rows) if "mp_fit_error" in r]
        if errors:
            out["_meta_mp_fit_errors"] = errors
        return out
