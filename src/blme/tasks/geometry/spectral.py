
from ...tasks.base import DiagnosticTask
from ...registry import register_task
import torch
import numpy as np
from tqdm import tqdm
import logging
logger = logging.getLogger("blme")

try:
    from transformers.pytorch_utils import Conv1D as _HFConv1D
except Exception:
    _HFConv1D = None


def _pl_mle_alpha_ks(evals: "np.ndarray", min_tail: int = 2) -> tuple:
    """Official-style HT-SR power-law fit (campaign-4 B8).

    Clauset-Shalizi-Newman continuous-MLE power-law fit to the EIGENVALUE
    spectral density lambda = sigma^2 with a KS-optimal x_min scan — the
    procedure WeightWatcher's fit_powerlaw implements (Martin & Mahoney;
    Yang et al., KDD 2023). For each candidate x_min (every order
    statistic, leaving at least ``min_tail`` tail points):

        alpha(x_min) = 1 + n_tail / sum(ln(lambda_i / x_min))
        KS(x_min)    = max_i | F_emp(lambda_i) - F_PL(lambda_i) |

    and the fit with minimal KS is returned as (alpha, x_min, ks, n_tail).
    This complements the fixed-20%-tail Hill proxy on SINGULAR values kept
    for continuity (at a matched cutoff, alpha_sigma = 2*alpha_lambda - 1).
    """
    lam = np.sort(np.asarray(evals, dtype=np.float64))
    # EXACT positivity. The PL-MLE alpha depends only on RATIOS lambda_i/x_min
    # and is therefore invariant to any positive rescaling of the operator; an
    # absolute 1e-12 floor is not. On diag(10..1) scaled by 1e-8 the stable
    # rank stayed 3.850000 while the absolute cut silently changed the tail
    # population (adversarial review, 2026-09-04).
    lam = lam[lam > 0]
    n = lam.size
    if n < min_tail + 2:
        return float("nan"), float("nan"), float("nan"), 0

    # SMALL-N BRANCH, ported from the released WWFit.
    #
    # WeightWatcher switches estimator below `SMALL_N_CUTOFF = 20`
    # (`WW_powerlaw.WWFit.fit_power_law` -> `fit_powerlaw_smallN`, printing
    # "SMALL N PL FIT"), and BLME implemented only the standard path. Measured
    # against LIVE `weightwatcher 0.7.7` `pl_fit(..., pl_package="ww")` on
    # diag(N..1):
    #
    #     N=20  BLME 3.187218099  WWFit 3.187218099   rel 7.0e-16   (exact)
    #     N=19  BLME 3.042563820  WWFit 2.787243343   rel 9.2e-02
    #     N=10  BLME 2.840775179  WWFit 1.622513605   rel 7.5e-01
    #     N=9   BLME 2.585706731  WWFit 1.482316218   rel 7.4e-01
    #     N=8   BLME 2.328817743  WWFit 1.330045362   rel 7.5e-01
    #
    # So the N >= 20 path is already bit-faithful and only N < 20 diverges.
    # Three things differ, all of them load-bearing:
    #   1. a BIAS-CORRECTED exponent, `1 + (n-1)/s`, not `1 + n/s`;
    #   2. a `k_min = 8` floor -- candidate tails shorter than 8 are SKIPPED,
    #      leaving `Ds[i]` at its initial 1.0 and `alphas[i]` at 0.0;
    #   3. the winner. `fit_powerlaw_smallN` selects by `argmin(Js)` with
    #      `J = D_ks - 0.868/sqrt(n)`, and then `WWFit.__init__` OVERWRITES
    #      that with `i = np.argmin(self.Ds)`. The J objective is computed and
    #      discarded. Only reading the constructor shows this; the method reads
    #      as though J decides.
    #
    # Ported verbatim rather than approximated, because the campaign rule is
    # that the released reference wins even where its behaviour looks odd.
    if n < 20:
        _log = np.log(lam)
        _alphas = np.zeros(n - 1, dtype=np.float64)
        _Ds = np.ones(n - 1, dtype=np.float64)
        _any_valid = False   # mirrors "np.isfinite(self.Js).any()"
        for _i in range(n - 1):
            _ni = n - _i
            if _ni < 8:                      # k_min
                continue
            _nf = float(_ni)
            _s = float(np.sum(_log[_i:]) - _nf * _log[_i])
            if _s <= 1e-12:
                continue
            _abc = 1.0 + (_nf - 1.0) / _s    # bias-corrected MLE
            _alphas[_i] = _abc               # recorded BEFORE the alpha>1 gate
            if _abc <= 1.0:
                continue
            _tail = lam[_i:]
            _F_fit = 1.0 - (_tail / lam[_i]) ** (1.0 - _abc)
            _F_emp = np.arange(_ni, dtype=np.float64) / _nf
            _Ds[_i] = float(np.max(np.abs(_F_emp - _F_fit)))
            _any_valid = True
        # FALLBACK, also from the released source. When `k_min = 8` rejects
        # every candidate -- which is every spectrum with fewer than 8 positive
        # eigenvalues -- WWFit falls back to i=0 with the WHOLE spectrum as the
        # tail. Without it this returned alphas[0] = 0.0 against WWFit's
        # 1.283126297 / 1.228201413 / 1.283998227 / 1.073586767 on four of 60
        # random spectra (all n < 8). Found by validating the port on random
        # matrices after it was already bit-exact on the diagonal fixture the
        # reviewer supplied -- one fixture is not a port.
        if not _any_valid:
            _nf = float(n)
            _s = float(np.sum(_log) - _nf * _log[0])
            if _s <= 1e-12:
                # WWFit's pathological branch: alpha 1.0, D 1.0.
                return 1.0, float(lam[0]), 1.0, int(n)
            _abc = 1.0 + (_nf - 1.0) / _s
            _alphas[0] = _abc
            _F_fit = 1.0 - (lam / lam[0]) ** (1.0 - _abc)
            _F_emp = np.arange(n, dtype=np.float64) / _nf
            _Ds[0] = float(np.max(np.abs(_F_emp - _F_fit)))
        _j = int(np.argmin(_Ds))             # the constructor's KS winner
        return (float(_alphas[_j]), float(lam[_j]), float(_Ds[_j]),
                int(n - _j))
    best = (np.inf, float("nan"), float("nan"), 0)
    # Scan candidate x_min over ALL order statistics up to a generous cap
    # (2000 covers every eigenvalue count that occurs in practice for a
    # single weight matrix; beyond it, evenly subsample).
    # EVERY order statistic. Subsampling to 2000 candidates does not deliver a
    # KS-OPTIMAL fit, which is what this function claims and what the released
    # WeightWatcher computes: on 4096 valid eigenvalues the exhaustive optimum
    # sits at xmin=1.0 with KS=0.085485, n_tail=10, alpha=24.985543, and the
    # subsampled grid skipped that order statistic entirely (adversarial review
    # round 2, 2026-09-04). Milliseconds at single-weight-matrix sizes.
    # min_tail lowered 8 -> 4 -> 2 (2026-09-04). Eight forbade every cutoff
    # leaving eight or fewer eigenvalues, so on a 50x50 diagonal spectrum the
    # scan could not reach the true KS optimum and settled at xmin=0.081658,
    # n_tail=46, alpha=1.379708. Four still excluded the 2- and 3-point
    # candidates the RELEASED WWFit considers, and on a 21x21 diagonal with
    # eigenvalues [1 x 18, 1.01, 1.1, 10] BLME picked xmin=1.0, n_tail=4,
    # alpha=2.661236064 where WWFit picks a shorter tail (adversarial review
    # round 4, 2026-09-04). Two is where the MLE is defined; KS decides.
    #
    # MEASURED CONSEQUENCE: on genuine power-law data this is a NO-OP. Over
    # four planted Pareto samples at n=200, min_tail 2 / 4 / 8 give IDENTICAL
    # fits -- the KS optimum is a 160-200 point tail and alpha recovers
    # 3.2257-3.7530 against a planted 3.5. A 2-point tail only wins on TINY
    # synthetic spectra (n = 10-21), where no meaningful power-law fit exists
    # at all; `_meta_n_alpha_short_tail` counts those so a fit resting on
    # almost nothing is visible rather than silently averaged in.
    cand_idx = np.arange(0, max(1, n - min_tail + 1))
    for i in cand_idx:
        x_min = lam[i]
        tail = lam[i:]
        n_tail = tail.size
        log_ratio = np.log(tail / x_min)
        denom = log_ratio.sum()
        if denom <= 0:
            continue
        # PLAIN n. NOT the (n-1) "small-sample branch".
        #
        # I added an `(n-1)/sum-log` form below 20 tail points on a round-4
        # reviewer's claim that the released WWFit uses it. I did not check it
        # against live WeightWatcher, and it was WRONG in a way that silently
        # destroyed the estimator: with the (n-1) numerator the KS statistic at
        # n_tail = 2 collapses to the DATA-INDEPENDENT constant
        # 1 - e^-1 - 0.5 = 0.132120558828558 -- verified identical on
        # [1,10], [1,1e6], [3,3.0001] and [2,100] -- so the KS scan selected a
        # 2-point tail on essentially every matrix. Against live WeightWatcher
        # 0.7.7 the plain-n form matches to mean 3.634e-15 while the (n-1) form
        # was off by mean 0.266 and max 6.64 over gpt2's 48 modules; one
        # module read alpha 9.631988 at n_tail=9 against 2.992027 at n_tail=144
        # (+221.9%) (independent Opus review, 2026-09-05).
        #
        # Plain n is also self-protecting here: its own n_tail=2 KS is the
        # constant 0.364664716763387, large enough that a 2-point tail does not
        # win against any real fit.
        alpha = 1.0 + n_tail / denom
        # KS between the empirical tail CDF and the fitted Pareto CDF.
        # LEFT-continuous empirical CDF (arange(0,n)/n) — the Clauset
        # plfit / powerlaw / WeightWatcher convention. The right-continuous
        # variant coincides on smooth planted spectra but lands in different
        # KS-minima on real weight spectra (audit 2026-07-16: mean |dalpha|
        # 0.41 vs live WW on gpt2's 48 modules; left-continuous: 0.07).
        emp = np.arange(n_tail) / n_tail
        model_cdf = 1.0 - (x_min / tail) ** (alpha - 1.0)
        ks = float(np.max(np.abs(emp - model_cdf)))
        if ks < best[0]:
            best = (ks, float(alpha), float(x_min), int(n_tail))
    if not np.isfinite(best[1]):
        return float("nan"), float("nan"), float("nan"), 0
    return best[1], best[2], best[0], best[3]


@register_task("geometry_spectral")
class WeightSpectralTask(DiagnosticTask):
    """
    Analyzes the spectral properties of weight matrices.
    Metrics:
    - Stable Rank: ||W||_F^2 / ||W||_2^2 (definition: Rudelson &
      Vershynin 2007, JACM 54(4); used in DNN generalization bounds by
      Neyshabur, Bhojanapalli & Srebro 2018, arXiv:1707.09564)
    - Power Law Alpha: Hill estimator over the top tail of the *singular
      values* (Martin & Mahoney HT-SR, 2021). NB: Martin & Mahoney fit the
      power law to the empirical spectral density of the *eigenvalues*
      λ = σ² (via the powerlaw MLE with a KS-optimal x_min), so this
      ``alpha`` is on a different scale (≈ 2·α_MM − 1).

      RANKING IS **NOT** PRESERVED. The identity α_σ = 2·α_λ − 1 holds only
      at a MATCHED x_min, which a fixed top-20% tail never supplies. Measured
      over gpt2's 49 modules: Spearman(per-module Hill, α_official) = **0.38**,
      and (hill − (2·α_official − 1)) has mean −1.29 with std 1.69. An earlier
      version of this docstring claimed ranking was preserved; it is not
      (adversarial review, 2026-09-03).

      Use ``avg_alpha_official`` for anything that needs to correspond to the
      published WeightWatcher α — it is the Clauset PL-MLE on the eigenvalues
      with a KS-optimal x_min and matches live weightwatcher 0.7.7 across all
      49 gpt2 modules (mean |Δα| 3e-4, max 2.1e-3). ``avg_alpha`` is the
      fixed-tail Hill proxy, kept for continuity. See AUDIT_V2 §5.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Weight Spectral Analysis...")

        # Configurable tail fraction for Hill estimator (default 20%)
        tail_fraction = self.config.get("tail_fraction", 0.2)

        results = {}
        layer_stats = {}

        # GPT-2 uses transformers.pytorch_utils.Conv1D (NOT torch.nn.Conv1d)
        # for its QKV/MLP projections. Skipping it degrades the analysis to
        # a single nn.Linear (lm_head) and makes std_alpha = 0.
        TARGET_MODULES = [torch.nn.Linear, torch.nn.Conv1d]
        if _HFConv1D is not None:
            TARGET_MODULES.append(_HFConv1D)
        TARGET_MODULES = tuple(TARGET_MODULES)

        alphas = []
        stable_ranks = []

        # Martin, Peng & Mahoney explicitly EXCLUDE the final classifier /
        # unembedding from the HT-SR layer scan -- it is not a transformer
        # weight matrix and its spectrum is shaped by the vocabulary, not by
        # training dynamics. Including it moved avg_alpha_hat 10.9424 -> 11.1009
        # (+1.4%) (adversarial review, 2026-09-04).
        _n_alpha_over_cap = [0]
        # Exclude by MODULE IDENTITY, not by name.
        #
        # A leaf-name match silently fails on nested heads: roberta-base's
        # unembedding is `lm_head.decoder` and BERT's is
        # `cls.predictions.decoder`, so neither the leaf `decoder` nor the full
        # name matched and the vocabulary-sized matrix was averaged straight
        # back in. `model.get_output_embeddings()` returns the right module on
        # gpt2, pythia-70m, opt-125m and roberta-base alike (adversarial
        # review, 2026-09-04). The name list is kept only as a fallback for
        # models that do not implement that accessor.
        _out_emb = None
        try:
            _out_emb = model.get_output_embeddings()
        except Exception:
            _out_emb = None
        _EXCLUDE_SUFFIXES = ("lm_head", "embed_out", "output_projection")
        modules_to_scan, _n_excluded = [], 0
        for name, module in model.named_modules():
            if isinstance(module, TARGET_MODULES):
                if "weight" in module._parameters and module.weight is not None:
                    _leaf = name.rsplit(".", 1)[-1]
                    # ADDITIVE, not either/or: identity catches nested heads
                    # the name list misses (roberta `lm_head.decoder`, BERT
                    # `cls.predictions.decoder`), and the name list catches
                    # heads an accessor might not return. Requiring the
                    # accessor to be None before consulting names left a hole.
                    _is_head = (
                        (module is _out_emb)
                        or _leaf in _EXCLUDE_SUFFIXES
                        or name in _EXCLUDE_SUFFIXES)
                    if _is_head:
                        _n_excluded += 1
                    # SCANNED EITHER WAY, tagged rather than dropped. The
                    # headline population still excludes the unembedding (HT-SR
                    # treats it as vocabulary-shaped, not training-shaped), but
                    # the all-linear-modules population is a defensible reading
                    # of the same estimand -- pythia-70m's untied 50304x512
                    # `embed_out` is a real linear module -- so it is reported
                    # alongside instead of being silently unavailable
                    # (adversarial review, 2026-09-04).
                    modules_to_scan.append((name, module, _is_head))

        logger.info(f"  Found {len(modules_to_scan)} linear modules.")

        alphas_official, alpha_hats = [], []
        alphas_head, alphas_nohead = [], []
        _ks_all, _ntail_all = [], []
        _sr_norms = []
        _n_short_tail = [0]
        _n_alpha_undefined = [0]
        for name, module, _is_head in tqdm(modules_to_scan, desc="Analyzing Weights"):
            W = module.weight.detach().float()

            # RESHAPE conv kernels, do not silently skip them. `torch.nn.Conv1d`
            # is in TARGET_MODULES and was COUNTED in
            # `_meta_n_modules_scanned`, yet its 3-D kernel fell through here,
            # so a Conv1d-only model reported a scanned module and three
            # headline values computed from nothing, with no undefined count
            # (adversarial review round 2, 2026-09-04). WeightWatcher folds a
            # conv kernel to (out_channels, in_channels * prod(kernel)), which
            # is the operator the layer applies; do the same.
            if W.dim() > 2:
                W = W.reshape(W.shape[0], -1)
            if W.dim() != 2:
                _n_alpha_undefined[0] += 1
                continue

            try:
                # `safe_svdvals`: numpy/torch compute the SVD internally and then CAST
                # the singular values back to the input dtype, and that cast overflows
                # to inf near 3.4e38 -- so a scale-FREE consumer downstream (effective
                # rank, RankMe, IsoScore, alpha) returns NaN on data it should handle
                # exactly. The helper factors out the max first, which only shifts
                # exponents (2026-09-07).
                from .utils import safe_svdvals
                S = safe_svdvals(W)
                S_np = S.cpu().numpy()

                # 1. Stable Rank: ||W||_F^2 / ||W||_2^2 -- a RATIO, so it is
                # exactly invariant to rescaling W. The arithmetic was not:
                # squaring unscaled float32 singular values made diag(10..1),
                # whose exact stable rank is 3.850000 at every nonzero scale,
                # read 4.000000 at 1e-23 (+3.90%) (adversarial review round 3,
                # 2026-09-04). Dividing by S[0] cancels in the ratio.
                if len(S_np) > 0 and S_np[0] > 0:
                    _sn = np.asarray(S_np, dtype=np.float64) / float(S_np[0])
                    stable_rank = float(np.sum(_sn ** 2))
                    # The MANUSCRIPT also asks for the width-normalized form,
                    # stable_rank / min(m, n), and it was not emitted at all.
                    # It matters: rho(avg_stable_rank, d_model) = +0.8944 over
                    # 5 models, so the raw value is largely a width proxy
                    # (independent Opus review, 2026-09-04). Note this is a
                    # DIVISION by an architecture scalar and so is subject to
                    # the same mechanism bar as the removed twins -- it is
                    # emitted as `_meta_` until it passes one.
                    _sr_norm = stable_rank / max(1, min(W.shape))
                else:
                    stable_rank = float("nan")
                    _sr_norm = float("nan")

                # 2. Power Law Alpha (Martin & Mahoney 2021)
                # Hill estimator on top tail_fraction of singular values.
                k = max(2, int(tail_fraction * len(S_np)))
                top_k = S_np[:k]

                # Hill Estimator: alpha = 1 + k / sum(ln(x_i / x_min))
                # `> 0`, not `> 1e-6`: the Hill estimator sees only the
                # ratios x_i/x_min, so it is exactly scale-invariant and an
                # absolute floor is not (same measurement as above).
                if k > 0 and top_k[-1] > 0:
                    x_min = top_k[-1]
                    log_sum = np.sum(np.log(top_k / x_min))
                    if log_sum > 0:
                        # NOT clipped to [0, 20]. A Hill estimate above 20 is a
                        # real (if extreme) measurement, and clipping made it
                        # indistinguishable from a genuine 20: a constructed
                        # spectrum with a true Hill of 133.259 emitted exactly
                        # 20.000, and gemma-2b already reports max_alpha = 20.0
                        # at the boundary (adversarial review, 2026-09-04).
                        # Censoring at the cap also compresses the top of the
                        # cross-model range toward a shared constant. Keep the
                        # value and record when it exceeds the old cap.
                        alpha = float(1.0 + k / log_sum)
                        if alpha > 20.0:
                            _n_alpha_over_cap[0] += 1
                    else:
                        # NaN, NOT 0.0. log_sum == 0 means a perfectly FLAT
                        # tail, where 1 + k/log_sum has no finite value -- the
                        # least heavy-tailed spectrum there is. Reporting 0.0
                        # placed it at the opposite, MOST heavy-tailed end of
                        # the scale: W = I_10 emitted avg/min/max alpha all
                        # 0.000000 (adversarial review, 2026-09-04).
                        alpha = float("nan")
                else:
                    alpha = float("nan")

                # Official-style PL-MLE alpha on eigenvalues (B8).
                evals = S_np.astype(np.float64) ** 2
                alpha_off, _xmin, _ks, _ntail = _pl_mle_alpha_ks(evals)
                lambda_max = float(evals.max()) if evals.size else float("nan")
                alpha_hat = (alpha_off * np.log10(lambda_max)
                             if np.isfinite(alpha_off) and lambda_max > 0
                             else float("nan"))

                layer_stats[name] = {
                    "stable_rank": float(stable_rank),
                    "alpha": float(alpha),
                    "alpha_official": float(alpha_off),
                    "alpha_official_xmin": float(_xmin),
                    "alpha_official_ks": float(_ks),
                    "alpha_official_n_tail": int(_ntail),
                    "alpha_hat": float(alpha_hat),
                    "spectral_norm": float(S_np[0]) if len(S_np) > 0 else 0,
                    "stable_rank_normalized": float(_sr_norm),
                }

                # NaN means "undefined for this matrix" (a perfectly flat
                # tail), not "zero". Excluded from the summaries and counted,
                # exactly as alpha_official already was; averaging it in would
                # NaN out the whole model.
                # HEADLINE = ALL LINEAR MODULES. Both the BLME manuscript
                # ("across all linear modules") and weightwatcher's default
                # layer scan include an untied output embedding, and
                # pythia-70m's 50304x512 `embed_out` is an ordinary Linear.
                # Excluding it made the headline match neither (adversarial
                # review round 2, 2026-09-04). The head-EXCLUDED population --
                # the HT-SR reading, in which the unembedding is shaped by the
                # vocabulary rather than by training dynamics -- is kept under
                # its own `_meta_` name so both remain available.
                if np.isfinite(alpha):
                    alphas.append(alpha)
                    (alphas_head if _is_head else alphas_nohead).append(alpha)
                else:
                    _n_alpha_undefined[0] += 1
                if np.isfinite(alpha_off):
                    alphas_official.append(alpha_off)
                    _ks_all.append(float(_ks))
                    _ntail_all.append(int(_ntail))
                    if int(_ntail) < 10:
                        _n_short_tail[0] += 1
                if np.isfinite(alpha_hat):
                    alpha_hats.append(alpha_hat)
                if np.isfinite(stable_rank):
                    stable_ranks.append(stable_rank)
                    _sr_norms.append(_sr_norm)

            except Exception as e:
                logger.info(f"Error analyzing {name}: {e}")
                continue

        results["avg_alpha"] = float(np.mean(alphas)) if alphas else float("nan")
        results["avg_stable_rank"] = float(np.mean(stable_ranks)) if stable_ranks else float("nan")
        results["min_alpha"] = float(np.min(alphas)) if alphas else float("nan")
        results["max_alpha"] = float(np.max(alphas)) if alphas else float("nan")
        results["std_alpha"] = float(np.std(alphas)) if alphas else float("nan")
        results["median_alpha"] = float(np.median(alphas)) if alphas else float("nan")
        # Official-style PL-MLE summaries (KS-optimal x_min on eigenvalues).
        results["avg_alpha_official"] = (float(np.mean(alphas_official))
                                         if alphas_official else float("nan"))
        results["median_alpha_official"] = (float(np.median(alphas_official))
                                            if alphas_official else float("nan"))
        results["avg_alpha_hat"] = (float(np.mean(alpha_hats))
                                    if alpha_hats else float("nan"))
        results["tail_fraction"] = tail_fraction

        results["_meta_n_modules_scanned"] = int(len(modules_to_scan))
        # RENAMED: nothing is excluded. The code deliberately SCANS the
        # unembedding and TAGS it (see the note at the scan loop), so a key
        # called `..._excluded_unembedding` reporting 1 while `avg_alpha`
        # includes it (5.5195 with, 5.5156 without) contradicted the code's own
        # decision (red-team review, 2026-09-05).
        results["_meta_n_head_modules_tagged"] = int(_n_excluded)
        # How many modules exceeded the old hard cap of 20 -- non-zero means
        # the pre-fix numbers were censored for this model.
        results["_meta_n_alpha_over_old_cap"] = int(_n_alpha_over_cap[0])
        results["_meta_n_alpha_undefined"] = int(_n_alpha_undefined[0])
        # The all-linear-modules population, unembedding included.
        # The HT-SR population (unembedding excluded), kept for continuity.
        results["_meta_avg_alpha_no_head"] = (float(np.mean(alphas_nohead))
                                              if alphas_nohead else float("nan"))
        # `_meta_n_head_modules_tagged` above already carries this count; two
        # names for one number is the collinearity this campaign has removed
        # elsewhere (independent Opus review, 2026-09-05).
        # FIT DIAGNOSTICS, exported. They were computed for every module and
        # then discarded, so a reader could not tell a well-supported
        # power-law fit from a marginal one: pythia-70m's `embed_out` retained
        # alpha_official = 2.601242 while dropping xmin = 45.063035,
        # KS = 0.023539 and n_tail = 192 (adversarial review round 3,
        # 2026-09-04).
        # Fits resting on fewer than 10 tail points: not a power-law
        # measurement, and now visible.
        results["_meta_n_alpha_short_tail"] = int(_n_short_tail[0])
        results["_meta_avg_stable_rank_normalized"] = (
            float(np.mean(_sr_norms)) if _sr_norms else float("nan"))
        results["_meta_alpha_official_ks_median"] = (
            float(np.median(_ks_all)) if _ks_all else float("nan"))
        results["_meta_alpha_official_ks_max"] = (
            float(np.max(_ks_all)) if _ks_all else float("nan"))
        results["_meta_alpha_official_n_tail_median"] = (
            float(np.median(_ntail_all)) if _ntail_all else float("nan"))
        return results
