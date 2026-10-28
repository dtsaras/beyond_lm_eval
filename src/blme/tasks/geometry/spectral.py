from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import safe_svdvals
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
    """WeightWatcher's power-law fit of an eigenvalue spectrum lambda = sigma^2.

    Clauset-Shalizi-Newman continuous MLE with a KS-optimal x_min scan, the
    procedure WeightWatcher's ``WWFit`` implements (Martin & Mahoney):

        alpha(x_min) = 1 + n_tail / sum(ln(lambda_i / x_min))
        KS(x_min)    = max_i | F_emp(lambda_i) - F_PL(lambda_i) |

    over every order statistic leaving at least ``min_tail`` points; the
    minimal-KS fit is returned as (alpha, x_min, ks, n_tail). Below 20
    eigenvalues the released small-N branch is ported verbatim. Pinned
    against live WWFit by tests/test_reference_parity_guard.py and
    tests/tasks/parity/test_b8_new_metrics_parity.py.
    """
    lam = np.sort(np.asarray(evals, dtype=np.float64))
    # Exact positivity: alpha depends only on ratios lambda_i / x_min, so an
    # absolute floor would break scale invariance.
    lam = lam[lam > 0]
    n = lam.size
    # The bail-out applies to the N >= 20 path only; WWFit fits n = 2 too.
    if n >= 20 and n < min_tail + 2:
        return float("nan"), float("nan"), float("nan"), 0
    if n < 2:
        return float("nan"), float("nan"), float("nan"), 0

    # Small-N branch of the released WWFit (SMALL_N_CUTOFF = 20): a
    # bias-corrected exponent 1 + (n-1)/s, a k_min = 8 floor (shorter tails
    # are skipped, leaving D = 1 and alpha = 0), and the constructor's
    # argmin(D) winner (the J objective is computed and discarded).
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
        # Released fallback: when k_min rejects every candidate (fewer than 8
        # positive eigenvalues), fit the whole spectrum from i = 0.
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
    # Every order statistic is a candidate (a subsampled grid misses the KS
    # optimum), down to 2-point tails as WWFit allows; plain-n MLE, whose
    # 2-point KS is a large constant and never beats a real fit.
    cand_idx = np.arange(0, max(1, n - min_tail + 1))
    for i in cand_idx:
        x_min = lam[i]
        tail = lam[i:]
        n_tail = tail.size
        log_ratio = np.log(tail / x_min)
        denom = log_ratio.sum()
        if denom <= 0:
            continue
        alpha = 1.0 + n_tail / denom
        # Left-continuous empirical CDF (arange(0, n)/n), the plfit /
        # powerlaw / WeightWatcher convention.
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
      ``alpha`` is on a different scale (≈ 2·α_MM − 1), and the ranking is
      not preserved (a fixed top-20% tail never matches the KS-optimal x_min).

      The headline is ``avg_alpha_official`` (WeightWatcher's alpha): the
      Clauset PL-MLE on the eigenvalues with a KS-optimal x_min, matching live
      weightwatcher 0.7.7 across all 49 gpt2 modules (mean |d alpha| 3e-4,
      max 2.1e-3). The fixed-tail Hill proxy is ``_meta_avg_alpha`` (and its
      min/max/std/median), kept for continuity.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Weight Spectral Analysis...")

        # Configurable tail fraction for Hill estimator (default 20%)
        tail_fraction = self.config.get("tail_fraction", 0.2)

        results = {}

        # GPT-2's projections are transformers' Conv1D, not torch.nn.Conv1d.
        TARGET_MODULES = [torch.nn.Linear, torch.nn.Conv1d]
        if _HFConv1D is not None:
            TARGET_MODULES.append(_HFConv1D)
        TARGET_MODULES = tuple(TARGET_MODULES)

        alphas = []
        stable_ranks = []
        _svd_zeros_total = 0
        _svd_modules_affected = 0
        _n_alpha_over_cap = [0]

        # The unembedding is scanned and TAGGED (by module identity, with a
        # name fallback for nested heads): the headline averages every linear
        # module, as WeightWatcher's default scan; the head-free mean is _meta_.
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
                    _is_head = (
                        (module is _out_emb)
                        or _leaf in _EXCLUDE_SUFFIXES
                        or name in _EXCLUDE_SUFFIXES)
                    if _is_head:
                        _n_excluded += 1
                    modules_to_scan.append((name, module, _is_head))

        logger.info(f"  Found {len(modules_to_scan)} linear modules.")

        alphas_official, alpha_hats = [], []
        alphas_nohead = []
        _ks_all, _ntail_all = [], []
        _sr_norms = []
        _n_short_tail = [0]
        _n_alpha_undefined = [0]
        for name, module, _is_head in tqdm(modules_to_scan, desc="Analyzing Weights"):
            W = module.weight.detach().float()

            # Conv kernels fold to (out, in * prod(kernel)), as WeightWatcher.
            if W.dim() > 2:
                W = W.reshape(W.shape[0], -1)
            if W.dim() != 2:
                _n_alpha_undefined[0] += 1
                continue

            try:
                # Max-factored SVD: exact at the top of the float32 range.
                S = safe_svdvals(W)
                S_np = S.cpu().numpy()

                # 1. Stable rank ||W||_F^2 / ||W||_2^2, a ratio: divide by S[0]
                # before squaring so it stays scale-invariant in arithmetic.
                if len(S_np) > 0 and S_np[0] > 0:
                    _sn = np.asarray(S_np, dtype=np.float64) / float(S_np[0])
                    stable_rank = float(np.sum(_sn ** 2))
                    # / min(m, n): a capacity ratio, _meta_ only.
                    _sr_norm = stable_rank / max(1, min(W.shape))
                else:
                    stable_rank = float("nan")
                    _sr_norm = float("nan")

                # Numerical zeros, resolved once for both alphas with numpy's
                # rank envelope eps * max(m, n) * S[0] (relative, so scale-free);
                # exact rotations of a rank-deficient matrix then agree with it
                # (tests/tasks/test_spectral_rank_rotation.py).
                _S64 = np.asarray(S_np, dtype=np.float64)
                _rank_tol = (float(np.finfo(np.float64).eps) * max(W.shape)
                             * float(_S64[0]) if _S64.size else 0.0)
                _S_res = _S64[_S64 > _rank_tol]
                _n_svd_zeros = int(_S64.size - _S_res.size)
                _svd_zeros_total += _n_svd_zeros
                _svd_modules_affected += int(_n_svd_zeros > 0)

                # 2. Hill estimator on the top tail_fraction of singular
                # values; k counts the full width, and alpha is undefined when
                # the k-th value is not among the resolved ones.
                k = max(2, int(tail_fraction * len(S_np)))
                top_k = _S_res[:k] if _S_res.size >= k else np.array([])

                # Hill Estimator: alpha = 1 + k / sum(ln(x_i / x_min))
                if k > 0 and top_k.size >= k and top_k[-1] > 0:
                    x_min = top_k[-1]
                    log_sum = np.sum(np.log(top_k / x_min))
                    # A tail whose relative spread is within the rank envelope
                    # is one repeated value: no finite MLE.
                    _spread = float((top_k[0] - x_min) / max(float(top_k[0]), 1e-300))
                    if _spread <= float(np.finfo(np.float64).eps) * max(W.shape):
                        log_sum = 0.0
                    if log_sum > 0:
                        # Not clipped; values above the old cap of 20 are counted.
                        alpha = float(1.0 + k / log_sum)
                        if alpha > 20.0:
                            _n_alpha_over_cap[0] += 1
                    else:
                        # A flat tail has no finite alpha: NaN, not 0.0.
                        alpha = float("nan")
                else:
                    alpha = float("nan")

                # Official-style PL-MLE alpha on the resolved eigenvalues.
                evals = _S_res ** 2
                alpha_off, _xmin, _ks, _ntail = _pl_mle_alpha_ks(evals)
                lambda_max = float(evals.max()) if evals.size else float("nan")
                alpha_hat = (alpha_off * np.log10(lambda_max)
                             if np.isfinite(alpha_off) and lambda_max > 0
                             else float("nan"))

                # NaN = undefined for this matrix: excluded and counted.
                if np.isfinite(alpha):
                    alphas.append(alpha)
                    if not _is_head:
                        alphas_nohead.append(alpha)
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

        # Singular values the rank envelope resolved to zero (0 on every
        # pretrained matrix tested): nonzero means a rank-deficient operator.
        results["_meta_alpha_official_n_svd_zeros"] = int(_svd_zeros_total)
        results["_meta_alpha_official_modules_with_svd_zeros"] = int(
            _svd_modules_affected)
        # The fixed-tail Hill proxy is _meta_: its tail length counts the
        # matrix width, so zero-padding a matrix moves it (not WeightWatcher's
        # alpha, which the headline keys below are).
        results["_meta_avg_alpha"] = float(np.mean(alphas)) if alphas else float("nan")
        results["avg_stable_rank"] = float(np.mean(stable_ranks)) if stable_ranks else float("nan")
        results["_meta_min_alpha"] = float(np.min(alphas)) if alphas else float("nan")
        results["_meta_max_alpha"] = float(np.max(alphas)) if alphas else float("nan")
        results["_meta_std_alpha"] = float(np.std(alphas)) if alphas else float("nan")
        results["_meta_median_alpha"] = float(np.median(alphas)) if alphas else float("nan")
        # Official-style PL-MLE summaries (KS-optimal x_min on eigenvalues).
        results["avg_alpha_official"] = (float(np.mean(alphas_official))
                                         if alphas_official else float("nan"))
        results["median_alpha_official"] = (float(np.median(alphas_official))
                                            if alphas_official else float("nan"))
        results["avg_alpha_hat"] = (float(np.mean(alpha_hats))
                                    if alpha_hats else float("nan"))
        # A config echo (0.2 on every model): _meta_.
        results["_meta_tail_fraction"] = tail_fraction

        results["_meta_n_modules_scanned"] = int(len(modules_to_scan))
        results["_meta_n_head_modules_tagged"] = int(_n_excluded)
        # Modules above the old hard cap of 20 (censored before the clip went).
        results["_meta_n_alpha_over_old_cap"] = int(_n_alpha_over_cap[0])
        results["_meta_n_alpha_undefined"] = int(_n_alpha_undefined[0])
        # The HT-SR population with the unembedding excluded.
        results["_meta_avg_alpha_no_head"] = (float(np.mean(alphas_nohead))
                                              if alphas_nohead else float("nan"))
        # Fit diagnostics; fits on fewer than 10 tail points are counted.
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
