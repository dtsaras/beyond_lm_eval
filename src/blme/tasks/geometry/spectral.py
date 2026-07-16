
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


def _pl_mle_alpha_ks(evals: "np.ndarray", min_tail: int = 8) -> tuple:
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
    lam = lam[lam > 1e-12]
    n = lam.size
    if n < min_tail + 2:
        return float("nan"), float("nan"), float("nan"), 0
    best = (np.inf, float("nan"), float("nan"), 0)
    # Scan candidate x_min over ALL order statistics up to a generous cap
    # (2000 covers every eigenvalue count that occurs in practice for a
    # single weight matrix; beyond it, evenly subsample).
    max_candidates = 2000
    cand_idx = np.unique(np.linspace(0, n - min_tail - 1,
                                     min(max_candidates, n - min_tail)).astype(int))
    for i in cand_idx:
        x_min = lam[i]
        tail = lam[i:]
        n_tail = tail.size
        log_ratio = np.log(tail / x_min)
        denom = log_ratio.sum()
        if denom <= 0:
            continue
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
      ``alpha`` is on a different scale (≈ 2·α_MM − 1) and is a
      monotone transform of the published WeightWatcher α — ranking is
      preserved but values are not comparable. See AUDIT_V2 §5.
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

        modules_to_scan = []
        for name, module in model.named_modules():
            if isinstance(module, TARGET_MODULES):
                if "weight" in module._parameters and module.weight is not None:
                    modules_to_scan.append((name, module))

        logger.info(f"  Found {len(modules_to_scan)} linear modules.")

        alphas_official, alpha_hats = [], []
        for name, module in tqdm(modules_to_scan, desc="Analyzing Weights"):
            W = module.weight.detach().float()

            if W.dim() != 2:
                continue

            try:
                S = torch.linalg.svdvals(W)
                S_np = S.cpu().numpy()

                # 1. Stable Rank: ||W||_F^2 / ||W||_2^2
                if len(S_np) > 0 and S_np[0] > 0:
                    fro_sq = np.sum(S_np ** 2)
                    spec_sq = S_np[0] ** 2
                    stable_rank = fro_sq / spec_sq
                else:
                    stable_rank = 0.0

                # 2. Power Law Alpha (Martin & Mahoney 2021)
                # Hill estimator on top tail_fraction of singular values.
                k = max(2, int(tail_fraction * len(S_np)))
                top_k = S_np[:k]

                # Hill Estimator: alpha = 1 + k / sum(ln(x_i / x_min))
                if k > 0 and top_k[-1] > 1e-6:
                    x_min = top_k[-1]
                    log_sum = np.sum(np.log(top_k / x_min))
                    if log_sum > 0:
                        alpha = float(np.clip(1 + k / log_sum, 0, 20))
                    else:
                        alpha = 0.0
                else:
                    alpha = 0.0

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
                    "alpha_hat": float(alpha_hat),
                    "spectral_norm": float(S_np[0]) if len(S_np) > 0 else 0,
                }

                alphas.append(alpha)
                if np.isfinite(alpha_off):
                    alphas_official.append(alpha_off)
                if np.isfinite(alpha_hat):
                    alpha_hats.append(alpha_hat)
                stable_ranks.append(stable_rank)

            except Exception as e:
                logger.info(f"Error analyzing {name}: {e}")
                continue

        results["avg_alpha"] = float(np.mean(alphas)) if alphas else 0.0
        results["avg_stable_rank"] = float(np.mean(stable_ranks)) if stable_ranks else 0.0
        results["min_alpha"] = float(np.min(alphas)) if alphas else 0.0
        results["max_alpha"] = float(np.max(alphas)) if alphas else 0.0
        results["std_alpha"] = float(np.std(alphas)) if alphas else 0.0
        results["median_alpha"] = float(np.median(alphas)) if alphas else 0.0
        # Official-style PL-MLE summaries (KS-optimal x_min on eigenvalues).
        results["avg_alpha_official"] = (float(np.mean(alphas_official))
                                         if alphas_official else float("nan"))
        results["median_alpha_official"] = (float(np.median(alphas_official))
                                            if alphas_official else float("nan"))
        results["avg_alpha_hat"] = (float(np.mean(alpha_hats))
                                    if alpha_hats else float("nan"))
        results["tail_fraction"] = tail_fraction

        return results
