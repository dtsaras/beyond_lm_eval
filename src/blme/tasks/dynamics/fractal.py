"""Fractal parameters of the log-likelihood stream — Alabdulmohsin, Tran &
Dehghani, NeurIPS 2024 ("Fractal Patterns May Illuminate the Success of
Next-Token Prediction", arXiv:2402.01825; no official code release).

The paper converts text into a stochastic process using the model itself:
each token contributes an increment z_t = -log p(w_t | w_{<t}) (its code
length under the model); the increments are normalized to zero mean and unit
variance (x_t) and integrated into X_t = sum_{k<=t} x_k. Three fractal
parameters of that process:

  * Self-similarity (Holder) exponent S: fit p_eps(tau) ~ tau^{-S}, where
    p_eps(tau) is the probability mass of {|X_{t+tau} - X_t| <= eps}
    (paper uses eps = 5e-3 on the normalized process).
  * Hurst parameter H: classic rescaled-range analysis, R(n)/S(n) ~ n^H,
    over log-spaced window sizes n.
  * Joseph exponent J: sigma_tau ~ tau^J, the std of tau-increments of X.

Headline finding: language is self-similar and long-range dependent
(H ~ 0.70), and SMALL cross-model differences in H improve on
bits-per-byte for predicting downstream performance — which makes H a
mandatory comparison baseline for capability-prediction studies.

PROTOCOL NOTE (refined adaptation): the paper restricts to documents longer
than 4K tokens and uses the first 2K tokens of each. BLME computes the
parameters over the CONCATENATED per-token NLL stream of its fixed
evaluation corpus (~128-token documents); the estimators are identical, the
corpus protocol is not. Estimator parity is pinned against ``nolds``
(hurst_rs) and analytic anchors (white noise H ~ 0.5; J = 0.5 for iid).
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


def _rs_hurst(x: np.ndarray, min_window: int = 8) -> float:
    """Classic rescaled-range Hurst estimator: for log-spaced window sizes n,
    average R/S over disjoint windows and fit log(R/S) ~ H log(n).

    R = max-min of the cumulative deviation-from-window-mean; S = window std.
    """
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    if N < 4 * min_window:
        return float("nan")
    ns = np.unique(np.floor(np.logspace(
        np.log10(min_window), np.log10(N // 4), 12)).astype(int))
    log_n, log_rs = [], []
    for n in ns:
        if n < min_window:
            continue
        n_win = N // n
        rs_vals = []
        for w in range(n_win):
            seg = x[w * n:(w + 1) * n]
            dev = seg - seg.mean()
            Y = np.cumsum(dev)
            R = Y.max() - Y.min()
            S = seg.std()
            if S > 0 and R > 0:
                rs_vals.append(R / S)
        if rs_vals:
            log_n.append(np.log(n))
            log_rs.append(np.log(np.mean(rs_vals)))
    if len(log_n) < 3:
        return float("nan")
    return float(np.polyfit(log_n, log_rs, 1)[0])


def _joseph_exponent(X: np.ndarray, taus: np.ndarray = None) -> float:
    """J from sigma(X_{t+tau} - X_t) ~ tau^J (X = integrated process)."""
    X = np.asarray(X, dtype=np.float64)
    N = X.size
    if N < 64:
        return float("nan")
    if taus is None:
        taus = np.unique(np.floor(np.logspace(0, np.log10(N // 8), 10)).astype(int))
    log_t, log_s = [], []
    for tau in taus:
        if tau < 1 or tau >= N:
            continue
        diffs = X[tau:] - X[:-tau]
        s = diffs.std()
        if s > 0:
            log_t.append(np.log(tau))
            log_s.append(np.log(s))
    if len(log_t) < 3:
        return float("nan")
    return float(np.polyfit(log_t, log_s, 1)[0])


def _holder_exponent(X: np.ndarray, eps: float = 5e-3,
                     taus: np.ndarray = None) -> float:
    """S from p_eps(tau) ~ tau^{-S}, p_eps(tau) = P(|X_{t+tau} - X_t| <= eps).

    eps = 5e-3 follows the paper (on the zero-mean unit-variance increment
    normalization). Taus with zero hits are skipped.
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.size
    if N < 64:
        return float("nan")
    if taus is None:
        taus = np.unique(np.floor(np.logspace(0, np.log10(N // 8), 10)).astype(int))
    log_t, log_p = [], []
    for tau in taus:
        if tau < 1 or tau >= N:
            continue
        p = float(np.mean(np.abs(X[tau:] - X[:-tau]) <= eps))
        if p > 0:
            log_t.append(np.log(tau))
            log_p.append(np.log(p))
    if len(log_t) < 3:
        return float("nan")
    return float(-np.polyfit(log_t, log_p, 1)[0])


def _fractal_params(nll_stream, eps: float = 5e-3) -> dict:
    """All three fractal parameters from a raw per-token NLL sequence."""
    z = np.asarray(nll_stream, dtype=np.float64)
    z = z[np.isfinite(z)]
    if z.size < 64 or z.std() == 0:
        return {"hurst_exponent": float("nan"),
                "joseph_exponent": float("nan"),
                "self_similarity_exponent": float("nan"),
                "n_tokens_in_stream": int(z.size)}
    x = (z - z.mean()) / z.std()          # normalized increments
    X = np.cumsum(x)                      # integrated process
    return {
        "hurst_exponent": _rs_hurst(x),
        "joseph_exponent": _joseph_exponent(X),
        "self_similarity_exponent": _holder_exponent(X, eps=eps),
        "n_tokens_in_stream": int(z.size),
    }


@register_task("dynamics_fractal")
class FractalLikelihoodTask(DiagnosticTask):
    """Hurst / Joseph / self-similarity exponents of the model's per-token
    NLL stream over the evaluation corpus."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Fractal Likelihood Analysis (Hurst)...")
        num_samples = self.config.get("num_samples", 100)
        eps = float(self.config.get("holder_eps", 5e-3))

        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog. " * 8}
                       for _ in range(20)]

        device = next(model.parameters()).device
        nlls = []
        with torch.no_grad():
            for i, sample in enumerate(dataset):
                if i >= num_samples:
                    break
                text = sample["text"] if isinstance(sample, dict) else str(sample)
                enc = tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=512)
                ids = enc["input_ids"].to(device)
                if ids.shape[1] < 2:
                    continue
                logits = model(ids).logits[0]
                logp = F.log_softmax(logits.float(), dim=-1)
                targets = ids[0, 1:]
                token_nll = -logp[:-1].gather(1, targets.unsqueeze(1)).squeeze(1)
                nlls.extend(token_nll.cpu().numpy().tolist())

        results = _fractal_params(nlls, eps=eps)
        results["protocol"] = "concatenated_eval_corpus_stream"
        return results
