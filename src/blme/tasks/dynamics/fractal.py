"""Fractal parameters of the log-likelihood stream — Alabdulmohsin, Tran &
Dehghani, NeurIPS 2024 ("Fractal Patterns May Illuminate the Success of
Next-Token Prediction", arXiv:2402.01825). Reference implementation:
google-research/fractals_language, fractal_patterns_in_language.ipynb.

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

CAVEAT on the self-similarity (Holder) exponent S: it is weakly identified at
small budgets; switching eps between the paper's 5e-3 and the notebook's 5e-2
reordered models on spliced passages (cross-model Spearman 0.143). The paper's
5e-3 is the default; eps is recorded as ``_meta_holder_eps``.

Headline finding: language is self-similar and long-range dependent
(H ~ 0.70), and SMALL cross-model differences in H improve on
bits-per-byte for predicting downstream performance — which makes H a
mandatory comparison baseline for capability-prediction studies.

PROTOCOL: natural documents of >= 4K tokens (whole WikiText-103 articles,
``cache.load_long_documents``), the first min(use_tokens, model context)
tokens of each scored in ONE forward pass, a 128-token burn-in prefix dropped,
and the three parameters fitted on the pooled per-document statistics
(per-document means as ``*_perdoc``). The concatenated-stream values are
``*_concat_legacy``.

Estimator parity is pinned against ``nolds`` (hurst_rs) and analytic anchors
(white noise H ~ 0.5; J = 0.5 for iid).
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F

from ...cache import load_long_documents
from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


# Minimum lag/window for every scaling fit (the notebook fits tau >= 25): the
# short-lag end is dominated by discretisation and tokenizer artifacts.
_TAU_MIN = 25

# Alabdulmohsin et al. 2024 Sec. 2.2. Two notebook defects are not reproduced
# (Q2): cell 27 scores log_softmax(logits)[i, ts[i]], the token AT i under the
# prediction for i+1 (BLME shifts targets by one), and cell 46's
# get_joseph_exponent never resets its accumulator across scales (BLME does).
PAPER_HOLDER_EPS = 5e-3


def _min_lag(N: int, upper: int) -> int:
    """The tau floor, backed off continuously for short streams: floor <=
    upper/4 keeps a factor-4 span for the fit with no cliff in N."""
    return int(min(_TAU_MIN, max(2, upper // 4)))


def _rs_hurst(x: np.ndarray, min_window: int = None) -> float:
    """Classic rescaled-range Hurst estimator: for log-spaced window sizes n,
    average R/S over disjoint windows and fit log(R/S) ~ H log(n).

    R = max-min of the cumulative deviation-from-window-mean; S = window std.
    """
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    if min_window is None:
        min_window = _min_lag(N, max(2, N // 4))
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
        _lo = _min_lag(N, max(2, N // 8))
        taus = np.unique(np.floor(np.logspace(
            np.log10(_lo), np.log10(max(_lo * 2, N // 8)), 10)).astype(int))
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
        _lo = _min_lag(N, max(2, N // 8))
        taus = np.unique(np.floor(np.logspace(
            np.log10(_lo), np.log10(max(_lo * 2, N // 8)), 10)).astype(int))
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
        # The paper's eps (Sec. 2.2: eps = 5e-3, S robust over [1e-3, 1e-2]); the
        # notebook's DEFAULT_EPS of 5e-2 is labelled there as differing from the
        # paper (Q2: the paper's definition wins).
        eps = float(self.config.get("holder_eps", PAPER_HOLDER_EPS))
        # Paper protocol (Sec. 2.1): documents longer than 4K tokens, first 2K
        # tokens of each.
        min_doc_tokens = int(self.config.get("min_doc_tokens", 4096))
        use_tokens = int(self.config.get("use_tokens", 2048))
        # The first tokens' NLL is a deterministic transient (no left context);
        # the notebook drops the first 128 scored tokens.
        burn_in = int(self.config.get("burn_in", 128))

        # Natural long documents (paper Sec. 2.1: documents > 4K tokens), whole
        # WikiText-103 articles in a fixed nested order. The shared corpus is
        # passages, and joining passages from different articles would measure
        # the splices; documents="dataset" scores each given row as a document.
        source = self.config.get("documents", "long_documents")
        doc_split = self.config.get("doc_split", "train")
        if source == "dataset":
            if not dataset:
                return {"error": "documents='dataset' needs a dataset"}
            texts = [s["text"] if isinstance(s, dict) else str(s)
                     for s in list(dataset)[:num_samples]]
            texts = [t for t in texts if t]
            source_name = "dataset"
        else:
            try:
                docs = load_long_documents(num_samples, min_tokens=min_doc_tokens,
                                           tokenizer=tokenizer, split=doc_split)
            except (ValueError, RuntimeError) as e:
                return {"error": f"dynamics_fractal: long documents unavailable: {e}"}
            texts = [d["text"] for d in docs]
            source_name = f"wikitext_document_level:{doc_split}"

        device = next(model.parameters()).device
        ctx = (getattr(model.config, "max_position_embeddings", None)
               or getattr(model.config, "n_positions", None) or 1024)
        if not isinstance(ctx, int) or ctx > 8192:
            ctx = 8192
        # Never ask for more tokens than the model can attend over.
        seg_len = int(min(use_tokens, ctx))
        if seg_len <= burn_in + 128:
            burn_in = max(0, seg_len // 8)

        per_doc, all_nll, per_doc_streams = [], [], []
        with torch.no_grad():
            for text in texts:
                enc = tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=seg_len)
                ids = enc["input_ids"].to(device)
                if ids.shape[1] < burn_in + 64:
                    continue
                logits = model(ids).logits[0]
                logp = F.log_softmax(logits.float(), dim=-1)
                targets = ids[0, 1:]
                token_nll = -logp[:-1].gather(
                    1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()
                all_nll.extend(token_nll.tolist())
                trimmed = token_nll[burn_in:]
                if trimmed.size >= 64:
                    per_doc.append(_fractal_params(trimmed, eps=eps))
                    per_doc_streams.append(trimmed)

        def _mean(key):
            vals = [d[key] for d in per_doc
                    if d.get(key) is not None and np.isfinite(d.get(key, np.nan))]
            return float(np.mean(vals)) if vals else float("nan")

        def _std(key):
            vals = [d[key] for d in per_doc
                    if d.get(key) is not None and np.isfinite(d.get(key, np.nan))]
            return float(np.std(vals)) if len(vals) > 1 else float("nan")

        # Pooled estimates are the headline, as the notebook accumulates the
        # statistic over documents per scale and fits one power law.
        pooled = _pooled_fractal_params(per_doc_streams, eps=eps)

        results = {
            "hurst_exponent": pooled["hurst_exponent"],
            "joseph_exponent": pooled["joseph_exponent"],
            "self_similarity_exponent": pooled["self_similarity_exponent"],
            "hurst_exponent_perdoc": _mean("hurst_exponent"),
            "joseph_exponent_perdoc": _mean("joseph_exponent"),
            "self_similarity_exponent_perdoc": _mean("self_similarity_exponent"),
            "hurst_exponent_std": _std("hurst_exponent"),
            "joseph_exponent_std": _std("joseph_exponent"),
            "self_similarity_exponent_std": _std("self_similarity_exponent"),
            "n_tokens_in_stream": int(sum(d["n_tokens_in_stream"]
                                          for d in per_doc)),
            "_meta_n_documents": len(per_doc),
            "_meta_document_source": source_name,
            "_meta_min_doc_tokens": int(min_doc_tokens),
            "_meta_segment_len": int(seg_len),
            "_meta_burn_in": int(burn_in),
            "protocol": "per_document_paper_protocol",
            "_meta_holder_eps": float(eps),
            "_meta_holder_eps_is_paper_value": bool(eps == PAPER_HOLDER_EPS),
            "_meta_tau_min": int(_TAU_MIN),
        }
        # Legacy concatenated-stream values.
        legacy = _fractal_params(all_nll, eps=eps)
        for k in ("hurst_exponent", "joseph_exponent",
                  "self_similarity_exponent"):
            results[f"{k}_concat_legacy"] = legacy[k]
        return results



def _pooled_fractal_params(streams, eps: float = PAPER_HOLDER_EPS) -> dict:
    """Pooled fractal parameters: accumulate the statistic across documents at
    each scale, then fit ONE power law (the notebook's get_hurst_exponent).

    Each document is standardised on its own increments before integration,
    so cross-document mean shifts cannot masquerade as long-range drift.
    """
    xs, Xs = [], []
    for z in streams:
        z = np.asarray(z, dtype=np.float64)
        z = z[np.isfinite(z)]
        if z.size < 64 or z.std() == 0:
            continue
        x = (z - z.mean()) / z.std()
        xs.append(x)
        Xs.append(np.cumsum(x))
    nan = {"hurst_exponent": float("nan"), "joseph_exponent": float("nan"),
           "self_similarity_exponent": float("nan")}
    if not xs:
        return nan

    n_min = min(x.size for x in xs)
    if n_min < 64:
        return nan

    # ---- Hurst: mean R/S over all documents at each window size (>= tau floor) ----
    w_min = max(8, min(_TAU_MIN, n_min // 8))
    windows = np.unique(np.floor(
        np.logspace(np.log10(w_min), np.log10(max(w_min * 2, n_min // 2)),
                    12)).astype(int))
    log_n, log_rs = [], []
    for n_w in windows:
        vals = []
        for x in xs:
            n_chunks = x.size // n_w
            for c in range(n_chunks):
                seg = x[c * n_w:(c + 1) * n_w]
                sd = seg.std()
                if sd <= 0:
                    continue
                dev = np.cumsum(seg - seg.mean())
                vals.append((dev.max() - dev.min()) / sd)
        if vals:
            log_n.append(np.log(n_w))
            log_rs.append(np.log(np.mean(vals)))
    hurst = (float(np.polyfit(log_n, log_rs, 1)[0])
             if len(log_n) >= 3 else float("nan"))

    # ---- Joseph: std of tau-increments, pooled over documents (tau >= floor) ----
    tau_min = max(1, min(_TAU_MIN, n_min // 8))
    taus = np.unique(np.floor(
        np.logspace(np.log10(tau_min), np.log10(max(tau_min + 1, n_min // 4)),
                    10)).astype(int))
    log_t, log_s = [], []
    for tau in taus:
        diffs = np.concatenate([X[tau:] - X[:-tau] for X in Xs if X.size > tau]) \
            if any(X.size > tau for X in Xs) else np.array([])
        if diffs.size > 1 and diffs.std() > 0:
            log_t.append(np.log(tau))
            log_s.append(np.log(diffs.std()))
    joseph = (float(np.polyfit(log_t, log_s, 1)[0])
              if len(log_t) >= 3 else float("nan"))

    # ---- Self-similarity: pooled p_eps(tau) over documents ----
    log_t2, log_p = [], []
    for tau in taus:
        hits = tot = 0
        for X in Xs:
            if X.size <= tau:
                continue
            d = np.abs(X[tau:] - X[:-tau])
            hits += int(np.count_nonzero(d <= eps))
            tot += d.size
        if tot > 0 and hits > 0:
            log_t2.append(np.log(tau))
            log_p.append(np.log(hits / tot))
    selfsim = (float(-np.polyfit(log_t2, log_p, 1)[0])
               if len(log_t2) >= 3 else float("nan"))

    return {"hurst_exponent": hurst, "joseph_exponent": joseph,
            "self_similarity_exponent": selfsim}
