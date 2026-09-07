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

CAVEAT on the self-similarity (Holder) exponent S: it is NOT identified at our
sample budget. Switching eps between the paper's 5e-3 and the released
notebook's own DEFAULT_EPS of 5e-2 gives a cross-model Spearman of 0.143 -- the
model ORDERING it produces is essentially a function of that free parameter, not
of the models (adversarial review, 2026-09-03). eps is recorded as
``_meta_holder_eps`` so the value behind any S is visible. Do not use S as a
cross-model feature without re-establishing its stability. J is the clean one:
all confounds null, Spearman 1.000 across tau grids, and 1.6% from the paper's
Wikipedia value of 0.52.

Headline finding: language is self-similar and long-range dependent
(H ~ 0.70), and SMALL cross-model differences in H improve on
bits-per-byte for predicting downstream performance — which makes H a
mandatory comparison baseline for capability-prediction studies.

PROTOCOL (2026-09-03): follows the paper. Consecutive corpus rows are joined
into contiguous documents of >= 4K tokens, the first min(2K, model context)
tokens of each are scored in ONE forward pass, a 128-token burn-in prefix is
dropped, and the three parameters are estimated PER DOCUMENT and averaged.

Until 2026-09-03 the per-token NLL streams of ~128-token corpus rows were
concatenated end-to-end and treated as a single process. That injects a
discontinuity every ~128 tokens into a statistic whose entire purpose is to
measure long-range dependence, and plants a periodic artifact at exactly the
document length. The legacy values are still emitted as ``*_concat_legacy``
so the size of the change is visible.

Estimator parity is pinned against ``nolds`` (hurst_rs) and analytic anchors
(white noise H ~ 0.5; J = 0.5 for iid).
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


# Minimum lag/window for every scaling fit.
#
# The short-lag end of a log-log scaling fit is dominated by discretisation and
# tokenizer artifacts rather than by the fractal regime, which is why the
# pooled path floors its windows at this value. The PER-SERIES estimators below
# did not get that floor: `_rs_hurst` carried min_window = 8 and both
# `_joseph_exponent` and `_holder_exponent` started their tau grid at
# logspace(0, ...) = 1. So `hurst_exponent`, `joseph_exponent` and
# `self_similarity_exponent` -- all exported keys -- were still fitting the
# regime the floor exists to exclude (adversarial review, 2026-09-03).
_TAU_MIN = 25


def _min_lag(N: int, upper: int) -> int:
    """The tau floor, backed off CONTINUOUSLY for short streams.

    A log-log scaling fit needs the floor to sit well below `upper`, or there
    is no range left to fit. The first version of this back-off switched
    abruptly -- 25 at upper = 26 and 8 at upper = 25 -- which is a CLIFF, not a
    back-off: per-document Joseph jumped 0.5893 -> 0.4387 between N = 207 and
    N = 208 on identical fGn(0.7), three times the published cross-model range,
    and `_rs_hurst` returned NaN for N in exactly [104, 107] because the floor
    left it too few distinct windows (adversarial review, 2026-09-04).

    Requiring floor <= upper/4 keeps at least a factor-4 span for the fit and
    makes the floor vary smoothly with N. At the shipped protocol (N = 895) it
    is 25 for every estimator, so nothing about the shipped numbers changes.
    """
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
        # 5e-2 is the released notebook's own DEFAULT_EPS, whose docstring warns
        # "if eps is too small, probability estimates will be unreliable".
        # At 5e-3 the counts are single-digit Poisson (one 895-token document
        # gave 1/4/2/1/0 pairs at tau = 1/4/16/64/256), and the estimate is
        # noise: S = 0.313 +/- 0.265 against the paper's 0.59 +/- 0.08. At
        # 5e-2 the same data gives 0.606 +/- 0.124 (adversarial review,
        # 2026-09-03).
        eps = float(self.config.get("holder_eps", 5e-2))
        # ...and a config can still put it back. `beyond_benchmarks`'
        # `configs/tasks.yaml` set `holder_eps: 0.005`, and user overrides win
        # in `resolve_task_config`, so the fix above was present and its effect
        # absent -- the "fix present, effect absent" pattern with the CONFIG as
        # the vector. End-to-end on gpt2 (red-team review, 2026-09-05),
        # package default vs study config:
        #
        #   self_similarity_exponent (headline, pooled)  0.569557 -> 0.043773  (-92.3%)
        #   self_similarity_exponent_perdoc             0.667857 -> 0.470090  (-29.6%)
        #   self_similarity_exponent_std                0.668682 -> 0.057779  (-91.4%)
        #   hurst_exponent / joseph_exponent            unchanged
        #
        # At 0.043773 the pooled estimator sits an order of magnitude below its
        # own per-document mean and nowhere near the paper's 0.59 +/- 0.08.
        # `_meta_holder_eps` recorded the value, but a number in a provenance
        # field is not a warning, so this shipped silently. It no longer does.
        _eps_ok = eps >= 5e-2
        if not _eps_ok:
            logger.warning(
                f"  dynamics_fractal: holder_eps={eps:g} is below the released "
                f"notebook's DEFAULT_EPS of 5e-2. The notebook's own docstring "
                f"warns that probability estimates are unreliable there, and "
                f"BLME measures the pooled self-similarity exponent collapsing "
                f"to ~0.04 (vs ~0.57 at 5e-2). `self_similarity_exponent` from "
                f"this run should NOT be reported.")
        # Paper protocol (Alabdulmohsin et al. 2024 Sec. 3): documents longer
        # than 4K tokens, first 2K tokens of each, parameters estimated PER
        # DOCUMENT and averaged.
        min_doc_tokens = int(self.config.get("min_doc_tokens", 4096))
        use_tokens = int(self.config.get("use_tokens", 2048))
        # The first tokens of a document have no left context, so their NLL is
        # a large deterministic transient. Integrated into X_t it is a
        # near-deterministic ramp, which biases R/S upward (the range R is
        # inflated) and the Joseph exponent as well. The paper's 4K-token
        # documents make this a negligible edge effect; at our lengths it is
        # not, so we drop a fixed prefix.
        burn_in = int(self.config.get("burn_in", 128))

        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog. " * 8}
                       for _ in range(20)]

        device = next(model.parameters()).device
        ctx = (getattr(model.config, "max_position_embeddings", None)
               or getattr(model.config, "n_positions", None) or 1024)
        if not isinstance(ctx, int) or ctx > 8192:
            ctx = 8192
        # Never ask for more tokens than the model can attend over.
        seg_len = int(min(use_tokens, ctx))
        if seg_len <= burn_in + 128:
            burn_in = max(0, seg_len // 8)

        # ---- Build LONG documents ----
        # The eval corpus rows are ~128 tokens. Concatenating their NLL streams
        # end-to-end (the pre-2026-09-03 behaviour) injects a discontinuity
        # every ~128 tokens into a process whose whole purpose is to measure
        # long-range dependence, and puts a periodic artifact at exactly the
        # document length. Here consecutive rows are joined into contiguous
        # TEXT first, so the model actually attends across the join and the
        # NLL stream is a single coherent process.
        texts, buf, buf_tok = [], [], 0
        for i, sample in enumerate(dataset):
            if len(texts) >= num_samples:
                break
            t = sample["text"] if isinstance(sample, dict) else str(sample)
            if not t:
                continue
            buf.append(t)
            buf_tok += len(tokenizer(t)["input_ids"])
            if buf_tok >= min_doc_tokens:
                texts.append("\n\n".join(buf))
                buf, buf_tok = [], 0
        if not texts and buf:
            # Corpus too short to make even one 4K document: use what we have
            # rather than returning nothing, and record it.
            texts.append("\n\n".join(buf))

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

        # POOLED estimates are the headline: the authors' released
        # `get_hurst_exponent` accumulates R/S over all documents per scale and
        # fits a SINGLE power law, rather than averaging per-document fits.
        # The difference is real (gpt2 on pg19: 0.6623 pooled vs 0.6206
        # per-document), so the pooled value is what to compare against the
        # paper. Per-document means are retained as `*_perdoc`.
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
            "_meta_segment_len": int(seg_len),
            "_meta_burn_in": int(burn_in),
            "protocol": "per_document_paper_protocol",
            "_meta_holder_eps": float(eps),
            # A machine-readable form of the warning above, so an analysis can
            # gate on it rather than reading logs.
            "_meta_holder_eps_at_or_above_reference": bool(_eps_ok),
            "_meta_tau_min": int(_TAU_MIN),
        }
        # Legacy concatenated-stream values, kept so the change in the headline
        # numbers is auditable rather than silent.
        legacy = _fractal_params(all_nll, eps=eps)
        for k in ("hurst_exponent", "joseph_exponent",
                  "self_similarity_exponent"):
            results[f"{k}_concat_legacy"] = legacy[k]
        return results



def _pooled_fractal_params(streams, eps: float = 5e-2) -> dict:
    """Pooled fractal parameters: accumulate the statistic across documents at
    each scale, then fit ONE power law.

    This is what the authors' released `fractal_patterns_in_language.ipynb`
    does (`get_hurst_exponent` takes the full 2-D document array and averages
    R/S over documents per scale before fitting). Averaging per-document fits
    instead is a different estimator, and measurably so: gpt2 on pg19 gives
    H = 0.6623 pooled vs 0.6206 per-document.

    Each document is normalized to zero mean and unit variance on its OWN
    increments before integration -- pooling the raw NLL across documents
    would let cross-document mean shifts masquerade as long-range drift.
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

    # ---- Hurst: mean R/S over all documents at each window size ----
    # The R/S window grid must respect the same lower cutoff as the tau grids.
    # The 2026-09-03 tau_min fix was applied to the Joseph and Holder grids but
    # NOT here, so it landed on two of the three exponents and left Hurst fitted
    # over windows starting at 8 (adversarial review, batch B08). The reference
    # fits over the large-scale regime; small windows are dominated by
    # short-range structure the exponent is not meant to describe.
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

    # ---- Joseph: std of tau-increments, pooled over documents ----
    # tau grid starts at TAU_MIN, not 1. The released notebook fits over
    # tau in [25, 300]; including tau = 1..8 puts most of the fitted points in a
    # regime the reference excludes and biases both exponents. Restricting to
    # tau >= 25 moves the Joseph exponent 0.5417 -> 0.4972, against the
    # reference's 0.4945 (adversarial review, 2026-09-03).
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
