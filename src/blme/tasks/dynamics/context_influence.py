"""``dynamics_context_influence``: how far back in its context a model's
prediction depends on a single token.

For a window x, a replaced position s and a lag k >= 1, the prediction made at
position t = s + k - 1 (the next-token distribution after reading x_<=t) is
compared with the same prediction when x_s alone is replaced by a token u drawn
from the corpus unigram distribution:

    I(k) = E_x E_s E_u JS( p(. | x_<=t), p(. | x_<=t with x_s := u) ),

in nats, bounded by log 2 per term. Every lag after s comes from the same
perturbed forward, so a window with R replacements costs 1 + R forwards. The
lags run up to K = window - replace_hi + 1, the largest lag observed for every
replacement, so each lag averages the same number of terms.

Reported: the decay exponent beta of I(k) ~ k^-beta (least squares in log-log
over log-spaced lag bins from lag 3 to K), the far mass sum_{k=9..K} I(k)
(nats), and the effective horizon exp(sum_k I(k) log k / sum_k I(k)) converted
to bytes with the windows' mean bytes per token. The binned profile, the
per-lag means, the lag bins in tokens and in bytes and window-bootstrap
standard errors are ``_meta_``.

The window is 1024 tokens for every model: the largest context every model of
the companion study supports (GPT-2's). Position 0 and massive-activation rows
are never replaced and their predictions are not scored (TOKEN_POLICY
``exclude_sinks``, the corpus-wide rule). Every random stream (each window's
start in its document, the replaced positions, the replacement tokens and the
bootstrap) comes from the ``seed`` config key.

Grounding: Khandelwal et al. 2018 (arXiv:1805.04623) perturb context at a
distance and measure the change in loss; Lin & Tegmark 2017
(arXiv:1606.06737) show the mutual information of natural text decays as a
power law with distance, and is zero beyond lag n for an n-gram source; this
task is the interventional counterpart for a model's own predictions.
"""

import hashlib
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ...cache import item_set, load_long_documents
from ...registry import register_task
from ...tasks.base import DiagnosticTask
from .._precision import effective_dtype_meta, fp32_compute
from .._tokens import middle_depth_layers, sink_keep_masks, sink_meta, strip_trailing_special
from ..common import get_max_positions

logger = logging.getLogger("blme")

DEFAULT_WINDOW = 1024
DEFAULT_NUM_SAMPLES = 96
DEFAULT_REPLACEMENTS = 3
DEFAULT_REPLACE_LO = 16
DEFAULT_REPLACE_HI = 64
DEFAULT_MIN_DOC_WORDS = 2048
DEFAULT_CHUNK_ELEMENTS = 2 ** 22
DEFAULT_BOOTSTRAP = 200
FAR_LAG = 9
FIT_FROM_LAG = 3
LOG2 = math.log(2.0)


def lag_bins(k_max: int) -> List[Tuple[int, int]]:
    """Inclusive lag bins [1], [2], [3, 4], [5, 8], ..., the last cut at k_max."""
    bins, lo = [], 1
    while lo <= k_max:
        hi = 1 if lo == 1 else min(2 * (lo - 1), k_max)
        bins.append((lo, hi))
        lo = hi + 1
    return bins


def js_rows(lp: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence (nats) between rows of two float64
    log-probability tensors.

    log m is formed as max + log1p(expm1(min - max) / 2), so identical rows give
    exactly 0 and disjoint supports exactly log 2; terms with zero probability
    contribute 0.
    """
    hi = torch.maximum(lp, lq)
    lo = torch.minimum(lp, lq)
    lm = hi + torch.log1p(torch.expm1(lo - hi) / 2.0)
    p, q = lp.exp(), lq.exp()
    zero = torch.zeros((), dtype=lp.dtype, device=lp.device)
    a = torch.where(p > 0, p * (lp - lm), zero).sum(-1)
    b = torch.where(q > 0, q * (lq - lm), zero).sum(-1)
    return 0.5 * (a + b)


def influence_summary(per_lag: np.ndarray, bytes_per_token: float) -> Dict[str, Any]:
    """The reported statistics of a per-lag influence profile I(1..K)."""
    I = np.asarray(per_lag, dtype=np.float64)
    K = int(I.size)
    lags = np.arange(1, K + 1, dtype=np.float64)
    ok = np.isfinite(I)
    bins = lag_bins(K)
    means, centers = [], []
    for lo, hi in bins:
        seg = I[lo - 1:hi]
        seg = seg[np.isfinite(seg)]
        means.append(float(seg.mean()) if seg.size else float("nan"))
        centers.append(float(np.exp(np.mean(np.log(np.arange(lo, hi + 1, dtype=np.float64))))))
    means, centers = np.array(means), np.array(centers)
    fit = np.array([lo >= FIT_FROM_LAG for lo, _ in bins]) & np.isfinite(means) & (means > 0)
    beta = r2 = float("nan")
    if fit.sum() >= 3:
        x, y = np.log(centers[fit]), np.log(means[fit])
        slope, icpt = np.polyfit(x, y, 1)
        beta = float(-slope)
        resid = y - (slope * x + icpt)
        ss = float(((y - y.mean()) ** 2).sum())
        r2 = float(1.0 - (resid ** 2).sum() / ss) if ss > 0 else float("nan")
    total = float(np.sum(I[ok]))
    far = float(np.sum(I[ok & (lags >= FAR_LAG)]))
    pos = ok & (I > 0)
    horizon = (float(np.exp(np.sum(I[pos] * np.log(lags[pos])) / np.sum(I[pos])))
               if pos.any() else float("nan"))
    return {
        "influence_decay_exponent": beta,
        "influence_far_mass": far,
        "influence_horizon_bytes": horizon * bytes_per_token,
        "horizon_tokens": horizon,
        "total_mass": total,
        "far_share": far / total if total > 0 else float("nan"),
        "fit_r2": r2,
        "n_fit_bins": int(fit.sum()),
        "bin_means": means,
        "bin_centers": centers,
        "bins": bins,
    }


def _window_rng(seed: int, i: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), 0, int(i)])


def _bootstrap_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), 1])


@register_task("dynamics_context_influence")
class ContextInfluenceTask(DiagnosticTask):
    """Interventional context-influence spectrum: JS change of the next-token
    distribution when one earlier token is resampled, by lag."""

    # Position 0 and massive-activation tokens are never replaced and their
    # predictions are not scored (one mask per token, the corpus-wide rule).
    TOKEN_POLICY = "exclude_sinks"
    SOURCES = {
        "paper": ("arXiv:1805.04623 (Khandelwal et al. 2018, context perturbation at a "
                  "distance); arXiv:1606.06737 (Lin & Tegmark 2017, power-law decay of "
                  "mutual information in text)"),
        "code": None,
        "llm_precedent": ("arXiv:2212.14815 (Cifka & Liutkus 2023, context-length probing of "
                          "pretrained transformer LMs)"),
    }
    ADAPTATION_NOTE = (
        "BLME construct after Khandelwal et al. (2018), who perturb an LSTM's context beyond a "
        "distance and measure the change in loss: here one token at a time is resampled from the "
        "corpus unigram distribution and the change of the full next-token distribution is "
        "measured as a Jensen-Shannon divergence at every later position, which needs no labels "
        "and is bounded by log 2. Windows are 1024 tokens for every model (GPT-2's context, the "
        "largest every study model supports), cut from WikiText-103 train articles of at least "
        "2048 words at a seeded word offset, so every model reads the same text. Three positions "
        "in [16, 64) per window are replaced, sinks excepted, and lags run to 961, the largest "
        "lag every replacement observes. beta is fitted over log-spaced bins from lag 3, the far "
        "mass sums lags 9 and beyond, and the horizon is the influence-weighted geometric mean "
        "lag in bytes. The forwards run in fp32 arithmetic at every size because the long-lag "
        "values sit near bf16 rounding, and the divergences are taken in float64."
    )
    HEADLINE_KEYS = ("influence_decay_exponent", "influence_far_mass", "influence_horizon_bytes")
    # A function-space quantity, but every key depends on the tokenizer (lags
    # in tokens) and the sink mask on the middle depths; no transform is claimed.
    SIZE_INVARIANT = {}

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running context influence...")
        cfg = self.config
        num_samples = int(cfg.get("num_samples", DEFAULT_NUM_SAMPLES))
        window = int(cfg.get("window", DEFAULT_WINDOW))
        R = int(cfg.get("n_replacements", DEFAULT_REPLACEMENTS))
        lo = int(cfg.get("replace_lo", DEFAULT_REPLACE_LO))
        hi = int(cfg.get("replace_hi", DEFAULT_REPLACE_HI))
        min_doc_words = int(cfg.get("min_doc_words", DEFAULT_MIN_DOC_WORDS))
        doc_split = str(cfg.get("doc_split", "train"))
        source = str(cfg.get("documents", "long_documents"))
        compute = str(cfg.get("compute_dtype", "float32"))
        chunk_elements = int(cfg.get("chunk_elements", DEFAULT_CHUNK_ELEMENTS))
        n_boot = int(cfg.get("n_bootstrap", DEFAULT_BOOTSTRAP))
        seed = int(cfg.get("seed", 42))
        policy = self.TOKEN_POLICY
        if compute not in ("float32", "model"):
            return {"error": f"compute_dtype must be 'float32' or 'model', got {compute!r}"}

        ctx = get_max_positions(model, window)
        W = int(min(window, ctx))
        lo = max(lo, 1)
        if not (1 <= lo < hi <= W) or R < 1 or hi - lo < R:
            return {"error": f"bad replacement range [{lo}, {hi}) for {R} replacements in {W} tokens"}
        K = W - hi + 1
        if K < FAR_LAG + 2:
            return {"error": f"window {W} leaves lags up to {K} only"}

        # ── documents and windows ──
        if source == "dataset":
            if not dataset:
                return {"error": "documents='dataset' needs a dataset"}
            texts = [s["text"] if isinstance(s, dict) else str(s) for s in list(dataset)[:num_samples]]
            source_name = "dataset"
        else:
            try:
                docs = load_long_documents(num_samples, min_words=min_doc_words, split=doc_split)
            except (ValueError, RuntimeError) as e:
                return {"error": f"dynamics_context_influence: long documents unavailable: {e}"}
            texts = [d["text"] for d in docs]
            source_name = f"wikitext_document_level:{doc_split}"
            if item_set():
                source_name += f":item{item_set()}"

        special = {int(i) for i in (getattr(tokenizer, "all_special_ids", None) or [])}
        wins: List[Dict[str, Any]] = []
        n_short = 0
        for i, text in enumerate(texts):
            rng = _window_rng(seed, i)
            starts = [m.start() for m in re.finditer(r"\S+", text)]
            room = len(starts) - W
            j = int(rng.integers(0, room + 1)) if room > 0 else 0
            # 4 W words hold W tokens for any tokenizer (at least one token per
            # word); the cut is far past the W tokens kept.
            end = j + 4 * W
            sub = (text[starts[j]:starts[end]] if end < len(starts) else text[starts[j]:]) \
                if starts else text
            enc = tokenizer(sub, return_tensors="pt", truncation=True, max_length=W)
            enc, _ = strip_trailing_special(dict(enc), tokenizer)
            ids = enc["input_ids"][0]
            if int(ids.numel()) < W:
                n_short += 1
                continue
            content = [int(x) for x in ids.tolist() if int(x) not in special]
            nbytes = len(tokenizer.decode(content, clean_up_tokenization_spaces=False).encode("utf-8"))
            wins.append({"i": i, "rng": rng, "offset": j, "ids": ids,
                         "bytes_per_token": nbytes / max(1, len(content))})
        if not wins:
            return {"error": f"no document yields a {W}-token window ({n_short} too short)"}

        # Corpus unigram over the windows' content tokens.
        allids = torch.cat([w["ids"] for w in wins]).numpy()
        V_tok = int(max(allids.max() + 1, len(tokenizer)))
        counts = np.bincount(allids, minlength=V_tok).astype(np.float64)
        for sid in special:
            if 0 <= sid < counts.size:
                counts[sid] = 0.0
        cdf = np.cumsum(counts / counts.sum())
        cdf[-1] = 1.0

        def _draw(rng, avoid):
            for _ in range(10000):
                u = int(np.searchsorted(cdf, rng.random(), side="right"))
                u = min(u, cdf.size - 1)
                if u != avoid and counts[u] > 0:
                    return u
            raise RuntimeError("could not draw a replacement token")

        device = next(model.parameters()).device
        model.eval()
        js = np.full((len(wins), R, K), np.nan)
        s_drawn = np.zeros((len(wins), R), dtype=np.int64)
        pre_max = 0.0
        norms: Optional[Dict[int, List[torch.Tensor]]] = None
        mid: Optional[List[int]] = None
        digest = hashlib.sha256()
        with torch.no_grad(), fp32_compute(model, enabled=(compute == "float32")) as prec:
            for wi, w in enumerate(wins):
                rng, ids = w["rng"], w["ids"]
                pos = rng.choice(np.arange(lo, hi), size=R, replace=False)
                us = [_draw(rng, int(ids[s])) for s in pos]
                s_drawn[wi] = pos
                digest.update(np.array([w["i"], w["offset"], *pos, *us], dtype=np.int64).tobytes())
                x = ids[None].to(device)
                out = model(input_ids=x, output_hidden_states=True)
                # post-final-norm: not read. Only middle depths (relative 0.1-0.9)
                # of the block outputs feed the sink norms.
                blocks = out.hidden_states[1:]
                if mid is None:
                    mid = middle_depth_layers(dict.fromkeys(range(len(blocks))))
                    norms = {k: [] for k in mid}
                for k in mid:
                    norms[k].append(torch.linalg.vector_norm(
                        blocks[k][0].to("cpu", torch.float64), dim=1).unsqueeze(1))
                clean = out.logits[0].detach()
                del out, blocks
                V = int(clean.shape[-1])
                step = max(1, chunk_elements // V)
                for r, (s, u) in enumerate(zip(pos, us)):
                    s = int(s)
                    xp = x.clone()
                    xp[0, s] = u
                    pert = model(input_ids=xp).logits[0].detach()
                    # Causality check: positions before s cannot move.
                    a = max(0, s - 8)
                    d0 = js_rows(torch.log_softmax(clean[a:s].double(), -1),
                                 torch.log_softmax(pert[a:s].double(), -1))
                    if d0.numel():
                        pre_max = max(pre_max, float(d0.abs().max()))
                    vals = []
                    for t0 in range(s, s + K, step):
                        t1 = min(t0 + step, s + K)
                        vals.append(js_rows(torch.log_softmax(clean[t0:t1].double(), -1),
                                            torch.log_softmax(pert[t0:t1].double(), -1)))
                    js[wi, r] = torch.cat(vals).cpu().numpy()
                    del pert

        # ── sinks: drop replacements at sink positions and sink predictions ──
        n_rep_sink = 0
        sink_info = {"n_massive_dropped": 0, "n_rows_pos_gt0": 0, "n_depths": 0}
        if policy in ("exclude_sinks", "exclude_pos0") and mid:
            if policy == "exclude_sinks":
                masks, sink_info = sink_keep_masks([norms[k] for k in mid])
            else:
                masks = [torch.arange(W) != 0 for _ in wins]
            for wi in range(len(wins)):
                keep = masks[wi].numpy().astype(bool)
                for r in range(R):
                    s = int(s_drawn[wi, r])
                    if not keep[s]:
                        js[wi, r] = np.nan
                        n_rep_sink += 1
                        continue
                    t_keep = keep[s:s + K]
                    js[wi, r, ~t_keep] = np.nan

        sums = np.nansum(js, axis=1)                    # (n_win, K)
        cnts = np.sum(np.isfinite(js), axis=1)
        bpt = float(np.mean([w["bytes_per_token"] for w in wins]))
        with np.errstate(invalid="ignore", divide="ignore"):
            per_lag = sums.sum(0) / cnts.sum(0)
        summ = influence_summary(per_lag, bpt)

        # Window bootstrap.
        brng = _bootstrap_rng(seed)
        boots = {k: [] for k in self.HEADLINE_KEYS}
        nw = len(wins)
        for _ in range(n_boot if nw > 1 else 0):
            pick = brng.integers(0, nw, size=nw)
            c = cnts[pick].sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                b = influence_summary(sums[pick].sum(0) / c, bpt)
            for k in boots:
                boots[k].append(b[k])

        def _sd(v):
            v = np.asarray(v, dtype=np.float64)
            v = v[np.isfinite(v)]
            return float(v.std(ddof=1)) if v.size > 1 else float("nan")

        out = {k: summ[k] for k in self.HEADLINE_KEYS}
        out.update({
            "_meta_influence_decay_exponent_se": _sd(boots["influence_decay_exponent"]),
            "_meta_influence_far_mass_se": _sd(boots["influence_far_mass"]),
            "_meta_influence_horizon_bytes_se": _sd(boots["influence_horizon_bytes"]),
            "_meta_influence_horizon_tokens": summ["horizon_tokens"],
            "_meta_influence_total_mass": summ["total_mass"],
            "_meta_influence_far_share": summ["far_share"],
            "_meta_influence_lag1": float(per_lag[0]),
            "_meta_decay_fit_r2": summ["fit_r2"],
            "_meta_decay_fit_bins": summ["n_fit_bins"],
            "_meta_decay_fit_lags": [FIT_FROM_LAG, K],
            "_meta_far_lag": FAR_LAG,
            "_meta_lag_bins": [list(b) for b in summ["bins"]],
            "_meta_influence_bin_means": [float(v) for v in summ["bin_means"]],
            "_meta_lag_bin_centers_tokens": [float(v) for v in summ["bin_centers"]],
            "_meta_lag_bin_centers_bytes": [float(v) * bpt for v in summ["bin_centers"]],
            "_meta_influence_per_lag": [float(v) for v in per_lag],
            "_meta_bytes_per_token": bpt,
            "_meta_max_lag": K,
            "_meta_window_tokens": W,
            "_meta_window_tokens_configured": window,
            "_meta_window_is_configured": bool(W == window),
            "_meta_replace_range": [lo, hi],
            "_meta_n_replacements_per_window": R,
            "_meta_n_windows": len(wins),
            "_meta_n_documents_too_short": n_short,
            "_meta_n_replacements": int(len(wins) * R - n_rep_sink),
            "_meta_n_replacements_on_sinks": n_rep_sink,
            "_meta_n_lags_empty": int(np.sum(cnts.sum(0) == 0)),
            "_meta_pre_replacement_max_js": pre_max,
            "_meta_seed": seed,
            "_meta_draw_digest": digest.hexdigest()[:16],
            "_meta_n_bootstrap": n_boot,
            "_meta_document_source": source_name,
            "_meta_min_doc_words": min_doc_words,
            "_meta_unigram_support": int(np.sum(counts > 0)),
            "_meta_divergence": "jensen_shannon_nats",
        })
        out.update(effective_dtype_meta(prec))
        out.update(sink_meta(sink_info))
        return out
