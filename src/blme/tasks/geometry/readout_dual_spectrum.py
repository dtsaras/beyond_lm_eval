"""``geometry_readout_dual_spectrum``: the dual spectrum of the LM readout.

A softmax head ``z = s (W h + b)`` makes the next-token distribution an
exponential family with natural parameter ``h`` (the state the head reads,
after the final norm) and expectation parameter ``mu = s W^T p``. For any two
positions x, x' the Jeffreys divergence is exactly

    KL(p_x || p_x') + KL(p_x' || p_x) = (h_x - h_x') . (mu_x - mu_x')
                                      = (z_x - z_x') . (p_x - p_x'),

so the Jeffreys dispersion ``D_J = 1/2 E_{x != x'} J(p_x, p_x') = tr Cov(h, mu)``
(unbiased covariance, nats). The logit form holds for any head, soft-capped
ones included, and is the reported level.

The pair (h, mu) transforms under the readout gauge ``(h, W, b) -> (A h + c,
W A^-1, b - W A^-1 c)``, which leaves every prediction unchanged, as
``Cov(h) -> A Cov(h) A^T`` and ``Cov(mu) -> A^-T Cov(mu) A^-1``, so the dual
spectrum ``kappa_i = sqrt(eig_i(Cov(h) Cov(mu)))`` is a property of the
predictive function. That invariance is exact in exact arithmetic only.
Computed from the two covariances, the kappa keys hold to about 1e-12 under
well-conditioned dense re-basings (cond(A) up to about 1e3-1e4) and drift
beyond (4% in sum kappa at cond(A) = 2e7; blind review, 2026-11-21); diagonal
gains are removed exactly at any scale by the balancing in ``dual_spectrum``.
``_meta_cov_h_*`` and ``_meta_cov_mu_*`` report the eigenvalue range,
condition number and zeroed-eigenvalue count of the two balanced covariances
the eigensolver reads, against ``_meta_cov_condition_limit`` = 1 / (d eps),
the condition number at which the rank tolerance starts zeroing eigenvalues.
``sum kappa >= tr Cov(h, mu)`` (the Bures coupling bound),
with equality exactly when mu responds linearly to h; the dual coherence
``c = tr Cov(h, mu) / sum kappa`` is the share of the maximal coupling the
softmax realises (1 at random initialisation). The shape of the spectrum is
reported as its Roy-Vetterli effective rank, that rank over d, and its decay
exponent over ranks 11-100.

Headline rows are the word-final positions (``word_final_positions``): the
prefix ends exactly at the end of a whitespace-delimited word, so the next
token begins a new word. This drops by construction the mid-character byte
positions (split UTF-8 characters) whose near-deterministic continuations
carried 9-15% of the trace on Pythia-70m/160m. The every-position value is
``_meta_``. Position 0 and massive-activation rows are excluded
(TOKEN_POLICY ``exclude_sinks``). Moments are streamed in float64; the number
of headline rows is pinned at ``points_per_dim * d`` because the effective
rank is biased low at finite N.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from .._precision import effective_dtype_meta, fp32_compute
from .._tokens import middle_depth_layers, sink_keep_masks, sink_meta, strip_trailing_special
from ..common import get_embeddings, get_lm_head, get_max_positions
from .utils import effective_rank, gpu_eigh, gpu_svdvals

logger = logging.getLogger("blme")

DEFAULT_POINTS_PER_DIM = 32
DEFAULT_NUM_SAMPLES = 4000
DEFAULT_MAX_LENGTH = 512
# Positions x vocabulary elements decoded at once (memory only).
DEFAULT_CHUNK_ELEMENTS = 2 ** 22
# A row whose norm exceeds this multiple of the running median at a middle
# depth is kept (its read state only) so that it can be removed exactly once
# the corpus-wide sink threshold (MASSIVE_FACTOR = 10) is known.
SINK_CANDIDATE_FACTOR = 2.5
# Ranks 11-100 (0-based 10..99) for the decay exponent: alpha-ReQ's window
# (reptrix get_powerlaw) and the prototype's.
TAIL_FIT = (10, 100)
WORD_FINAL_RULE = ("prefix ends at a word end: token t owns a non-empty character span "
                   "[s, e) with text[e-1] not whitespace and text[e] whitespace, and token "
                   "t+1 starts at or after e (no character shared across the boundary)")
_TEST = "tests/tasks/test_readout_dual_spectrum.py::"


# ── word-final positions ─────────────────────────────────────────────────

def word_final_positions(text: str, offsets) -> List[bool]:
    """Per token, whether the text read up to and including it ends a word.

    ``offsets`` are the tokenizer's character spans (``return_offsets_mapping``)
    into ``text``. Token t is word-final when its span is non-empty, the last
    character it reaches is not whitespace, the next character of the text is
    whitespace, and token t+1 starts at or after that point. The last rule
    excludes the pieces of a character a byte-level tokenizer splits across
    tokens (each piece reports the whole character's span), so a prediction
    made in the middle of a UTF-8 character is never word-final. The last
    token is never word-final (no successor to check). Whitespace-only tokens
    and special tokens (empty spans) are never word-final.
    """
    T = len(offsets)
    out = [False] * T
    n = len(text)
    for t in range(T - 1):
        s, e = int(offsets[t][0]), int(offsets[t][1])
        if e <= s or e <= 0 or e >= n:
            continue
        if text[e - 1].isspace() or not text[e].isspace():
            continue
        if int(offsets[t + 1][0]) < e:
            continue
        out[t] = True
    return out


# ── the readout as an explicit function of h ─────────────────────────────

class _Readout:
    """``z = c tanh(s (W h + b) / c)`` (softcap and scale optional) and its
    expectation parameter ``mu = grad_h log sum_v exp(z_v) = s W^T (p * g)``,
    ``g = sech^2(s (W h + b) / c)`` with a softcap and 1 without (then
    ``mu = s W^T p`` exactly)."""

    def __init__(self, W, bias, scale, cap, dtype):
        self.W = W.detach().to(dtype)
        self.b = None if bias is None else bias.detach().to(device=self.W.device, dtype=dtype)
        self.scale = scale
        self.cap = cap
        self.dtype = dtype

    @property
    def device(self):
        return self.W.device

    def rows(self, h):
        """``(z, log p, p, mu)`` for rows ``h`` (n, d): z in the compute dtype,
        the rest float64."""
        h = h.to(device=self.W.device, dtype=self.dtype)
        u = h @ self.W.T
        if self.b is not None:
            u = u + self.b
        if self.scale is not None:
            u = u * self.scale
        slope = None
        if self.cap is not None:
            a = u / self.cap
            z = self.cap * torch.tanh(a)
            # sech^2(a) without cancellation where tanh rounds to one.
            t = torch.exp(-2.0 * a.abs())
            slope = 4.0 * t / (1.0 + t).square()
        else:
            z = u
        lp = torch.log_softmax(z.double(), dim=-1)
        p = lp.exp()
        q = p if slope is None else p * slope.double()
        mu = (q.to(self.dtype) @ self.W).double()
        if self.scale is not None:
            mu = mu * float(self.scale)
        return z, lp, p, mu


# ── streaming float64 moments ────────────────────────────────────────────

class _Moments:
    """First and second moments of paired rows (h, mu) and the logit-space
    Jeffreys sums, in float64. Rows are shifted by a fixed reference (the
    first rows' mean) before accumulation, which leaves every covariance
    unchanged and keeps the centring well conditioned."""

    def __init__(self, d, V, device):
        f = dict(dtype=torch.float64, device=device)
        self.n_d = 0
        self.n_v = 0
        self.ref_h = None
        self.ref_m = None
        self.sh = torch.zeros(d, **f)
        self.sm = torch.zeros(d, **f)
        self.shh = torch.zeros(d, d, **f)
        self.smm = torch.zeros(d, d, **f)
        self.shm = torch.zeros((), **f)
        self.slp = torch.zeros(V, **f)
        self.sp = torch.zeros(V, **f)
        self.slpp = torch.zeros((), **f)

    def add_d(self, h, mu, sign=1.0):
        if h.shape[0] == 0:
            return
        if self.ref_h is None:
            self.ref_h = h.mean(0)
            self.ref_m = mu.mean(0)
        x = h - self.ref_h
        y = mu - self.ref_m
        self.n_d += int(sign) * int(x.shape[0])
        self.sh += sign * x.sum(0)
        self.sm += sign * y.sum(0)
        self.shh += sign * (x.T @ x)
        self.smm += sign * (y.T @ y)
        self.shm += sign * (x * y).sum()

    def add_v(self, lp, p, sign=1.0):
        if lp.shape[0] == 0:
            return
        self.n_v += int(sign) * int(lp.shape[0])
        self.slp += sign * lp.sum(0)
        self.sp += sign * p.sum(0)
        self.slpp += sign * (lp * p).sum()

    @property
    def n(self):
        return self.n_d

    def jeffreys(self):
        """``(D_J logit space, D_J d space)``, both 1/2 E J over distinct pairs."""
        n = self.n_d
        if n < 2 or self.n_v != n:
            return float("nan"), float("nan")
        dj_v = float((self.slpp - (self.slp @ self.sp) / n) / (n - 1))
        dj_d = float((self.shm - (self.sh @ self.sm) / n) / (n - 1))
        return dj_v, dj_d

    def covariances(self):
        n = self.n_d
        S_hh = (self.shh - torch.outer(self.sh, self.sh) / n) / (n - 1)
        S_mm = (self.smm - torch.outer(self.sm, self.sm) / n) / (n - 1)
        return (S_hh + S_hh.T) / 2, (S_mm + S_mm.T) / 2


# ── the dual spectrum ────────────────────────────────────────────────────

def _psd_sqrt(S, info: Optional[Dict[str, Any]] = None):
    """Symmetric square root of a PSD matrix. Eigenvalues under the numerical
    rank tolerance (e_max * n * eps) are set to 0: an exactly singular
    covariance (a LayerNorm state lies on a hyperplane) otherwise keeps a
    rounding-level eigenvalue whose square root is ~1e-8 of the scale.

    ``info``, when given, receives the largest eigenvalue, the smallest one
    kept and how many were set to 0 (read only; the root is unchanged)."""
    e, U = gpu_eigh(S)
    tol = float(e.max()) * S.shape[0] * float(torch.finfo(e.dtype).eps) if e.numel() else 0.0
    if info is not None:
        kept = e[e > tol]
        info.update({"eig_max": float(e.max()) if e.numel() else float("nan"),
                     "eig_min_kept": float(kept.min()) if kept.numel() else float("nan"),
                     "n_zeroed": int(e.numel() - kept.numel())})
    e = torch.where(e > tol, e, torch.zeros_like(e))
    return (U * e.sqrt()) @ U.T


def dual_spectrum(S_hh, S_mm, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """``kappa = sqrt(eig(S_hh S_mm))``, descending, float64.

    The per-coordinate scales are balanced first (h_i -> g_i h_i, mu_i ->
    mu_i / g_i with g_i = (S_mm,ii / S_hh,ii)^(1/4)), an exact diagonal gauge
    move that equalises the two diagonals and keeps a few large final-norm
    gains from dominating the eigensolver; kappa are then the singular values
    of S_hh^(1/2) S_mm^(1/2), which avoids squaring them.

    ``info``, when given, receives ``{"h": ..., "mu": ...}``: the eigenvalue
    range and zeroed count of each balanced covariance (``_psd_sqrt``).
    """
    S_hh = torch.as_tensor(S_hh, dtype=torch.float64).cpu()
    S_mm = torch.as_tensor(S_mm, dtype=torch.float64).cpu()
    a, b = torch.diagonal(S_hh), torch.diagonal(S_mm)
    ok = (a > 0) & (b > 0)
    g = torch.ones_like(a)
    g[ok] = (b[ok] / a[ok]) ** 0.25
    G = torch.outer(g, g)
    A = S_hh * G
    B = S_mm / G
    info_h = info_m = None
    if info is not None:
        info_h, info_m = info.setdefault("h", {}), info.setdefault("mu", {})
    k = gpu_svdvals(_psd_sqrt(A, info_h) @ _psd_sqrt(B, info_m))
    return np.sort(k.cpu().numpy().astype(np.float64))[::-1]


def tail_exponent(kappa: np.ndarray, lo: int = TAIL_FIT[0], hi: int = TAIL_FIT[1]):
    """``(alpha, (first rank, last rank))`` of kappa_r ~ r^-alpha, weighted
    least squares of log kappa on -log r with weights 1/r over ranks lo+1..hi,
    truncated to the numerically nonzero kappa when fewer (NaN under 3
    points). Nonzero is above kappa_max * n * eps (the numerical-rank
    tolerance), so zero-padded or hyperplane-constrained directions never
    enter the window."""
    k = np.asarray(kappa, dtype=np.float64)
    if k.size == 0 or not np.isfinite(k).all() or k[0] <= 0:
        return float("nan"), (lo + 1, lo)
    n_pos = int(np.sum(k > k[0] * k.size * np.finfo(np.float64).eps))
    hi = min(hi, n_pos)
    if hi - lo < 3:
        return float("nan"), (lo + 1, hi)
    r = np.arange(lo, hi) + 1.0
    y = np.log(k[lo:hi])
    w = np.sqrt(1.0 / r)
    A = np.column_stack([np.ones_like(r), -np.log(r)])
    beta, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    return float(beta[1]), (lo + 1, hi)


def spectrum_summary(acc: "_Moments", d: int) -> Dict[str, Any]:
    """Every reported statistic of one accumulator."""
    dj_v, dj_d = acc.jeffreys()
    nan = float("nan")
    out = {"jeffreys_dispersion": dj_v, "jeffreys_dispersion_dspace": dj_d,
           "dual_coupling": nan, "dual_coherence": nan, "dual_effective_rank": nan,
           "dual_effective_rank_frac": nan, "dual_tail_exponent": nan,
           "euclidean_participation_ratio": nan, "kappa": [], "tail_ranks": None,
           "n": int(acc.n), "conditioning": None}
    if acc.n < 2 or not (math.isfinite(dj_v) and math.isfinite(dj_d)):
        return out
    S_hh, S_mm = acc.covariances()
    if not (torch.isfinite(S_hh).all() and torch.isfinite(S_mm).all()):
        return out
    conditioning: Dict[str, Any] = {}
    kappa = dual_spectrum(S_hh, S_mm, info=conditioning)
    out["conditioning"] = conditioning
    sk = float(kappa.sum())
    er = effective_rank(kappa)
    alpha, ranks = tail_exponent(kappa)
    tr = float(torch.trace(S_hh))
    fro2 = float((S_hh * S_hh).sum())
    out.update({
        "dual_coupling": sk,
        "dual_coherence": (dj_d / sk) if sk > 0 else nan,
        "dual_effective_rank": er,
        "dual_effective_rank_frac": (er / d) if d > 0 else nan,
        "dual_tail_exponent": alpha,
        "euclidean_participation_ratio": (tr * tr / fro2) if fro2 > 0 else nan,
        "kappa": kappa,
        "tail_ranks": ranks,
    })
    return out


def conditioning_meta(conditioning: Optional[Dict[str, Any]], d: int) -> Dict[str, Any]:
    """The ``_meta_cov_*`` keys: for each balanced covariance (``h``, ``mu``)
    the eigenvalue range [smallest kept, largest], their ratio (the condition
    number on the kept subspace) and the count of eigenvalues the rank
    tolerance set to 0; and the limit 1 / (d eps) at which zeroing starts.
    NaN (and None for the counts) when the spectrum was not computed."""
    nan = float("nan")
    out: Dict[str, Any] = {}
    for side in ("h", "mu"):
        c = (conditioning or {}).get(side) or {}
        lo, hi = float(c.get("eig_min_kept", nan)), float(c.get("eig_max", nan))
        out[f"_meta_cov_{side}_eig_range"] = [lo, hi]
        out[f"_meta_cov_{side}_condition"] = (hi / lo) if lo > 0 and math.isfinite(hi) else nan
        out[f"_meta_cov_{side}_eigs_zeroed"] = int(c["n_zeroed"]) if "n_zeroed" in c else None
    out["_meta_cov_condition_limit"] = (1.0 / (d * float(np.finfo(np.float64).eps))) if d > 0 else nan
    return out


# ── the head of a model ──────────────────────────────────────────────────

def _logit_scale(model):
    cfg = getattr(model, "config", None)
    val = getattr(cfg, "logit_scale", None) if cfg is not None else None
    try:
        f = float(val) if val is not None else None
    except (TypeError, ValueError):
        return None
    return None if f is None or f == 1.0 else f


def _softcap(model):
    cfg = getattr(model, "config", None)
    cap = getattr(cfg, "final_logit_softcapping", None)
    if cap is None and cfg is not None and callable(getattr(cfg, "get_text_config", None)):
        cap = getattr(cfg.get_text_config(), "final_logit_softcapping", None)
    if cap is None:
        return None
    cap = float(cap)
    if not (math.isfinite(cap) and cap > 0):
        raise ValueError(f"final_logit_softcapping must be finite and positive, got {cap}")
    return cap


def _text(sample) -> str:
    if isinstance(sample, dict):
        return str(sample.get("text", ""))
    return str(sample)


@register_task("geometry_readout_dual_spectrum")
class ReadoutDualSpectrumTask(DiagnosticTask):
    """Jeffreys dispersion and the gauge-invariant dual spectrum of the LM readout."""

    # Point-cloud statistics of a BLME construct: position 0 and massive
    # activations are dropped (one mask per token from the middle depths), and
    # the headline keeps word-final positions only.
    TOKEN_POLICY = "exclude_sinks"
    SOURCES = {
        "paper": ("Amari & Nagaoka 2000 (Methods of Information Geometry, ch. 3: dual "
                  "coordinates of an exponential family); Bhatia, Jain & Lim 2019 "
                  "(arXiv:1712.01504, the Bures-Wasserstein coupling tr (A^1/2 B A^1/2)^1/2)"),
        "code": None,
        "llm_precedent": ("arXiv:2602.15293 (Park et al. 2026, dual coordinates of the softmax "
                          "head); arXiv:2609.11063 (Picozzi 2026, information geometry of LLM "
                          "outputs); no use of these invariants as a cross-model diagnostic found"),
    }
    ADAPTATION_NOTE = (
        "BLME construct: D_J = 1/2 E J(p_x, p_x') over distinct position pairs equals tr Cov(h, mu) "
        "exactly for an affine head, mu = s W^T p, and is computed in its logit form tr Cov(z, p), "
        "which is exact for any head, a final softcap included. The dual spectrum kappa = "
        "sqrt(eig(Cov(h) Cov(mu))) is invariant to every invertible affine re-basing of the read "
        "state with the compensating head change (exactly in exact arithmetic; numerically for "
        "re-basings with cond up to about 1e3-1e4, with the covariance conditioning in "
        "_meta_cov_*); for a soft-capped head mu is the gradient of "
        "the log-partition and the d-dimensional spectrum is a first-order approximation "
        "(_meta_d_space_exact false). The headline rows are word-final positions, which remove the "
        "mid-character byte positions that dominated the trace on Pythia, with position 0 and "
        "massive-activation rows excluded; every position is _meta_. N is pinned at points_per_dim "
        "* d headline rows because the effective rank is biased low at finite N. The read state is "
        "captured at the head's input; the forwards run in the model's dtype, the head in float32 "
        "(float64 on a float64 model), the softmax and every moment in float64, and the recomputed "
        "logits are checked against the model's own on every passage."
    )
    HEADLINE_KEYS = ("jeffreys_dispersion", "dual_coupling", "dual_coherence",
                     "dual_effective_rank", "dual_effective_rank_frac", "dual_tail_exponent")
    # At a fixed number of rows (the default pins N to points_per_dim * d, so
    # padding the width moves N): every key but the /d twin.
    SIZE_INVARIANT = {k: (("readout_affine_gauge", "zero_pad", "isometry"),
                          _TEST + "test_size_invariant_entry")
                      for k in ("jeffreys_dispersion", "dual_coupling", "dual_coherence",
                                "dual_effective_rank", "dual_tail_exponent")}

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running readout dual spectrum...")
        compute = str(self.config.get("compute_dtype", "model"))
        if compute not in ("float32", "model"):
            return {"error": f"compute_dtype must be 'float32' or 'model', got {compute!r}"}
        # The model's own dtype by default: on bf16 gpt2 weights, bf16 against
        # fp32 arithmetic moved every headline by under 0.05%. float32 wraps
        # the forwards, the refetch included, so a re-read state is the state
        # first read.
        with torch.no_grad(), fp32_compute(model, enabled=(compute == "float32")) as prec:
            out = self._evaluate(model, tokenizer, dataset)
        if "error" not in out:
            out.update(effective_dtype_meta(prec))
        return out

    def _evaluate(self, model, tokenizer, dataset):
        num_samples = int(self.config.get("num_samples", DEFAULT_NUM_SAMPLES))
        points_per_dim = float(self.config.get("points_per_dim", DEFAULT_POINTS_PER_DIM))
        n_rows_cfg = self.config.get("n_rows")
        max_length = int(self.config.get("max_length", DEFAULT_MAX_LENGTH))
        max_length = min(max_length, get_max_positions(model, max_length))
        chunk_elements = int(self.config.get("chunk_elements", DEFAULT_CHUNK_ELEMENTS))
        policy = self.TOKEN_POLICY
        if dataset is None:
            from ...cache import default_corpus_size, load_default_corpus
            dataset = load_default_corpus(min(num_samples, default_corpus_size()))
        samples = list(dataset)[:num_samples]
        if not samples:
            return {"error": "No passages"}

        head = get_lm_head(model)
        if head is not None:
            W, bias = head.weight, getattr(head, "bias", None)
        else:
            W, bias = get_embeddings(model), None
            if W is None:
                return {"error": "No unembedding matrix found"}
        try:
            cap = _softcap(model)
        except (TypeError, ValueError) as e:
            return {"error": f"Unusable final_logit_softcapping: {e}"}
        scale = _logit_scale(model)
        cd = torch.float64 if W.dtype == torch.float64 else torch.float32
        readout = _Readout(W, bias, scale, cap, cd)
        V, d = int(readout.W.shape[0]), int(readout.W.shape[1])
        n_target = int(n_rows_cfg) if n_rows_cfg is not None else int(math.ceil(points_per_dim * d))
        rows_per_chunk = max(1, chunk_elements // max(1, V))
        dev = readout.device

        acc_wb = _Moments(d, V, dev)
        acc_all = _Moments(d, V, dev)
        captured: Dict[str, torch.Tensor] = {}
        hook = None
        if head is not None:
            def _grab(module, args, kwargs=None):
                captured["h"] = args[0]
            hook = head.register_forward_pre_hook(_grab)

        offsets_ok = True
        offsets_reason = None
        model_device = next(model.parameters()).device
        mid: Optional[List[int]] = None
        doc_norms_all: List[Optional[Dict[int, torch.Tensor]]] = []
        doc_T: List[int] = []
        med_rows: Dict[int, List[torch.Tensor]] = {}
        med: Dict[int, float] = {}
        med_at = 0
        candidates: Dict[Tuple[int, int], Tuple[torch.Tensor, bool]] = {}
        used_upto: List[int] = []            # last accumulated position per doc (-1: none)
        wf_flags: List[List[bool]] = []
        n_wb = n_all = n_docs = n_stripped = 0
        guard_max = 0.0
        guard_tol = None
        done = False
        try:
            with torch.no_grad():
                for di, sample in enumerate(samples):
                    if done:
                        break
                    text = _text(sample)
                    kw = dict(return_tensors="pt", truncation=True, max_length=max_length)
                    offsets = None
                    if offsets_ok:
                        try:
                            enc = tokenizer(text, return_offsets_mapping=True, **kw)
                            offsets = enc.pop("offset_mapping", None)
                            if offsets is None:
                                offsets_ok = False
                                offsets_reason = "tokenizer returned no offset_mapping"
                        except (NotImplementedError, TypeError, ValueError) as e:
                            offsets_ok = False
                            offsets_reason = f"tokenizer gives no offsets ({type(e).__name__})"
                    if offsets is None:
                        enc = tokenizer(text, **kw)
                    enc, k = strip_trailing_special(dict(enc), tokenizer)
                    n_stripped += k
                    T = int(enc["input_ids"].shape[1])
                    n_docs += 1
                    wf = (word_final_positions(text, offsets[0, :T].tolist())
                          if offsets is not None else [False] * T)
                    wf_flags.append(wf)
                    doc_T.append(T)
                    if T < 2:
                        used_upto.append(-1)
                        doc_norms_all.append(None)
                        continue
                    captured.pop("h", None)
                    inputs = {kk: v.to(model_device) for kk, v in enc.items()
                              if isinstance(v, torch.Tensor)}
                    out = model(**inputs, output_hidden_states=True)
                    h, blocks = self._read_state(out, captured)
                    # Sink norms: middle depths of the block outputs only.
                    if mid is None:
                        mid = middle_depth_layers(dict.fromkeys(range(len(blocks))))
                        med_rows = {kk: [] for kk in mid}
                    doc_norms = {}
                    for kk in mid:
                        nr = torch.linalg.vector_norm(blocks[kk][0].to("cpu", torch.float64), dim=1)
                        med_rows[kk].append(nr[1:])
                        doc_norms[kk] = nr
                    doc_norms_all.append(doc_norms)
                    del blocks
                    n_seen = sum(int(x.numel()) for x in med_rows[mid[0]]) if mid else 0
                    if mid and (not med or n_seen >= 1.25 * med_at):
                        med = {kk: float(torch.cat(med_rows[kk]).median()) if n_seen else 0.0
                               for kk in mid}
                        med_at = n_seen

                    # Rows 1..cut: stop once the headline target is reached.
                    count_wf = offsets_ok
                    pos = torch.arange(1, T)
                    wf_t = torch.tensor(wf[1:], dtype=torch.bool)
                    sel = wf_t if count_wf else torch.ones(T - 1, dtype=torch.bool)
                    have = n_wb if count_wf else n_all
                    left = n_target - have
                    idx = torch.nonzero(sel).flatten()
                    cut = T - 1
                    if idx.numel() >= left:
                        cut = int(pos[idx[left - 1]]) if left > 0 else 0
                        done = True
                    used_upto.append(cut)
                    if cut < 1:
                        continue
                    rows = torch.arange(1, cut + 1)
                    wb_mask = torch.tensor(wf[1:cut + 1], dtype=torch.bool)

                    # Candidate sink rows: keep their read state for exact removal.
                    if policy == "exclude_sinks" and mid:
                        ratio = torch.zeros(T, dtype=torch.float64)
                        for kk in mid:
                            if med.get(kk, 0.0) > 0:
                                ratio = torch.maximum(ratio, doc_norms[kk] / med[kk])
                        for t in torch.nonzero(ratio[1:cut + 1] > SINK_CANDIDATE_FACTOR).flatten().tolist():
                            pt = t + 1
                            candidates[(di, pt)] = (h[pt].detach().to("cpu").clone(), bool(wf[pt]))

                    if guard_tol is None:
                        guard_tol = max(1e-3, 3.0 * float(torch.finfo(out.logits.dtype).eps))
                    if not bool(torch.isfinite(out.logits).all()):
                        return {"error": "model produced non-finite logits"}
                    mus = []
                    for a in range(0, rows.numel(), rows_per_chunk):
                        r = rows[a:a + rows_per_chunk]
                        z, lp, p, mu = readout.rows(h[r.to(h.device)])
                        real = out.logits[0, r.to(out.logits.device)].to(device=z.device, dtype=z.dtype)
                        den = float(real.abs().max())
                        g = float((z - real).abs().max()) / den if den > 0 else float("inf")
                        guard_max = max(guard_max, g)
                        if not math.isfinite(g) or g > guard_tol:
                            return {"error": (
                                "the head does not reproduce the model's logits from the state it "
                                f"reads (rel {g:.3g} > {guard_tol:.3g}); the readout identity does "
                                "not apply to this architecture")}
                        m = wb_mask[a:a + rows_per_chunk].to(lp.device)
                        acc_all.add_v(lp, p)
                        acc_wb.add_v(lp[m], p[m])
                        mus.append(mu)
                    h64 = h[rows.to(h.device)].to(device=dev, dtype=torch.float64)
                    mu64 = torch.cat(mus)
                    acc_all.add_d(h64, mu64)
                    acc_wb.add_d(h64[wb_mask.to(dev)], mu64[wb_mask.to(dev)])
                    n_all += int(rows.numel())
                    n_wb += int(wb_mask.sum())
                    del out
        finally:
            if hook is not None:
                hook.remove()

        if acc_all.n < 2:
            return {"error": "Too few token positions"}
        n_wb_raw, n_all_raw = acc_wb.n, acc_all.n
        dj_incl, _ = acc_wb.jeffreys()
        dj_all_incl, _ = acc_all.jeffreys()

        # Exact sink removal: the corpus-wide masks, then each accumulated
        # massive row is subtracted (recomputed from its read state).
        sink_info = {"n_massive_dropped": 0, "n_rows_pos_gt0": 0, "n_depths": 0}
        n_removed = n_refetched = 0
        if policy == "exclude_sinks" and mid:
            per_depth = [[(dn[kk] if dn is not None else torch.zeros(T, dtype=torch.float64)).unsqueeze(1)
                          for dn, T in zip(doc_norms_all, doc_T)] for kk in mid]
            masks, sink_info = sink_keep_masks(per_depth)
            todo: Dict[int, List[int]] = {}
            kept_h, kept_wf = [], []
            for di, (mask, cut) in enumerate(zip(masks, used_upto)):
                for t in torch.nonzero(~mask).flatten().tolist():
                    if not 1 <= t <= cut:
                        continue
                    if (di, t) in candidates:
                        hr, wf_r = candidates[(di, t)]
                        kept_h.append(hr)
                        kept_wf.append(wf_r)
                    else:
                        todo.setdefault(di, []).append(int(t))
            if kept_h:
                self._remove(readout, acc_all, acc_wb, torch.stack(kept_h), kept_wf, rows_per_chunk)
                n_removed += len(kept_h)
            for di, ts in todo.items():
                hr = self._refetch(model, tokenizer, samples[di], max_length, head, ts)
                self._remove(readout, acc_all, acc_wb, hr, [wf_flags[di][t] for t in ts],
                             rows_per_chunk)
                n_removed += len(ts)
                n_refetched += len(ts)

        head_s = spectrum_summary(acc_wb, d)
        all_s = spectrum_summary(acc_all, d)
        dj, dj_d = head_s["jeffreys_dispersion"], head_s["jeffreys_dispersion_dspace"]
        if not offsets_ok:
            for k in self.HEADLINE_KEYS:
                head_s[k] = float("nan")
        n_eff = int(acc_wb.n)
        kappa = head_s["kappa"]
        out = {k: head_s[k] for k in self.HEADLINE_KEYS}
        out.update({
            "_meta_jeffreys_dispersion_dspace": dj_d,
            "_meta_identity_rel_gap": (abs(dj - dj_d) / abs(dj)) if dj else float("nan"),
            "_meta_d_space_exact": cap is None,
            "_meta_dual_coherence_logit_level": (dj / head_s["dual_coupling"])
            if head_s["dual_coupling"] and math.isfinite(head_s["dual_coupling"]) else float("nan"),
            "_meta_euclidean_participation_ratio": head_s["euclidean_participation_ratio"],
            "_meta_kappa_top": [float(x) for x in kappa[:20]],
            "_meta_dual_tail_fit_ranks": list(head_s["tail_ranks"]) if head_s["tail_ranks"] else None,
            "_meta_jeffreys_dispersion_incl_sinks": dj_incl,
            "_meta_jeffreys_dispersion_all_positions_incl_sinks": dj_all_incl,
            "_meta_n_rows": n_eff,
            "_meta_n_rows_target": n_target,
            "_meta_n_target_met": bool((n_wb_raw if offsets_ok else n_all_raw) >= n_target),
            "_meta_points_per_dim": n_eff / d if d else float("nan"),
            "_meta_points_per_dim_target": points_per_dim if n_rows_cfg is None else n_target / d,
            "_meta_n_rows_all_positions": int(acc_all.n),
            "_meta_word_final_share": (n_wb_raw / n_all_raw) if n_all_raw else float("nan"),
            "_meta_word_final_rule": WORD_FINAL_RULE,
            "_meta_word_final_available": bool(offsets_ok),
            "_meta_d_model": d,
            "_meta_vocab_rows": V,
            "_meta_n_docs_read": n_docs,
            "_meta_n_docs_available": len(samples),
            "_meta_max_length": max_length,
            "_meta_n_trailing_special_stripped": n_stripped,
            "_meta_sink_rows_removed": n_removed,
            "_meta_sink_rows_refetched": n_refetched,
            "_meta_head_form": ("softcap" if cap is not None else
                                "affine_scaled" if scale is not None else "affine"),
            "_meta_final_logit_softcapping": cap,
            "_meta_logit_scale": scale,
            "_meta_head_bias": bias is not None,
            "_meta_head_guard_max_rel": guard_max,
            "_meta_read_state": "head_input" if head is not None else "hidden_states_last",
            "_meta_head_compute_dtype": str(cd).replace("torch.", ""),
            "_meta_moment_dtype": "float64",
        })
        out.update(conditioning_meta(head_s.get("conditioning"), d))
        if offsets_reason:
            out["_meta_word_final_reason"] = offsets_reason
        for k in ("jeffreys_dispersion", "dual_coupling", "dual_coherence", "dual_effective_rank",
                  "dual_effective_rank_frac", "dual_tail_exponent"):
            out[f"_meta_{k}_all_positions"] = all_s[k]
        out.update(sink_meta(sink_info))
        return out

    @staticmethod
    def _read_state(out, captured):
        """``(h, block outputs)``: the head's input (captured by the hook, or
        HF's last hidden state when the model has no head module) and the
        block outputs used for the sink norms.

        post-final-norm: DELIBERATELY the post-norm state -- the vector the
        head reads. The block outputs feed only the middle-depth sink norms
        (relative depth 0.1-0.9), which never reach the last entry.
        """
        hs = out.hidden_states
        h = captured.get("h")
        if h is None:
            h = hs[-1]
        return h[0], hs[1:]

    @staticmethod
    def _remove(readout, acc_all, acc_wb, h_rows, wf_rows, rows_per_chunk):
        """Subtract rows ``h_rows`` from both accumulators (from acc_wb only
        where ``wf_rows`` is set)."""
        wf = torch.tensor(wf_rows, dtype=torch.bool)
        for a in range(0, h_rows.shape[0], rows_per_chunk):
            hr = h_rows[a:a + rows_per_chunk]
            m = wf[a:a + rows_per_chunk].to(readout.device)
            _, lp, p, mu = readout.rows(hr)
            h64 = hr.to(device=readout.device, dtype=torch.float64)
            acc_all.add_v(lp, p, sign=-1.0)
            acc_all.add_d(h64, mu, sign=-1.0)
            acc_wb.add_v(lp[m], p[m], sign=-1.0)
            acc_wb.add_d(h64[m], mu[m], sign=-1.0)

    def _refetch(self, model, tokenizer, sample, max_length, head, positions):
        """Read states of ``positions`` of one passage, by a second forward."""
        captured: Dict[str, torch.Tensor] = {}
        hook = None
        if head is not None:
            def _grab(module, args, kwargs=None):
                captured["h"] = args[0]
            hook = head.register_forward_pre_hook(_grab)
        try:
            enc = tokenizer(_text(sample), return_tensors="pt", truncation=True,
                            max_length=max_length)
            enc, _ = strip_trailing_special(dict(enc), tokenizer)
            dev = next(model.parameters()).device
            with torch.no_grad():
                out = model(**{k: v.to(dev) for k, v in enc.items() if isinstance(v, torch.Tensor)},
                            output_hidden_states=True)
            h, _ = self._read_state(out, captured)
            return h[torch.tensor(positions, device=h.device)].detach().to("cpu")
        finally:
            if hook is not None:
                hook.remove()
