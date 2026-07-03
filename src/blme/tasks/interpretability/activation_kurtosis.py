"""Activation kurtosis — an outlier / quantizability signal (2024-2025).

Modern LLMs concentrate a disproportionate share of their activation
magnitude in a tiny number of heavy-tailed *outlier features* — a handful
of hidden dimensions whose per-channel distribution across tokens has an
extremely heavy tail. These outlier channels are precisely what breaks
naive low-bit quantization (the dynamic range of one channel forces a
coarse scale on every other channel in the same group) and are the same
"massive activations" that produce attention sinks.

The natural scalar summary of a channel's tailedness is its **excess
(Fisher) kurtosis**:

    for an activation channel x (a hidden dimension) with values across
    tokens,

        excess_kurtosis(x) = E[(x - mu)^4] / sigma^4  -  3

    where mu, sigma^2 are the mean and variance of x. A normal channel has
    excess kurtosis 0; a Laplace channel ~3; a uniform channel ~-1.2; a
    channel with a single massive spike has very large positive kurtosis.

High per-channel kurtosis therefore flags heavy-tailed outlier activations
that hurt post-training quantization and indicate the massive-activation
regime.

This task reports, from a cloud of per-layer hidden states A = (N_tokens,
hidden_dim):

  - ``kurtosis_mean``          : mean over channels & layers of the
                                 per-channel excess kurtosis.
  - ``kurtosis_max``           : the single most heavy-tailed channel,
                                 max over channels & layers.
  - ``kurtosis_frac_above_thr``: fraction of (channel, layer) pairs whose
                                 excess kurtosis exceeds a threshold
                                 (default 10) — how widespread the
                                 outlier-feature phenomenon is.
  - ``kurtosis_tensor``        : excess kurtosis of the flattened
                                 activation tensor (all entries pooled),
                                 the global heavy-tailedness of the
                                 residual stream.
  - per-layer profiles (``*_per_layer``) and the last-layer scalars.

The per-channel kurtosis kernel is verified BIT-EXACTLY against
``scipy.stats.kurtosis(A, axis=0, fisher=True, bias=True)`` — scipy's
default convention (Fisher = excess, bias=True = population moments, i.e.
NO n-based sample-size correction). See
``tests/tasks/parity/test_activation_kurtosis_parity.py``.

References:
    Akhondzadeh, Bojchevski, Eleftheriou, Dazzi, "KurTail: Kurtosis-based
        LLM Quantization", Findings of EMNLP 2025 (arXiv:2503.01483) —
        rotates activations to minimise per-channel kurtosis before 4-bit
        quantization; the primary motivation for kurtosis as a
        quantizability signal.
    Sun, Chen, Bai, Hu, Xiong, Kolter, "Massive Activations in Large
        Language Models", arXiv:2402.17762 (2024) — the heavy-tailed
        outlier activations kurtosis measures.
    Dettmers, Lewis, Belkada, Zettlemoyer, "LLM.int8(): 8-bit Matrix
        Multiplication for Transformers at Scale", NeurIPS 2022
        (arXiv:2208.07339) — the outlier-feature phenomenon that motivates
        per-channel outlier metrics.

    Numeric reference for kurtosis: ``scipy.stats.kurtosis``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")

# Channels whose across-token variance is below this (in the squared
# activation units) are treated as degenerate (a constant channel has
# undefined kurtosis: 0/0). They are dropped from the per-channel
# statistics rather than propagated as NaN/inf.
_ZERO_VAR_EPS = 1e-12


def _activation_kurtosis_stats(
    A,
    threshold: float = 10.0,
) -> Dict[str, object]:
    """Per-channel excess-kurtosis statistics for one activation cloud.

    This is the verified numeric artifact of the task. The per-channel
    excess kurtosis is computed to match ``scipy.stats.kurtosis`` with
    scipy's DEFAULT convention:

        fisher=True  -> "excess" kurtosis, normal -> 0
        bias=True    -> population moments E[(x-mu)^4]/E[(x-mu)^2]^2 - 3,
                        i.e. NO n-based sample-size (Fisher) correction.

    Args:
        A: array-like of shape (N_tokens, D) — activations for one layer,
           rows are tokens, columns are hidden channels.
        threshold: excess-kurtosis cut-off for ``frac_above_threshold``
                   (KurTail-style "how many channels are outlier-heavy").

    Returns:
        Dict with:
          - ``per_channel_kurtosis``: (D,) float64 array of excess
            kurtosis per channel, ``nan`` for degenerate (zero-variance)
            channels. This array is what the parity test compares to
            scipy element-for-element on the non-degenerate channels.
          - ``mean``  : mean over finite (non-degenerate) channels.
          - ``max``   : max  over finite channels.
          - ``frac_above_threshold`` : fraction of finite channels whose
            excess kurtosis > ``threshold``.
          - ``tensor_kurtosis``: excess kurtosis of ALL entries pooled
            (the flattened tensor), scipy-convention.
          - ``n_channels``, ``n_finite_channels``, ``n_tokens``.

    Empty / all-degenerate inputs yield NaN summaries and empty arrays
    rather than raising.
    """
    A = np.asarray(A, dtype=np.float64)
    empty = {
        "per_channel_kurtosis": np.empty(0, dtype=np.float64),
        "mean": float("nan"),
        "max": float("nan"),
        "frac_above_threshold": float("nan"),
        "tensor_kurtosis": float("nan"),
        "n_channels": 0,
        "n_finite_channels": 0,
        "n_tokens": 0,
    }
    if A.ndim != 2 or A.shape[0] < 2 or A.shape[1] < 1:
        return empty

    N, D = A.shape

    # ── Per-channel excess kurtosis (scipy fisher=True, bias=True) ──────
    # Population central moments along the token axis (axis=0), matching
    # scipy's bias=True path exactly: m_k = mean((x - mean(x))**k).
    mu = A.mean(axis=0, keepdims=True)                 # (1, D)
    dev = A - mu                                       # (N, D)
    m2 = np.mean(dev ** 2, axis=0)                     # (D,) population var
    m4 = np.mean(dev ** 4, axis=0)                     # (D,)

    per_channel = np.full(D, np.nan, dtype=np.float64)
    # Guard zero-variance (constant) channels: kurtosis is 0/0 -> undefined.
    nz = m2 > _ZERO_VAR_EPS
    per_channel[nz] = m4[nz] / (m2[nz] ** 2) - 3.0

    finite = np.isfinite(per_channel)
    n_finite = int(finite.sum())
    if n_finite > 0:
        vals = per_channel[finite]
        mean_k = float(np.mean(vals))
        max_k = float(np.max(vals))
        frac_above = float(np.mean(vals > threshold))
    else:
        mean_k = max_k = frac_above = float("nan")

    # ── Overall activation-tensor kurtosis (all entries pooled) ────────
    flat = A.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size >= 2:
        fmu = flat.mean()
        fdev = flat - fmu
        fm2 = np.mean(fdev ** 2)
        fm4 = np.mean(fdev ** 4)
        tensor_k = (
            float(fm4 / (fm2 ** 2) - 3.0) if fm2 > _ZERO_VAR_EPS else float("nan")
        )
    else:
        tensor_k = float("nan")

    return {
        "per_channel_kurtosis": per_channel,
        "mean": mean_k,
        "max": max_k,
        "frac_above_threshold": frac_above,
        "tensor_kurtosis": tensor_k,
        "n_channels": int(D),
        "n_finite_channels": n_finite,
        "n_tokens": int(N),
    }


@register_task("interpretability_activation_kurtosis")
class ActivationKurtosisTask(DiagnosticTask):
    """Per-model activation-kurtosis / outlier-feature diagnostic.

    Computes per-channel excess (Fisher) kurtosis of the residual-stream
    activations at every layer, following KurTail (Akhondzadeh et al.,
    Findings of EMNLP 2025, arXiv:2503.01483): high per-channel kurtosis
    ⇒ heavy-tailed outlier activations that hurt low-bit quantization and
    signal the massive-activation regime (Sun et al. 2024).

    Consumes the shared hidden-state cache when available (per-CHANNEL
    kurtosis across tokens, per layer); otherwise runs its own forward
    pass. Only summary scalars + per-layer profiles are emitted; counts
    are ``_meta_``-prefixed so they cannot leak into the analysis feature
    matrix as size proxies.

    Outputs (flat floats; excess/Fisher convention, normal ⇒ 0):
        kurtosis_mean               — mean per-channel excess kurtosis
                                      (over channels & layers)
        kurtosis_max                — most heavy-tailed channel
        kurtosis_frac_above_thr     — fraction of channels above threshold
        kurtosis_tensor             — kurtosis of the whole pooled tensor
        kurtosis_mean_last_layer    — last-layer per-channel mean
        kurtosis_max_last_layer     — last-layer per-channel max
        *_per_layer                 — per-layer profiles
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Activation-Kurtosis (outlier-feature / quantizability)...")

        num_samples = int(self.config.get("num_samples", 100))
        threshold = float(self.config.get("kurtosis_threshold", 10.0))
        use_cache = self.config.get("use_cache", True)

        # ── Gather per-layer hidden states {layer_idx: (N_tokens, D)} ───
        per_layer: Optional[Dict[int, object]] = None
        if cache is not None and cache.is_populated and use_cache:
            per_layer = cache.get_hidden_states(
                layer_idx="all", num_samples=num_samples,
            )

        if per_layer is None:
            # Fallback: own forward pass via the geometry collector, which
            # returns {layer_idx: (TotalTokens, D)} for layer_idx="all".
            from ..geometry.utils import collect_hidden_states
            from ...cache import load_default_corpus

            if dataset is None:
                dataset = load_default_corpus(num_samples)
            per_layer = collect_hidden_states(
                model, tokenizer, dataset,
                num_samples=num_samples, layer_idx="all",
            )

        if not per_layer:
            return {"error": "No hidden states available for activation kurtosis"}

        layer_keys = sorted(per_layer.keys())

        mean_per_layer: List[float] = []
        max_per_layer: List[float] = []
        frac_per_layer: List[float] = []
        tensor_per_layer: List[float] = []
        # Pool per-channel kurtoses across layers for the global mean/max
        # (each layer contributes its own D channels — long-tailed layers
        # are not down-weighted relative to short ones).
        all_channel_kurt: List[np.ndarray] = []
        n_tokens_used = 0

        for k in layer_keys:
            X = per_layer[k]
            if isinstance(X, torch.Tensor):
                X = X.detach().float().cpu().numpy()
            stats = _activation_kurtosis_stats(X, threshold=threshold)
            n_tokens_used = max(n_tokens_used, stats["n_tokens"])
            mean_per_layer.append(stats["mean"])
            max_per_layer.append(stats["max"])
            frac_per_layer.append(stats["frac_above_threshold"])
            tensor_per_layer.append(stats["tensor_kurtosis"])
            pc = stats["per_channel_kurtosis"]
            pc = pc[np.isfinite(pc)]
            if pc.size:
                all_channel_kurt.append(pc)

        if not all_channel_kurt:
            return {"error": "No finite-variance channels for activation kurtosis"}

        pooled = np.concatenate(all_channel_kurt)
        mean_arr = np.asarray(mean_per_layer, dtype=np.float64)
        max_arr = np.asarray(max_per_layer, dtype=np.float64)
        frac_arr = np.asarray(frac_per_layer, dtype=np.float64)
        tensor_arr = np.asarray(tensor_per_layer, dtype=np.float64)

        def _finite_mean(a):
            a = a[np.isfinite(a)]
            return float(np.mean(a)) if a.size else float("nan")

        def _finite_max(a):
            a = a[np.isfinite(a)]
            return float(np.max(a)) if a.size else float("nan")

        result: Dict[str, object] = {
            # Pooled over ALL channels of ALL layers (the headline signal).
            "kurtosis_mean": float(np.mean(pooled)),
            "kurtosis_max": float(np.max(pooled)),
            "kurtosis_frac_above_thr": float(np.mean(pooled > threshold)),
            "kurtosis_median": float(np.median(pooled)),
            # Whole-tensor kurtosis, averaged over layers.
            "kurtosis_tensor": _finite_mean(tensor_arr),
            # Last-layer scalars (comparable across models: final residual).
            "kurtosis_mean_last_layer": float(mean_arr[-1]),
            "kurtosis_max_last_layer": float(max_arr[-1]),
            # Depth profiles.
            "kurtosis_mean_per_layer": [float(v) for v in mean_arr],
            "kurtosis_max_per_layer": [float(v) for v in max_arr],
            "kurtosis_frac_above_thr_per_layer": [float(v) for v in frac_arr],
            "kurtosis_threshold": threshold,
            # _meta_ prefix => excluded from the analysis feature matrix so
            # these architecture/sampling counts cannot leak in as size
            # proxies (Audit-V2 convention).
            "_meta_n_layers": int(len(layer_keys)),
            "_meta_n_channels_pooled": int(pooled.size),
            "_meta_n_tokens_used": int(n_tokens_used),
        }
        return result
