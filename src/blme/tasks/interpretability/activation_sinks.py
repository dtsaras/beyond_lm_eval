"""Attention sinks + massive activations + compression valley (2024-2025).

The 2024-2025 literature unifies three closely-related phenomena:

  1. **Attention sink** (Xiao et al. 2023, Gu et al. ICLR 2025,
     arXiv:2410.10781): a small number of tokens — typically the BOS
     token — absorb a disproportionate share of attention in almost
     every head of almost every layer, acting as an input-agnostic
     bias channel. Gu et al. define the Sinkε metric: the fraction
     of (head, position, token) triples for which the per-head
     attention weight on that token, summed over the queries that
     *could* have attended to it (causal mask), exceeds a threshold
     ε normalised by position count.

  2. **Massive activation** (Sun et al. 2024, arXiv:2402.17762): a
     tiny fraction of residual-stream entries (often < 0.01%) have
     magnitudes 100-1000× the typical activation. These are
     concentrated on start tokens and delimiters, and mechanistically
     *produce* the attention-sink phenomenon via their outsize impact
     on the softmax denominator.

  3. **Compression valley** (Arroyo et al. 2025, arXiv:2510.06477):
     the middle layers of every modern LLM exhibit a sharp dip in
     representation entropy — a "valley" where information is
     compressed before later layers expand it. Arroyo et al. connect
     theoretically that massive activations in the residual stream
     necessarily produce compression with bounds on the resulting
     entropy reduction, unifying all three phenomena.

This task reports:
  - ``sink_epsilon_fraction``: Gu et al. Sinkε at the default
    ε = 0.3 threshold. Higher = more attention concentrated on a
    few positions per head.
  - ``bos_attn_fraction``: simpler variant — mean attention weight
    on the first token, averaged over (sample, head, layer, query).
  - ``massive_activation_fraction``: fraction of residual-stream
    entries with magnitude > 100× the median |activation|.
  - ``massive_activation_max_ratio``: the max/median ratio of
    activation magnitudes (how outlier-heavy the tails are).
  - ``valley_layer``: 0-indexed layer of minimum entropy in the
    per-layer matrix-entropy profile (re-computed here so the task
    doesn't depend on caching).
  - ``valley_layer_norm``: ``valley_layer / (n_layers - 1)`` for
    cross-depth comparability.
  - ``valley_depth``: ``(endpoint mean) − (valley entropy)``;
    larger depth = sharper compression valley.

References:
  Xiao, Tian, Chen, Han, Han, "Efficient Streaming Language Models
    with Attention Sinks", ICLR 2024, arXiv:2309.17453.
  Gu, Pang, Du, Liu, Collier, Lin, "When Attention Sink Emerges in
    Language Models: An Empirical View", ICLR 2025 Spotlight,
    arXiv:2410.10781.
  Sun, Chen, Bai, Hu, Xiong, Kolter, "Massive Activations in Large
    Language Models", arXiv:2402.17762 (2024).
  Arroyo, Barbero, Dong, Bronstein, LeCun, Shwartz-Ziv,
    "Attention Sinks and Compression Valleys in LLMs are Two Sides
    of the Same Coin", arXiv:2510.06477 (2025).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


# ── Math helpers ────────────────────────────────────────────────────


def _sink_epsilon(attn: torch.Tensor, epsilon: float = 0.3) -> float:
    """Gu et al. 2025 Sinkε metric on a single sample's attention.

    Args:
        attn: (num_layers, num_heads, T, T) causal attention.
        epsilon: threshold (default 0.3 per the paper).

    Returns:
        Fraction of (layer, head, token) triples whose aggregated
        importance exceeds ε.

    Formula (reproduced from the reference code at
    https://github.com/sail-sg/Attention-Sink):

        ratios[k] = T - k      (how many queries can attend to key k
                                under the causal mask)
        importance[k] = Σ_q attn[q, k] / ratios[k]
        Sink₁ε = mean_{layer, head} [importance[first token] > ε]

    This is Gu et al.'s headline Sink₁ε: the fraction of (layer, head)
    pairs for which the FIRST token is an attention sink. (Fixed
    2026-06-22: previously averaged the indicator over ALL key positions
    k, a diluted statistic that does not match the reference's metric1[0].)
    """
    if attn.dim() == 4:
        # (L, H, T, T)
        L, H, T, T2 = attn.shape
        assert T == T2
    elif attn.dim() == 5:
        # (samples, L, H, T, T) — reduce to list of samples
        return float(np.mean([
            _sink_epsilon(a, epsilon) for a in attn
        ]))
    else:
        raise ValueError(f"attn.dim()={attn.dim()} not in (4, 5)")

    # ratios[k] = T - k (for k = 0, 1, ..., T-1)
    ratios = torch.arange(T, 0, -1, dtype=attn.dtype, device=attn.device)
    ratios = ratios.view(1, 1, 1, T)  # broadcast over (L, H, T_query, T_key)
    # importance: sum over query axis of attn/ratio, giving (L, H, T_key).
    importance = (attn / ratios).sum(dim=-2)  # (L, H, T_key)
    is_sink = (importance > epsilon).float()
    # Gu et al.'s Sink₁ε: the FIRST-token (k=0) sink indicator, averaged
    # over (layer, head). NOT a mean over all key positions.
    return float(is_sink[..., 0].mean().item())


def _massive_activation_fraction(
    hidden: torch.Tensor, threshold_ratio: float = 100.0
) -> float:
    """Fraction of entries in ``hidden`` with magnitude above
    ``threshold_ratio`` × the median absolute activation.

    Sun et al. 2024 characterise "massive activations" as entries
    ~10³-10⁴× larger than the bulk; 100× is a conservative cut-off
    that flags only the heaviest tail.
    """
    H = hidden.detach().float().flatten()
    if H.numel() == 0:
        return 0.0
    mag = H.abs()
    # Finite filter
    mask = torch.isfinite(mag)
    if not mask.any():
        return 0.0
    mag = mag[mask]
    med = float(mag.median().item())
    if med <= 0:
        return 0.0
    return float((mag > threshold_ratio * med).float().mean().item())


def _massive_activation_max_ratio(hidden: torch.Tensor) -> float:
    """Max absolute activation divided by median absolute activation.
    Well-behaved activations have ratio O(10); models with massive
    activations have ratio 10³+.
    """
    H = hidden.detach().float().flatten()
    if H.numel() == 0:
        return float("nan")
    mag = H.abs()
    mask = torch.isfinite(mag)
    if not mask.any():
        return float("nan")
    mag = mag[mask]
    med = float(mag.median().item())
    if med <= 0:
        return float("nan")
    return float(mag.max().item()) / med


def _compression_valley(entropy_profile) -> Dict[str, float]:
    """Characterise the middle-layer entropy dip.

    Returns:
      - ``valley_layer``: argmin of the profile (0-indexed).
      - ``valley_layer_norm``: ``valley_layer / (n-1)``, in [0, 1].
      - ``valley_depth``: ``mean(H[first], H[last]) − H[valley]``;
        negative values indicate no valley (monotone profile).
    """
    arr = np.asarray(list(entropy_profile), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 3:
        return {
            "valley_layer": -1,
            "valley_layer_norm": float("nan"),
            "valley_depth": float("nan"),
        }
    idx = int(np.argmin(arr))
    depth = 0.5 * (arr[0] + arr[-1]) - float(arr[idx])
    return {
        "valley_layer": idx,
        "valley_layer_norm": idx / float(max(1, n - 1)),
        "valley_depth": depth,
    }


def _row_normalised_entropy(X: torch.Tensor) -> float:
    """Matrix entropy (Wei et al. 2024) of ``X``: center, row-L2-
    normalise, compute eigenvalues of XᵀX/N, divide by tr, and take
    the Shannon entropy — then ``/ log d`` so the value lies in [0, 1].

    A duplicated (simpler) copy of matrix_entropy._matrix_entropy to
    keep this task self-contained.
    """
    if X.dim() != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        return float("nan")
    d = X.shape[1]
    X = X.float()
    X = X - X.mean(dim=0, keepdim=True)
    rn = X.norm(p=2, dim=1, keepdim=True)
    keep = rn.squeeze(-1) > 1e-12
    if int(keep.sum().item()) < 2:
        return float("nan")
    X = X[keep] / rn[keep]
    try:
        S = torch.linalg.svdvals(X)
    except Exception:
        return float("nan")
    lam = (S * S) / float(X.shape[0])
    total = float(lam.sum().item())
    if not math.isfinite(total) or total <= 0:
        return float("nan")
    lam = lam / total
    lam = lam.clamp(min=1e-30)
    H = float(-torch.sum(lam * torch.log(lam)).item())
    return H / math.log(max(d, 2))


# ── Task ────────────────────────────────────────────────────────────


@register_task("interpretability_activation_sinks")
class ActivationSinksTask(DiagnosticTask):
    """Per-model attention-sink / massive-activation / compression-
    valley diagnostic.

    Shares the forward pass across all three measurements by asking
    for ``output_attentions=True`` and ``output_hidden_states=True``
    once per sample.
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Activation-Sinks / Massive-Activations / Compression-Valley...")

        num_samples = int(self.config.get("num_samples", 10))
        max_length = int(self.config.get("max_length", 128))
        sink_epsilon = float(self.config.get("sink_epsilon", 0.3))
        massive_ratio = float(self.config.get("massive_threshold_ratio", 100.0))

        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)
        samples = list(dataset)[:num_samples]
        if not samples:
            return {"error": "Need at least one sample"}

        device = next(model.parameters()).device

        # Per-sample aggregates
        sink_fracs = []
        bos_attn_means = []
        massive_fracs_per_layer: Dict[int, List[float]] = {}
        massive_max_ratios_per_layer: Dict[int, List[float]] = {}
        entropy_per_layer: Dict[int, List[float]] = {}

        import torch as _torch

        with _torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(
                    text, return_tensors="pt",
                    truncation=True, max_length=max_length,
                )
                # BatchEncoding vs dict.
                if hasattr(inputs, "to") and callable(inputs.to):
                    inputs = inputs.to(device)
                else:
                    inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                              for k, v in inputs.items()}

                try:
                    out = model(
                        **inputs,
                        output_hidden_states=True,
                        output_attentions=True,
                    )
                except Exception as e:
                    logger.info(f"  forward failed: {type(e).__name__}: {e}")
                    continue

                hs = getattr(out, "hidden_states", None)
                attns = getattr(out, "attentions", None)

                # ── Attention sink (needs attentions) ───────────────
                if attns is not None and all(a is not None for a in attns):
                    # Stack to (L, H, T, T).
                    try:
                        a_stack = _torch.stack([a[0] for a in attns], dim=0)
                        if a_stack.dim() == 4:
                            sink_fracs.append(_sink_epsilon(a_stack, epsilon=sink_epsilon))
                            # Mean attention weight on token 0 across
                            # (L, H, queries).
                            bos_attn_means.append(
                                float(a_stack[:, :, :, 0].mean().item())
                            )
                    except Exception as e:
                        logger.info(f"  attention-sink parse failed: {e}")

                # ── Massive activations (per-layer residual stream) ─
                if hs is not None:
                    # Skip the embedding-output entry (index 0).
                    for li, h in enumerate(hs[1:]):
                        if h is None:
                            continue
                        H_flat = h[0].detach().float()  # (T, D)
                        frac = _massive_activation_fraction(H_flat, threshold_ratio=massive_ratio)
                        ratio = _massive_activation_max_ratio(H_flat)
                        massive_fracs_per_layer.setdefault(li, []).append(frac)
                        massive_max_ratios_per_layer.setdefault(li, []).append(ratio)
                        # Reuse the matrix-entropy formula for the
                        # compression-valley trajectory.
                        H_entropy = _row_normalised_entropy(H_flat)
                        if np.isfinite(H_entropy):
                            entropy_per_layer.setdefault(li, []).append(H_entropy)

        # ── Aggregate ───────────────────────────────────────────────
        if not sink_fracs and not massive_fracs_per_layer and not entropy_per_layer:
            return {"error": "No usable outputs"}

        result: Dict[str, object] = {}

        if sink_fracs:
            result["sink_epsilon_fraction"] = float(np.mean(sink_fracs))
            result["sink_epsilon"] = sink_epsilon
        if bos_attn_means:
            result["bos_attn_fraction"] = float(np.mean(bos_attn_means))

        # Massive activations: layer profile + global aggregate.
        if massive_fracs_per_layer:
            layer_fracs = [
                float(np.mean(massive_fracs_per_layer[li]))
                for li in sorted(massive_fracs_per_layer.keys())
            ]
            layer_ratios = [
                float(np.mean(massive_max_ratios_per_layer[li]))
                for li in sorted(massive_max_ratios_per_layer.keys())
            ]
            result["massive_activation_fraction"] = float(np.mean(layer_fracs))
            result["massive_activation_fraction_per_layer"] = layer_fracs
            result["massive_activation_max_ratio"] = float(np.max(layer_ratios))
            result["massive_activation_max_ratio_per_layer"] = layer_ratios

        # Compression valley.
        if entropy_per_layer:
            layer_entropy = [
                float(np.mean(entropy_per_layer[li]))
                for li in sorted(entropy_per_layer.keys())
            ]
            valley = _compression_valley(layer_entropy)
            result["valley_layer"] = int(valley["valley_layer"])
            result["valley_layer_norm"] = float(valley["valley_layer_norm"])
            result["valley_depth"] = float(valley["valley_depth"])
            result["entropy_per_layer"] = layer_entropy

        result["n_layers"] = max(
            len(result.get("massive_activation_fraction_per_layer", [])),
            len(result.get("entropy_per_layer", [])),
        )
        return result
