"""
Attention Rollout — cumulative token-to-token influence across layers.

Reference:
    Abnar, S. & Zuidema, W. (2020). "Quantifying Attention Flow in
    Transformers." ACL 2020, arXiv:2005.00928.

Definition (Abnar & Zuidema 2020, §3 "Attention Rollout"):
    Raw attention weights ignore the residual connections that carry a
    token's own representation forward. To account for the residual,
    each (head-averaged) layer attention matrix ``A_l`` (row-stochastic,
    ``A_l[i, j]`` = attention paid by query token ``i`` to key token
    ``j``) is augmented with the identity and re-normalized to stay
    row-stochastic::

        Ã_l = normalize(0.5·A_l + 0.5·I)      (equivalently normalize(A_l + I))

    The rollout is then the cumulative matrix product from the first
    layer up to the last::

        rollout = Ã_L · Ã_{L-1} · ... · Ã_1

    ``rollout[i, j]`` is the cumulative influence of input token ``j`` on
    the layer-L representation at position ``i`` — how much information
    "flows" from ``j`` to ``i`` through the whole stack. It is
    row-stochastic and non-negative.

    This reproduces the OFFICIAL reference bit-for-bit:
    ``samiraabnar/attention_flow`` @ 8044f53 ``attention_graph_util.py``
    ``compute_joint_attention(add_residual=True)`` — ``joint[0] = aug[0]``,
    ``joint[i] = aug[i].dot(joint[i-1])``. The verified artifact is the
    module-level helper :func:`_attention_rollout` below (parity proven in
    ``tests/tasks/parity/test_attention_rollout_parity.py``).

Implementation notes:
    * Tier-2 task: consumes the shared cache via
      ``cache.get_attentions()`` which returns
      ``{layer_idx: [tensor (H, T, T), ...]}`` — one per-sample tensor
      per layer, already on CPU. Each layer is HEAD-AVERAGED first, then
      the per-sample stack of head-averaged layers is passed to
      :func:`_attention_rollout`. Without a populated cache the task runs
      its own forward pass with ``output_attentions=True`` (matching the
      sibling attention-entropy task's fallback).
    * The augmentation ``0.5·A + 0.5·I`` and ``A + I`` produce the SAME
      matrix after row-normalization (the 0.5 factor cancels), so the
      helper follows the reference's ``A + I`` form. Both are verified
      identical in the parity test.
    * Core math runs in float64; degenerate (all-zero) rows are guarded
      so a fully-masked attention row teleports uniformly instead of
      producing NaN.
    * Emits only flat float summaries; ``_meta_`` prefixed keys are
      counts excluded from the analysis feature matrix; ``{"error": ...}``
      on failure.
"""

import logging

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")

# Rows whose sum falls below this are treated as degenerate (fully masked
# / all-zero attention) and replaced with a uniform row so the row-
# normalization and cumulative product never produce NaN.
_ZERO_ROW_EPS = 1e-12


def _row_normalize(M: np.ndarray) -> np.ndarray:
    """Row-normalize a non-negative matrix to be row-stochastic.

    Degenerate rows (sum < ``_ZERO_ROW_EPS``) are replaced with a uniform
    distribution over the row so no NaN propagates.
    """
    M = np.asarray(M, dtype=np.float64)
    row_sums = M.sum(axis=-1, keepdims=True)
    n = M.shape[-1]
    out = np.divide(
        M,
        row_sums,
        out=np.full_like(M, 1.0 / n),
        where=row_sums > _ZERO_ROW_EPS,
    )
    return out


def _attention_rollout(per_layer_attention) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema 2020) — the verified artifact.

    Reproduces ``samiraabnar/attention_flow`` @ 8044f53
    ``attention_graph_util.py:104-119`` ``compute_joint_attention(att_mat,
    add_residual=True)`` bit-for-bit (< 1e-9).

    Args:
        per_layer_attention: array-like of shape ``(L, N, N)`` — the
            HEAD-AVERAGED, row-stochastic attention matrix of each of the
            ``L`` layers for a single sample. ``A[l][i, j]`` is the
            attention paid by query token ``i`` to key token ``j`` at
            layer ``l``, layers ordered first -> last.

    Returns:
        ``(N, N)`` float64 cumulative rollout matrix
        ``Ã_L · Ã_{L-1} · ... · Ã_1``. Row-stochastic, non-negative.
        ``rollout[i, j]`` = cumulative influence of input token ``j`` on
        the final-layer representation at position ``i``.

    Raises:
        ValueError: if the input is not a 3-D ``(L, N, N)`` square stack
            with ``L >= 1``.
    """
    A = np.asarray(per_layer_attention, dtype=np.float64)
    if A.ndim != 3 or A.shape[1] != A.shape[2] or A.shape[0] < 1:
        raise ValueError(
            f"_attention_rollout expects (L, N, N) with L>=1, got {A.shape}"
        )

    L, N, _ = A.shape
    eye = np.eye(N, dtype=np.float64)

    # Augment each layer with the identity (residual) and re-normalize.
    # normalize(A + I) == normalize(0.5A + 0.5I); we use the reference's
    # A + I form.
    aug = np.stack([_row_normalize(A[l] + eye) for l in range(L)])

    # Cumulative matrix product: joint[0] = aug[0]; joint[i] = aug[i] @ joint[i-1].
    joint = aug[0]
    for l in range(1, L):
        joint = aug[l].dot(joint)
    return joint


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative 1-D array (0 = uniform, ->1 = concentrated)."""
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    n = x.size
    total = x.sum()
    if n == 0 or total <= 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / total) / n)


def _row_entropy(row: np.ndarray) -> float:
    """Shannon entropy (nats) of one row-stochastic distribution."""
    p = np.asarray(row, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _rollout_summaries(rollout: np.ndarray) -> dict:
    """Scalar summaries of a single ``(N, N)`` rollout matrix.

    Returns a dict of per-sample statistics (later averaged across
    samples by the task). ``rollout[i, j]`` = influence of input token
    ``j`` on final position ``i``; a COLUMN ``rollout[:, j]`` is the total
    influence emitted by input token ``j`` onto every position.
    """
    N = rollout.shape[0]
    diag = np.diag(rollout)                       # self-influence per position
    to_bos = rollout[:, 0]                         # influence FROM the BOS token
    to_last = rollout[:, -1]                       # influence FROM the last token
    # Row-wise concentration of where each position draws its information.
    row_ginis = np.array([_gini(rollout[i]) for i in range(N)])
    row_entropies = np.array([_row_entropy(rollout[i]) for i in range(N)])
    # Normalize entropy by log(N) so it is comparable across seq lengths.
    norm = np.log(N) if N > 1 else 1.0
    return {
        "mean_influence_to_bos": float(np.mean(to_bos)),
        "mean_influence_to_last": float(np.mean(to_last)),
        "mean_self_influence": float(np.mean(diag)),
        "mean_rollout_gini": float(np.nanmean(row_ginis)),
        "mean_rollout_norm_entropy": float(np.mean(row_entropies) / norm),
    }


@register_task("interpretability_attention_rollout")
class AttentionRolloutTask(DiagnosticTask):
    """
    Attention Rollout (Abnar & Zuidema 2020, arXiv:2005.00928).

    Head-averages each layer's attention, augments with the identity to
    account for residual connections, and takes the cumulative cross-layer
    matrix product to obtain token-to-token influence. Summarizes the
    final-layer rollout across samples.

    Outputs (flat floats):
        rollout_mean_influence_to_bos    — mean cumulative influence flowing
                                           from the first (BOS) token.
        rollout_mean_influence_to_last   — mean influence from the last token.
        rollout_mean_self_influence      — mean diagonal (how much a position
                                           retains its own input; higher =>
                                           less mixing / more identity-like).
        rollout_mean_gini                — mean row-wise Gini of the rollout
                                           (influence concentration; higher =>
                                           each position draws from few inputs).
        rollout_mean_norm_entropy        — mean row entropy / log(N) (diffuseness
                                           of influence; complements Gini).
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Attention Rollout Analysis...")

        use_cache = self.config.get("use_cache", True)
        num_samples = self.config.get("num_samples", 10)

        # --- Gather per-sample lists of per-layer (H, T, T) attentions ---
        # Cache fast-path: ``get_attentions()`` -> ``{layer: [(H,T,T), ...]}``.
        per_sample_attentions = None
        if cache is not None and cache.is_populated and use_cache:
            cached = cache.get_attentions(num_samples=num_samples)
            if cached:
                n_layers = max(cached.keys()) + 1
                max_samples = (
                    min(len(cached[0]), num_samples) if 0 in cached else 0
                )
                if max_samples > 0:
                    per_sample_attentions = []
                    for s_i in range(max_samples):
                        layer_attns = []
                        valid = True
                        for li in range(n_layers):
                            attn = cached.get(li, [None] * (s_i + 1))[s_i]
                            if attn is None:
                                valid = False
                                break
                            layer_attns.append(attn)
                        if valid:
                            per_sample_attentions.append(layer_attns)

        if per_sample_attentions is None:
            # Fallback: our own forward pass with output_attentions=True.
            if dataset is None:
                from ...cache import load_default_corpus
                dataset = load_default_corpus(num_samples)

            per_sample_attentions = []
            with torch.no_grad():
                for i, sample in enumerate(dataset):
                    if i >= num_samples:
                        break
                    text = (
                        sample if isinstance(sample, str)
                        else sample.get("text", "")
                    )
                    inputs = tokenizer(
                        text, return_tensors="pt", truncation=True, max_length=128,
                    ).to(model.device)
                    outputs = model(**inputs, output_attentions=True)
                    attentions = outputs.attentions
                    if not attentions:
                        return {
                            "error": (
                                "Model does not return attention weights. "
                                "Reload with attn_implementation='eager'."
                            )
                        }
                    if any(a is None for a in attentions):
                        return {
                            "error": (
                                "Model returned None attentions — likely "
                                "SDPA / FlashAttention. Reload with "
                                "attn_implementation='eager'."
                            )
                        }
                    per_sample_attentions.append(
                        [a.squeeze(0).detach().cpu() for a in attentions]
                    )

        if not per_sample_attentions:
            return {"error": "No attentions computed for attention rollout"}

        # --- Head-average each layer, roll out, summarize per sample ---
        summaries = {
            "mean_influence_to_bos": [],
            "mean_influence_to_last": [],
            "mean_self_influence": [],
            "mean_rollout_gini": [],
            "mean_rollout_norm_entropy": [],
        }
        n_used = 0
        for layer_attns in per_sample_attentions:
            if not layer_attns:
                continue
            # Each layer tensor is (H, T, T); head-average -> (T, T).
            try:
                per_layer = np.stack([
                    a.float().numpy().mean(axis=0) for a in layer_attns
                ])  # (L, T, T)
            except Exception:
                continue
            T = per_layer.shape[1]
            if T < 2:
                continue  # rollout on a single token is trivially [[1]]

            rollout = _attention_rollout(per_layer)
            s = _rollout_summaries(rollout)
            for k in summaries:
                summaries[k].append(s[k])
            n_used += 1

        if n_used == 0:
            return {"error": "Sequence lengths too short for attention rollout"}

        return {
            "rollout_mean_influence_to_bos": float(np.mean(summaries["mean_influence_to_bos"])),
            "rollout_mean_influence_to_last": float(np.mean(summaries["mean_influence_to_last"])),
            "rollout_mean_self_influence": float(np.mean(summaries["mean_self_influence"])),
            "rollout_mean_gini": float(np.nanmean(summaries["mean_rollout_gini"])),
            "rollout_mean_norm_entropy": float(np.mean(summaries["mean_rollout_norm_entropy"])),
            # _meta_ prefix => excluded from the analysis feature matrix so these
            # sampling/architecture counts cannot leak in as size proxies.
            "_meta_n_samples_used": int(n_used),
        }
