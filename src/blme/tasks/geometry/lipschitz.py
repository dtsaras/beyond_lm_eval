"""Relative-change ratio between adjacent layers ("empirical Lipschitz").

Per residual update l -> l+1:
    ratio(l, x_i) = ||h_{l+1}(x_i) - h_l(x_i)|| / ||h_l(x_i)||
    L_hat(l)      = mean_i ratio(l, x_i)
summarised over the block-to-block updates; the embedding -> block 0 update
is reported on its own (it reads the embedding-scale convention, and it was
the maximum on every model measured). This is not an operator
Lipschitz constant (Scaman & Virmaux 2018 bound Jacobian norms); it measures
how much each block rewrites the residual stream relative to its size.
Position 0 is excluded (``TOKEN_POLICY``): the attention-sink transition
dominates single layers.
"""

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .._tokens import hidden_states_for_task, massive_token_share, position0_share, select_rows
from ..common import get_num_layers
from .utils import as_numpy
import numpy as np
import logging
logger = logging.getLogger("blme")


def _rownorm(M):
    """Row-wise L2 norms, exact at any representable magnitude (each row's
    largest entry is factored out before squaring)."""
    M = np.asarray(M, dtype=np.float64)
    m = np.max(np.abs(M), axis=1)
    out = np.zeros(M.shape[0], dtype=np.float64)
    nz = m > 0
    out[nz] = m[nz] * np.linalg.norm(M[nz] / m[nz, None], axis=1)
    return out


def _transitions(rows, layer_indices):
    """Per-update mean relative change and contraction rate over the given
    rows, plus the paper text's epsilon form."""
    means, rates, paper = [], [], []
    for i in range(len(layer_indices) - 1):
        h_l = as_numpy(rows[layer_indices[i]])
        h_next = as_numpy(rows[layer_indices[i + 1]])
        n_tokens = min(len(h_l), len(h_next))
        h_l = h_l[:n_tokens]
        h_next = h_next[:n_tokens]
        diff_norms = _rownorm(h_next - h_l)
        input_norms = _rownorm(h_l)
        if n_tokens:
            # The paper text's literal ||dh|| / (||h|| + 1e-8), kept as _meta_:
            # the epsilon breaks scale invariance.
            paper.append(float(np.mean(diff_norms / (input_norms + 1e-8))))
        # Only an exactly-zero state has no relative change; an absolute floor
        # would break the ratio's scale invariance.
        valid = input_norms > 0
        if not np.any(valid):
            means.append(float("nan"))
            rates.append(float("nan"))
            continue
        means.append(float(np.mean(diff_norms[valid] / input_norms[valid])))
        rates.append(float(np.mean(_rownorm(h_next)[valid] / input_norms[valid])))
    return means, rates, (float(np.mean(paper)) if paper else float("nan"))


@register_task("geometry_lipschitz")
class LipschitzContinuityTask(DiagnosticTask):
    """Relative layer-wise change ratio ||h_{l+1}(x) - h_l(x)|| / ||h_l(x)||.

    The per-update value is the mean over tokens (position 0 excluded);
    ``relative_change_max`` is the maximum of those per-update means over the
    block-to-block updates, and ``embedding_relative_change`` is the
    embedding -> block 0 update. The deepest entry is the hooked raw block output (the post-ln_f
    entry is trimmed when the hook does not fire).
    """

    TOKEN_POLICY = "exclude_pos0"

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Lipschitz Continuity Analysis...")
        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]

        num_samples = self.config.get("num_samples", 20)
        use_cache = self.config.get("use_cache", True)

        # The embedding is included: the statistic covers all L residual
        # updates, starting from the input to block 0.
        all_layers, info = hidden_states_for_task(
            model, tokenizer, dataset, cache, num_samples=num_samples,
            replace_final_norm=True, layer_idx="all", use_cache=use_cache,
            include_embedding=True, per_sample=True)
        last_is_block = bool(info["last_is_block"])

        if not all_layers:
            return {"error": "No hidden states collected"}

        layer_indices = sorted(all_layers.keys())
        # Key 0 is the embedding output when the source supplied it (one more
        # entry than blocks); decided before the post-norm trim below.
        n_blocks = int(get_num_layers(model) or 0)
        has_embedding = n_blocks > 0 and len(layer_indices) == n_blocks + 1
        # Without the hooked block output the deepest entry is ln_f(b_{L-1}),
        # not a block output: drop it.
        if not last_is_block:
            if len(layer_indices) > 2:
                layer_indices = layer_indices[:-1]
        n_layers = len(layer_indices)

        if n_layers < 2:
            return {"error": "Need at least 2 layers for Lipschitz analysis"}

        # Rows under the token policy (the same rows at every layer), and every
        # row for the inclusive provenance value.
        policy = "exclude_pos0" if self.TOKEN_POLICY == "exclude_pos0" else "all"
        rows, row_meta = {}, None
        for k in layer_indices:
            rows[k], row_meta = select_rows(all_layers[k], policy)
        every = {k: select_rows(all_layers[k], "all")[0] for k in layer_indices}

        block_indices = layer_indices[1:] if has_embedding else layer_indices
        if len(block_indices) < 2:
            return {"error": "Need at least 2 block outputs for Lipschitz analysis"}
        lipschitz_means, contraction_rates, _paper_eps_mean = _transitions(rows, block_indices)
        incl_means, _, _ = _transitions(every, block_indices)
        if has_embedding:
            emb_means, emb_rates, _ = _transitions(rows, layer_indices[:2])
            emb_rel, emb_rate = emb_means[0], emb_rates[0]
        else:
            emb_rel = emb_rate = float("nan")

        lip_arr = np.array(lipschitz_means)
        _lf = lip_arr[np.isfinite(lip_arr)]
        _contractive_fraction = (float(np.mean(_lf < 1.0)) if _lf.size
                                 else float("nan"))
        _cr_arr = np.asarray(contraction_rates, dtype=np.float64)
        if _cr_arr.size and np.any(np.isfinite(_cr_arr)):
            _cr_mean = float(np.nanmean(_cr_arr))
            _cr_std = float(np.nanstd(_cr_arr))
        else:
            _cr_mean = _cr_std = float("nan")

        # nan-aware: a degenerate update records NaN, never a legal-looking 0.0.
        if not np.any(np.isfinite(lip_arr)):
            mean_v = max_v = std_v = float("nan")
            max_layer = -1
        else:
            mean_v = float(np.nanmean(lip_arr))
            max_v = float(np.nanmax(lip_arr))
            std_v = float(np.nanstd(lip_arr))
            max_layer = int(np.nanargmax(lip_arr))
        incl = np.asarray(incl_means, dtype=np.float64)

        mid = all_layers[layer_indices[n_layers // 2]]
        return {
            "relative_change_mean": mean_v,
            # The maximum over per-update MEANS, not a token-level maximum.
            "relative_change_max": max_v,
            "relative_change_std": std_v,
            # An absolute depth index; the fraction is the comparable form.
            "_meta_relative_change_max_layer": max_layer,
            "_meta_relative_change_max_layer_frac": (
                float(max_layer) / max(1, len(lip_arr) - 1)
                if max_layer >= 0 else float("nan")),
            "contractive_fraction": _contractive_fraction,
            "_meta_relative_change_mean_paper_eps": _paper_eps_mean,
            # Every row, position 0 included (provenance).
            "_meta_relative_change_mean_incl_pos0": (
                float(np.nanmean(incl)) if np.any(np.isfinite(incl)) else float("nan")),
            # Legacy aliases of relative_change_* (bit-identical duplicates).
            "_meta_lipschitz_mean": mean_v,
            "_meta_lipschitz_max": max_v,
            "_meta_lipschitz_std": std_v,
            "_meta_lipschitz_max_layer": max_layer,
            "_meta_last_entry_is_block_output": bool(last_is_block),
            "mean_contraction_rate": _cr_mean,
            "contraction_std": _cr_std,
            # The embedding -> block 0 update, on its own.
            "embedding_relative_change": emb_rel,
            "_meta_embedding_contraction_rate": emb_rate,
            "_meta_embedding_transition_available": bool(has_embedding),
            # Which rows were scored, and how much of the middle layer's centred
            # sum of squares sits in position 0 / in massive non-initial rows.
            **row_meta.as_meta(),
            "_meta_position0_share": position0_share(mid),
            "_meta_massive_token_share": massive_token_share(mid),
        }
