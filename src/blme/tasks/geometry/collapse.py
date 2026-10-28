from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .._tokens import (hidden_states_for_task, massive_token_share,
                       middle_depth_layers, position0_share, select_rows,
                       sink_keep_masks, sink_meta)
from .utils import collected_layer_count, effective_rank, safe_svdvals
import numpy as np
from tqdm import tqdm
import logging
logger = logging.getLogger("blme")


def _qdl_entropy(Z) -> float:
    """Queipo-de-Llano et al. 2025 Eq. (1) for one sequence's (T, d) states.

    H(X) = -sum_j p_j log p_j with p_j = s_j^2 / ||X||_F^2 over the singular
    values of the UNCENTRED matrix, every row kept (row 0 is the BOS/first
    token whose massive norm produces the compression valley, Theorem 1).
    NaN for fewer than two rows or a zero matrix.
    """
    X = np.asarray(Z.float().numpy() if hasattr(Z, "numpy") else Z, dtype=np.float64)
    X = X[np.all(np.isfinite(X), axis=1)]
    if X.shape[0] < 2:
        return float("nan")
    try:
        S = np.asarray(safe_svdvals(X), dtype=np.float64)
    except Exception:
        return float("nan")
    # p_j is scale-free: divide by s_1 before squaring.
    if not (S.size and S[0] > 0 and np.isfinite(S[0])):
        return float("nan")
    s2 = (S / S[0]) ** 2
    p = s2 / s2.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _rv_profile(clouds, layers):
    """Roy-Vetterli erank (raw sigma) and the sigma^2 twin of each centred cloud."""
    erank, erank_sigma2 = [], []
    for idx in tqdm(layers, desc="Computing Effective Rank per Layer"):
        X = clouds[idx].float().numpy()
        # Filter NaN/Inf rows (fp16 models may produce extreme values)
        finite_mask = np.all(np.isfinite(X), axis=1)
        if not np.all(finite_mask):
            X = X[finite_mask]
        if len(X) < 5:
            # NaN, counted in _meta_n_failed_layers.
            erank.append(float("nan"))
            erank_sigma2.append(float("nan"))
            continue
        # Centre in float64 (float32 overflows near the top of the range)
        # and take a max-factored SVD (tests/test_scale_invariance.py).
        X = np.asarray(X, dtype=np.float64)
        X = X - np.mean(X, axis=0)
        try:
            S = safe_svdvals(X)
        except Exception:
            erank.append(float("nan"))
            erank_sigma2.append(float("nan"))
            continue
        # Roy & Vetterli Eq. (1) on RAW singular values; the sigma^2
        # spectrum is a separate key, not their erank.
        erank.append(effective_rank(S))
        erank_sigma2.append(effective_rank(np.asarray(S, dtype=np.float64) ** 2))
    return erank, erank_sigma2


@register_task("geometry_collapse")
class RepresentationCollapseTask(DiagnosticTask):
    """
    Detects representation collapse by tracking Effective Rank across layers.
    A sharp drop in effective rank indicates dimensional collapse.

    References:
      * Jing, Vincent, LeCun, Tian 2021 — "Understanding Dimensional
        Collapse in Contrastive Self-supervised Learning", ICLR 2022,
        arXiv:2110.09348. Motivating paper for collapse-ratio / erank
        diagnostics.
      * Roy, Vetterli 2007 — "The Effective Rank: A Measure of Effective
        Dimensionality", European Signal Processing Conference. The
        exp(Shannon-entropy-of-normalised-singular-values) erank formula
        this task reports under ``erank_per_layer``.
      * Queipo-de-Llano, Arroyo, Barbero, Dong, Bronstein, LeCun,
        Shwartz-Ziv 2025 —
        "Attention Sinks and Compression Valleys in LLMs are Two Sides
        of the Same Coin", arXiv:2510.06477. The erank-vs-depth profile
        is a BLME collapse/valley proxy, not the full paper pipeline.
    """
    # Jing / Roy-Vetterli keys (non-LLM spectra of a pooled cloud): position 0
    # and massive rows dropped, else the sink is the "valley" they detect.
    TOKEN_POLICY = "exclude_sinks"
    # The Queipo-de-Llano family keeps every token, as the paper does: the
    # BOS/first-token row is the cause of the valley (Theorem 1).
    QDL_TOKEN_POLICY = ("paper_protocol:QueipoDeLlano2025 Eq. 1, per sequence, "
                        "uncentred, position 0 included")

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Representation Collapse Detection...")

        if dataset is None:
            dataset = [
                {"text": "The quick brown fox jumps over the lazy dog."}
                for _ in range(50)
            ]

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)

        # Block outputs 0..L-1 with the deepest entry the hooked raw block
        # (replace_final_norm); `last_is_block` says whether the swap happened.
        per_doc, _info = hidden_states_for_task(
            model, tokenizer, dataset, cache, num_samples=num_samples,
            replace_final_norm=True, layer_idx="all", use_cache=use_cache,
            per_sample=True)
        _last_is_block = bool(_info["last_is_block"])
        layers = sorted(per_doc.keys())
        layer_activations = {li: select_rows(per_doc[li], "all")[0] for li in layers}

        # Jing / Roy-Vetterli family: TOKEN_POLICY rows, one keep mask per
        # document from the middle depths, the same rows at every layer.
        _mid_keys = middle_depth_layers(per_doc)
        _keep, _sink_info = sink_keep_masks([per_doc[k] for k in _mid_keys])
        _rows_meta = None
        layer_activations = {}
        for li in layers:
            layer_activations[li], _rows_meta = select_rows(
                per_doc[li], self.TOKEN_POLICY, keep=_keep)

        # From the collected depth, not the config: a config without
        # num_hidden_layers would otherwise trim a hookless profile to 2 layers.
        _n_block_layers = collected_layer_count(model, n_collected=len(layers))
        erank_per_layer, erank_raw_per_layer = _rv_profile(layer_activations, layers)
        # Every-token profile (the pre-sink-policy values), kept as _meta_.
        erank_all, _ = _rv_profile(
            {li: select_rows(per_doc[li], "all")[0] for li in layers}, layers)
        _all = np.asarray(erank_all, dtype=np.float64)
        _all_ok = np.isfinite(_all)
        max_erank_all = float(np.max(_all[_all_ok])) if _all_ok.any() else float("nan")
        min_erank_all = float(np.min(_all[_all_ok])) if _all_ok.any() else float("nan")

        # Detect collapse: ratio of min erank to max erank
        erank_arr = np.array(erank_per_layer, dtype=np.float64)
        n_failed = int(np.isnan(erank_arr).sum())
        max_erank = float(np.nanmax(erank_arr)) if n_failed < len(erank_arr) else float("nan")
        min_erank = float(np.nanmin(erank_arr)) if n_failed < len(erank_arr) else float("nan")
        # Exact ratio: only a zero maximum is undefined.
        collapse_ratio = (min_erank / max_erank if max_erank > 0
                          else float("nan"))

        # Largest single-layer drop, over block-to-block transitions only: a
        # post-norm deepest entry is not a block, so it is trimmed here.
        if _last_is_block:
            _prof_for_drop = erank_arr
        else:
            _prof_for_drop = (erank_arr[:_n_block_layers]
                              if _n_block_layers and len(erank_arr) > _n_block_layers
                              else erank_arr)
        diffs = np.diff(_prof_for_drop)
        n_finite_diffs = int(np.count_nonzero(np.isfinite(diffs)))
        if n_finite_diffs > 0:
            max_drop = float(np.nanmin(diffs))
            max_drop_layer = int(np.nanargmin(diffs)) + 1
        else:
            max_drop = float("nan")
            max_drop_layer = -1

        d_model = int(layer_activations[layers[0]].shape[-1])
        _mid_docs = per_doc[layers[len(layers) // 2]]
        max_drop_frac = max_drop / d_model if d_model else float("nan")

        # Queipo-de-Llano et al. 2025 Eq. (1): per sequence, uncentred, every
        # token (QDL_TOKEN_POLICY), averaged over sequences at each layer; the
        # compression valley is the profile minimum.
        _qpol = ("all" if self.QDL_TOKEN_POLICY.startswith("paper_protocol")
                 else self.QDL_TOKEN_POLICY)
        qdl_profile, qdl_n_docs = [], 0
        for idx in layers:
            rows = [select_rows([Z], _qpol)[0] for Z in per_doc[idx]]
            hs = [h for h in (_qdl_entropy(Z) for Z in rows) if np.isfinite(h)]
            qdl_n_docs = max(qdl_n_docs, len(hs))
            qdl_profile.append(float(np.mean(hs)) if hs else float("nan"))
        _qdl = np.asarray(qdl_profile, dtype=np.float64)
        _qdl_ok = np.isfinite(_qdl)
        qdl_min = float(np.min(_qdl[_qdl_ok])) if _qdl_ok.any() else float("nan")
        qdl_argmin = int(np.nanargmin(_qdl)) if _qdl_ok.any() else -1

        return {
            # Queipo-de-Llano Eq. (1) valley depth: the headline of this family.
            "qdl_entropy_min": qdl_min,
            "_meta_qdl_entropy_per_layer": qdl_profile,
            "_meta_qdl_entropy_min_layer_frac": (
                qdl_argmin / (len(layers) - 1) if qdl_argmin >= 0 and len(layers) > 1
                else float("nan")),
            "_meta_qdl_n_docs": int(qdl_n_docs),
            "erank_per_layer": erank_per_layer,
            "max_erank": max_erank,
            "min_erank": min_erank,
            "collapse_ratio": collapse_ratio,
            "max_drop": max_drop,
            # /d_model forms are _meta_: zero-padding leaves erank unchanged
            # while these move by the padding factor.
            "_meta_max_drop_frac": max_drop_frac,
            "_meta_max_erank_frac": max_erank / d_model if d_model else float("nan"),
            "_meta_min_erank_frac": min_erank / d_model if d_model else float("nan"),
            "erank_sigma2_per_layer": erank_raw_per_layer,
            "_meta_erank_per_layer_frac": [
                (float(v) / d_model if (d_model and np.isfinite(v)) else float("nan"))
                for v in erank_arr
            ],
            "_meta_max_drop_layer": max_drop_layer,
            # Profile entries the drop scan actually covered.
            "_meta_max_drop_n_layers": int(len(_prof_for_drop)),
            "_meta_n_failed_layers": n_failed,
            "_meta_last_entry_is_block_output": bool(_last_is_block),
            # Every-token Roy-Vetterli profile, before the sink policy.
            "_meta_erank_per_layer_all_tokens": erank_all,
            "_meta_max_erank_all_tokens": max_erank_all,
            "_meta_min_erank_all_tokens": min_erank_all,
            "_meta_collapse_ratio_all_tokens": (
                min_erank_all / max_erank_all if max_erank_all > 0 else float("nan")),
            **sink_meta(_sink_info),
            **(_rows_meta.as_meta() if _rows_meta is not None else {}),
            "_meta_massive_token_share": massive_token_share(_mid_docs),
            "_meta_position0_share": position0_share(_mid_docs),
            # On a monotonically increasing profile max_drop is the smallest rise.
            "_meta_max_drop_is_actually_increase": bool(
                np.isfinite(max_drop) and max_drop > 0),
        }
