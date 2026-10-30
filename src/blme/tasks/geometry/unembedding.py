from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings, get_final_norm, get_lm_head
from .utils import effective_rank as _effective_rank, safe_svdvals, safe_unit_rows
import torch
import numpy as np
import json
import os
import logging
logger = logging.getLogger("blme")


def _load_category_labels(categories, tokenizer):
    """Map token-id → category-name for every entry in the bundled file.

    ``categories`` may contain either flat lists of strings (``pronouns``,
    ``colors``, …) or list-of-pair entries (``singular_plural``,
    ``present_past``). We walk nested lists so every leaf string is
    considered. Each word is tried both with and without a leading space
    because SentencePiece-style tokenizers encode ``"apple"`` vs
    ``" apple"`` as distinct tokens.
    """
    cat_labels: dict[int, str] = {}
    if not hasattr(tokenizer, "encode"):
        return cat_labels

    def _iter_words(obj):
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, (list, tuple)):
            for sub in obj:
                yield from _iter_words(sub)

    for cat, entries in categories.items():
        for word in _iter_words(entries):
            for variant in (word, f" {word}"):
                try:
                    ids = tokenizer.encode(variant, add_special_tokens=False)
                except Exception:
                    continue
                if ids and len(ids) == 1:
                    cat_labels.setdefault(int(ids[0]), cat)
    return cat_labels

@register_task("geometry_unembedding")
class UnembeddingDiagnosticsTask(DiagnosticTask):
    """
    Analyzes the output embedding matrix (unembedding).
    Checks if weights are tied and measures effective rank and category purity of the unembedding space.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Unembedding Diagnostics...")
        # Purity query cap, decoupled from the harness-wide num_samples (a
        # 24-document run used to subsample the queries to 24). None = every
        # labelled token.
        n_sample = self.config.get("purity_num_queries", None)
        k = self.config.get("k", 20)
        categories_path = self.config.get("categories_path", None)
        
        # Get Output Embeddings (W_out)
        head = get_lm_head(model)
        if head is not None:
            W_out = head.weight.detach()
        else:
            # If no head, try input embeddings (tied weights)
            W_out = get_embeddings(model)
            if W_out is None:
                return {"error": "Could not find lm_head or output embeddings"}
            
        W_out_np = W_out.float().cpu().numpy()
        
        # Get Input Embeddings (E_in)
        E_in = get_embeddings(model)
        is_tied = False
        if E_in is not None:
            E_in_np = E_in.float().cpu().numpy()
            if E_in_np.shape == W_out_np.shape:
                is_tied = np.allclose(W_out_np, E_in_np, atol=1e-5)
        
        # Roy-Vetterli effective rank of the RAW matrix (their definition; a
        # centred copy is a separate key). The (V, d) SVD runs on a GPU when
        # one is present; safe_svdvals is exact near the float32 maximum.
        try:
            W_dev = W_out.detach().float()
            if torch.cuda.is_available() and not W_dev.is_cuda:
                W_dev = W_dev.cuda()
            S_t = safe_svdvals(W_dev)
            S = S_t.detach().cpu().numpy()
            eff_rank = _effective_rank(S)
            # Centred variant, centred in double (float32 overflows near the
            # top of the range).
            _Wc = W_dev.double()
            _Wc = _Wc - _Wc.mean(dim=0, keepdim=True)
            eff_rank_centered = _effective_rank(
                safe_svdvals(_Wc).detach().cpu().numpy())
            del W_dev, _Wc, S_t
        except Exception as _e:
            # NaN, not 0.0, for a failed SVD (e.g. GPU OOM); every other key
            # is still computed (tests/tasks/test_unembedding_svd_failure.py).
            logger.info(f"  unembedding SVD failed: {type(_e).__name__}: {_e}")
            eff_rank = float("nan")
            eff_rank_centered = float("nan")
            
        # Category purity: for each labelled token, the fraction of its top-k
        # nearest unembedding rows in the same category (labelled tokens are
        # iterated directly).
        if not categories_path:
            candidate = os.path.join(
                os.path.dirname(__file__),
                "../../assets/categories.json",
            )
            if os.path.exists(candidate):
                categories_path = candidate

        cat_labels: dict[int, str] = {}
        purity_mean = float("nan")
        n_queries = 0
        n_category_tokens = 0
        if (
            categories_path and os.path.exists(categories_path)
            and tokenizer is not None and hasattr(tokenizer, "encode")
        ):
            try:
                with open(categories_path, "r") as f:
                    categories = json.load(f)
                cat_labels = _load_category_labels(categories, tokenizer)
                n_category_tokens = len(cat_labels)

                if cat_labels:
                    # Exact unit rows at any scale (no norm epsilon).
                    W_norm = safe_unit_rows(W_out_np)

                    # Optionally subsample labelled tokens to keep very
                    # large label sets tractable. Deterministic seed so
                    # runs are reproducible across models.
                    query_ids = sorted(cat_labels.keys())
                    if n_sample and n_sample < len(query_ids):
                        rng = np.random.default_rng(42)
                        query_ids = sorted(
                            rng.choice(query_ids, size=n_sample, replace=False)
                        )

                    scores = []
                    for idx in query_ids:
                        if idx >= W_norm.shape[0]:
                            continue
                        my_cat = cat_labels[idx]
                        # Squared chord distances (exact; ranks like cosine on
                        # unit rows) and a stable tie rule.
                        _d2 = ((W_norm - W_norm[idx]) ** 2).sum(axis=1)
                        _d2[idx] = np.inf
                        top_k_idx = np.lexsort(
                            (np.arange(len(_d2)), _d2))[:k]
                        match_count = sum(
                            1 for t in top_k_idx if cat_labels.get(int(t)) == my_cat
                        )
                        scores.append(match_count / k)

                    n_queries = len(scores)
                    if scores:
                        purity_mean = float(np.mean(scores))
            except Exception as e:
                logger.info(f"Error computing purity: {e}")

        # Embedding alignment: cosine between each input embedding row AFTER
        # the final norm and its unembedding row (the logit-lens convention;
        # without the norm a tied model reads exactly 1).
        emb_alignment_mean = float("nan")
        emb_alignment_std = float("nan")
        emb_high_alignment_frac = float("nan")
        if E_in is not None and E_in_np.shape == W_out_np.shape:
            final_norm = get_final_norm(model)
            try:
                E_ref = E_in.detach().float()
                if final_norm is not None:
                    # Run every embedding row through the final norm so
                    # the comparison happens in the same space the LM
                    # head sees at inference time.
                    with torch.no_grad():
                        E_ref = final_norm(E_ref.to(next(final_norm.parameters()).dtype))
                    E_ref = E_ref.detach().float().cpu().numpy()
                else:
                    E_ref = E_ref.cpu().numpy()

                cos_sims = np.sum(
                    safe_unit_rows(E_ref) * safe_unit_rows(W_out_np), axis=1
                )
                emb_alignment_mean = float(np.mean(cos_sims))
                emb_alignment_std = float(np.std(cos_sims))
                emb_high_alignment_frac = float(np.mean(cos_sims > 0.9))
            except Exception as e:
                logger.info(f"  embedding alignment failed: {e}")

        return {
            "unembedding_is_tied": is_tied,
            "unembedding_eff_rank": float(eff_rank),
            "unembedding_eff_rank_centered": float(eff_rank_centered),
            # _meta_: the task itself flags purity as not comparable across
            # models (which words map to single tokens is a tokenizer property).
            "_meta_unembedding_purity_mean": float(purity_mean),
            "_meta_purity_n_queries": int(n_queries),
            # A tokenizer property (single-token category words), not geometry.
            "_meta_unembedding_n_category_tokens": int(n_category_tokens),
            # Describes the purity field only.
            "_meta_purity_cross_model_comparable": (
                False if n_category_tokens > 0 else True),
            "_meta_comparability_note": (
                "tokenizer_dependent_category_token_mapping"
                if n_category_tokens > 0
                else "intrinsic_unembedding_geometry"
            ),
            "embedding_alignment_mean": emb_alignment_mean,
            "embedding_alignment_std": emb_alignment_std,
            "embedding_high_alignment_frac": emb_high_alignment_frac,
        }
