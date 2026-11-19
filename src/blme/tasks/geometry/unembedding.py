from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings, get_final_norm, get_lm_head
from .utils import effective_rank as _effective_rank, gpu_linalg_device, knn_sorted_batched, safe_svdvals, safe_unit_rows
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
    TOKEN_POLICY = "not_token_based"
    SOURCES = {
        "paper": "doi:10.5281/zenodo.40328 (Roy & Vetterli 2007, Eq. 1, Def. 1: effective rank)",
        "code": "none (category purity and embedding alignment are BLME diagnostics)",
        "llm_precedent": "none verified for the unembedding (the logit-lens convention "
                         "of reading rows through the final norm is followed)",
    }
    ADAPTATION_NOTE = (
        "The unembedding is summarised by the Roy-Vetterli effective rank of its raw rows, "
        "a centred variant, a category-purity score (the share of each category word's 20 "
        "nearest unembedding rows in its category) and the cosine between each row and its "
        "final-normed input embedding. No published LLM protocol defines these; they "
        "describe how the output vocabulary is laid out. The raw erank follows Roy and "
        "Vetterli exactly; centring is not in their definition and stays a separate key. "
        "Purity depends on which words the tokenizer maps to single tokens, so it is a "
        "within-tokenizer _meta_ diagnostic, scored on every labelled token unless "
        "purity_num_queries caps it. On a tied model the alignment compares a row with its "
        "own final-normed copy, so it measures the final norm. The erank reads only the "
        "rows the tokenizer can emit (ids below len(tokenizer)), so padded rows no token "
        "reaches do not enter it, and the all-rows values are kept as _meta_; those rows "
        "still enter the alignment and move it slightly."
    )
    HEADLINE_KEYS = ("unembedding_eff_rank", "unembedding_eff_rank_centered",
)
    # Width: zero-padding the hidden dimension of W_out keeps its spectrum. V:
    # rows past len(tokenizer) are never read. The alignment keys have no
    # size-free form (the final norm re-centres over padded coordinates).
    SIZE_INVARIANT = {k: (("zero_pad", "pad_vocabulary"),
                          "tests/tasks/invariance/test_unembedding.py::"
                          f"test_size_invariant_entry[{k}]")
                      for k in ("unembedding_eff_rank", "unembedding_eff_rank_centered")}
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
        # centred copy is a separate key) over the emittable rows only: padded
        # rows past len(tokenizer) are never produced
        # (tests/tasks/invariance/test_unembedding.py::test_padded_rows_do_not_enter_the_erank).
        # The (V, d) SVD runs on a GPU when one is present; safe_svdvals is
        # exact near the float32 maximum.
        from .categories import _n_emittable
        n_emit = _n_emittable(tokenizer, W_out.shape[0])

        def _eranks(W):
            S_t = safe_svdvals(W)
            raw = _effective_rank(S_t.detach().cpu().numpy())
            # Centred variant, centred in double (float32 overflows near the
            # top of the range).
            _Wc = W.double()
            _Wc = _Wc - _Wc.mean(dim=0, keepdim=True)
            return raw, _effective_rank(safe_svdvals(_Wc).detach().cpu().numpy())

        def _all_eranks(device):
            W_dev = W_out.detach().float().to(device)
            out = _eranks(W_dev[:n_emit])
            out_all = out if n_emit == W_dev.shape[0] else _eranks(W_dev)
            del W_dev
            return out + out_all

        svd_error = None
        devices = (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]
        for _dev in devices:
            try:
                (eff_rank, eff_rank_centered,
                 eff_rank_all, eff_rank_centered_all) = _all_eranks(_dev)
                svd_error = None
                break
            except Exception as _e:
                # A float64 SVD of a (V, d) table can exceed GPU memory (the
                # Qwen3-1.7B GPU smoke run, 2026-11-16): retry on the CPU.
                svd_error = f"{_dev}: {type(_e).__name__}: {_e}"
                logger.info(f"  unembedding SVD failed on {svd_error}")
                if _dev == "cuda":
                    torch.cuda.empty_cache()
        if svd_error is not None:
            # NaN with its reason, not 0.0; every other key is still computed
            # (tests/tasks/test_unembedding_svd_failure.py).
            eff_rank = eff_rank_centered = float("nan")
            eff_rank_all = eff_rank_centered_all = float("nan")
            
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
                    _q = [i for i in query_ids if i < W_norm.shape[0]]
                    if gpu_linalg_device() is not None and _q:
                        # All queries at once on the GPU (float64 chord distances,
                        # same stable index tie-break as the CPU loop below).
                        for idx, top_k_idx in zip(_q, knn_sorted_batched(W_norm, _q, k)):
                            my_cat = cat_labels[idx]
                            scores.append(sum(1 for t in top_k_idx if cat_labels.get(int(t)) == my_cat) / k)
                        query_ids = []
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
            # Every row of W_out, padded rows included (the previous headline).
            "_meta_unembedding_eff_rank_all_rows": float(eff_rank_all),
            "_meta_unembedding_eff_rank_centered_all_rows": float(eff_rank_centered_all),
            "_meta_n_unused_rows": int(W_out.shape[0] - n_emit),
            "_meta_unembedding_svd_error": svd_error,
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
            "_meta_embedding_alignment_mean": emb_alignment_mean,
            "_meta_embedding_alignment_std": emb_alignment_std,
            "_meta_embedding_high_alignment_frac": emb_high_alignment_frac,
        }
