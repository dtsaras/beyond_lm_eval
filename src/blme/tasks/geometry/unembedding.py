from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings, get_final_norm, get_lm_head
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
        n_sample = self.config.get("num_samples", 2000)
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
            
        device = W_out.device
        W_out_np = W_out.float().cpu().numpy()
        
        # Get Input Embeddings (E_in)
        E_in = get_embeddings(model)
        is_tied = False
        if E_in is not None:
            E_in_np = E_in.float().cpu().numpy()
            if E_in_np.shape == W_out_np.shape:
                is_tied = np.allclose(W_out_np, E_in_np, atol=1e-5)
        
        # Effective Rank — canonical Roy-Vetterli on σ² (shared helper).
        # The unembedding matrix is (V, D) with V up to ~200 k on
        # modern models; a CPU SVD there was the single slowest call
        # in the library (~70 s on Qwen-2B). Run it on GPU when
        # available — ~100× faster on 3090-class hardware.
        try:
            W_dev = W_out.detach().float()
            if torch.cuda.is_available() and not W_dev.is_cuda:
                W_dev = W_dev.cuda()
            W_centered_t = W_dev - W_dev.mean(dim=0, keepdim=True)
            S_t = torch.linalg.svdvals(W_centered_t)
            S = S_t.detach().cpu().numpy()
            from .utils import effective_rank as _effective_rank
            eff_rank = _effective_rank(S)
            del W_dev, W_centered_t, S_t
        except Exception as _e:
            # Historic bug: returning 0.0 silently masked SVD failures
            # (GPU OOM on qwen-9b/27b) so the column looked like
            # "collapsed to rank-0" rather than "couldn't be computed".
            logger.info(f"  unembedding SVD failed: {type(_e).__name__}: {_e}")
            eff_rank = float("nan")
            
        # Category Purity. For each labelled token, measure the fraction
        # of its top-k nearest unembedding rows that belong to the same
        # category. Historic bug: the previous implementation randomly
        # sampled ids from the full vocab before intersecting with the
        # label set — with a 50 k-vocab and ~200 labels, the sample
        # almost never hit a labelled token and purity collapsed to 0.
        # We now iterate the labelled tokens directly.
        if not categories_path:
            candidate = os.path.join(
                os.path.dirname(__file__),
                "../../assets/categories.json",
            )
            if os.path.exists(candidate):
                categories_path = candidate

        cat_labels: dict[int, str] = {}
        purity_mean = float("nan")
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
                    W_norm = W_out_np / (
                        np.linalg.norm(W_out_np, axis=1, keepdims=True) + 1e-10
                    )

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
                        sims = W_norm @ W_norm[idx]
                        sims[idx] = -np.inf
                        top_k_idx = np.argpartition(sims, -k)[-k:]
                        match_count = sum(
                            1 for t in top_k_idx if cat_labels.get(int(t)) == my_cat
                        )
                        scores.append(match_count / k)

                    if scores:
                        purity_mean = float(np.mean(scores))
            except Exception as e:
                logger.info(f"Error computing purity: {e}")

        # Embedding alignment: per-token cosine similarity between the
        # input embedding row *after the final layer norm* and the
        # output (unembedding) row. Applying the final norm is essential
        # for tied architectures: without it, tied ``E_in == W_out``
        # trivially gives cosine = 1 for every token, and the field only
        # duplicates ``unembedding_is_tied``. With the final norm, the
        # transform breaks the tautology and the reported alignment
        # reflects how the norm reshapes the token representations — the
        # same convention Nostalgebraist's logit-lens uses.
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

                e_norms = np.linalg.norm(E_ref, axis=1, keepdims=True)
                w_norms = np.linalg.norm(W_out_np, axis=1, keepdims=True)
                safe_e = np.where(e_norms > 0, e_norms, 1.0)
                safe_w = np.where(w_norms > 0, w_norms, 1.0)
                cos_sims = np.sum(
                    (E_ref / safe_e) * (W_out_np / safe_w), axis=1
                )
                emb_alignment_mean = float(np.mean(cos_sims))
                emb_alignment_std = float(np.std(cos_sims))
                emb_high_alignment_frac = float(np.mean(cos_sims > 0.9))
            except Exception as e:
                logger.info(f"  embedding alignment failed: {e}")

        return {
            "unembedding_is_tied": is_tied,
            "unembedding_eff_rank": float(eff_rank),
            "unembedding_purity_mean": float(purity_mean),
            "unembedding_n_category_tokens": int(n_category_tokens),
            "embedding_alignment_mean": emb_alignment_mean,
            "embedding_alignment_std": emb_alignment_std,
            "embedding_high_alignment_frac": emb_high_alignment_frac,
        }
