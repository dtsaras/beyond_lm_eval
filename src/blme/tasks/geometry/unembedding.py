from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings, get_final_norm, get_lm_head
from .utils import safe_unit_rows
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
        
        # Effective Rank — Roy-Vetterli on raw singular values (shared helper).
        # The unembedding matrix is (V, D) with V up to ~200 k on
        # modern models; a CPU SVD there was the single slowest call
        # in the library (~70 s on Qwen-2B). Run it on GPU when
        # available — ~100× faster on 3090-class hardware.
        try:
            W_dev = W_out.detach().float()
            if torch.cuda.is_available() and not W_dev.is_cuda:
                W_dev = W_dev.cuda()
            # Roy & Vetterli's effective rank is defined on the matrix ITSELF,
            # not on a centered copy. Centering `W_out` here is BLME's own
            # addition and it is not a harmless one: against the paper's closed
            # form (Sec. 2.3, the 4x4 circulant example) the centered value is
            # **-11.4% at rho = 0.5 and +48.8% at rho = 0.9** -- the error
            # changes SIGN with the matrix -- and on real unembedding matrices
            # it ranges 0.5% to 57.9% (pythia-70m 296.11 -> 467.53) and
            # REORDERS models (adversarial review, 2026-09-03).
            #
            # The shared `effective_rank` helper is exact (<= 3e-16 against the
            # closed form); the defect was here.
            # `safe_svdvals`: numpy/torch compute the SVD internally and then CAST
            # the singular values back to the input dtype, and that cast overflows
            # to inf near 3.4e38 -- so a scale-FREE consumer downstream (effective
            # rank, RankMe, IsoScore, alpha) returns NaN on data it should handle
            # exactly. The helper factors out the max first, which only shifts
            # exponents (2026-09-07).
            from .utils import safe_svdvals
            S_t = safe_svdvals(W_dev)
            S = S_t.detach().cpu().numpy()
            from .utils import effective_rank as _effective_rank
            eff_rank = _effective_rank(S)
            # Centered variant retained under a name that says what it is.
            # CENTRE IN DOUBLE. The raw effective rank is already exact, but
            # the optional centred variant erased it: an exact raw
            # 2.600490005990 became NaN, and the centred auxiliary returned NaN
            # instead of 2.729838563104 at 2.406159650500376e38 (gpt-5.6-sol
            # round 9, 2026-09-07). `W_dev - mean` in the model's dtype
            # overflows near the top of the range -- centred entries of a
            # 2.4e38 matrix exceed 3.4e38 -- so `safe_svdvals` (correctly)
            # refuses a non-finite matrix. Exactly the defect found in
            # `collapse` today, where wiring `safe_svdvals` did not help because
            # the overflow was in the line ABOVE the SVD.
            _Wc = W_dev.double()
            _Wc = _Wc - _Wc.mean(dim=0, keepdim=True)
            eff_rank_centered = _effective_rank(
                # `safe_svdvals`: numpy/torch compute the SVD internally and then CAST
                # the singular values back to the input dtype, and that cast overflows
                # to inf near 3.4e38 -- so a scale-FREE consumer downstream (effective
                # rank, RankMe, IsoScore, alpha) returns NaN on data it should handle
                # exactly. The helper factors out the max first, which only shifts
                # exponents (2026-09-07).
                safe_svdvals(_Wc).detach().cpu().numpy())
            del W_dev, _Wc, S_t
        except Exception as _e:
            # Historic bug: returning 0.0 silently masked SVD failures
            # (GPU OOM on qwen-9b/27b) so the column looked like
            # "collapsed to rank-0" rather than "couldn't be computed".
            logger.info(f"  unembedding SVD failed: {type(_e).__name__}: {_e}")
            eff_rank = float("nan")
            # The 2026-09-03 fix added `unembedding_eff_rank_centered` to the
            # return dict but bound the variable ONLY inside the `try`, so
            # this handler -- written specifically for the OOM the comment
            # above names -- raised `UnboundLocalError` on the way out and
            # `core.py` recorded the WHOLE task as failed. Tying, purity,
            # category count and embedding alignment were all lost along with
            # the one number that actually could not be computed. Reproduced
            # by forcing `svdvals` to raise `torch.cuda.OutOfMemoryError`
            # (red-team review, 2026-09-05); pinned by
            # `tests/tasks/test_unembedding_svd_failure.py`.
            eff_rank_centered = float("nan")
            
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
                    # `safe_unit_rows`, not `norm + 1e-10`.
                    #
                    # Adding to the norm is not normalisation: it shrinks small
                    # rows toward the origin and changes the neighbour geometry
                    # -- the exact anti-pattern `hubness.py` documents and that
                    # `safe_unit_rows` exists to remove. It also underflows,
                    # because the norm SQUARES each coordinate long before the
                    # data itself is denormal. This site was missed when the
                    # rest of the campaign moved to the helper (found by an AST
                    # sweep for the class, 2026-09-07).
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
                        # DIRECT chord distances, and a STABLE tie rule.
                        #
                        # Two defects in the previous `argpartition` on
                        # `W_norm @ W_norm[idx]`:
                        #  1. cancellation -- with unit rows every dot product
                        #     sits near 1 and the separating quantity 1 - u.v
                        #     is the difference of two ~1.0 sums, which float32
                        #     cannot resolve on an anisotropic vocabulary. This
                        #     is the same defect confirmed in `hubness.py` on
                        #     2026-09-07; found here by sweeping the class.
                        #  2. `argpartition` leaves ties in arbitrary order, so
                        #     a tied k-th neighbour made the score depend on
                        #     array order -- the same flaw `hubness.py` removed
                        #     when it dropped `topk` for a stable argsort.
                        # Ranking by squared chord distance is equivalent for
                        # unit rows (||u-v||^2 = 2 - 2u.v) but differences the
                        # coordinates before squaring, which is exact.
                        _d2 = ((W_norm - W_norm[idx]) ** 2).sum(axis=1)
                        _d2[idx] = np.inf
                        top_k_idx = np.lexsort(
                            (np.arange(len(_d2)), _d2))[:k]
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

                # `safe_unit_rows`, not `np.where(norm > 0, norm, 1.0)`.
                #
                # `np.linalg.norm` SQUARES before summing, so at 1e-25 it
                # underflows to exactly 0, the guard divides the row by ONE, and
                # the "unit" vectors are the raw tiny rows. A TIED matrix whose
                # cosine is exactly 1.0 by construction returned **0.0 at 1e-25
                # and 1e25** (gpt-5.6-sol round 9, 2026-09-07). The guard reads
                # as a zero-row guard but fires on rows that are merely small --
                # the identical defect fixed in `contextualization` today.
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
            # Was computed and then DISCARDED, so the promised key did not
            # exist and a second full (V, d) SVD was thrown away every run.
            "unembedding_eff_rank_centered": float(eff_rank_centered),
            "unembedding_purity_mean": float(purity_mean),
            # `_meta_`: a VOCABULARY PROXY. It counts how many of the 21
            # category words the tokenizer happens to render as single tokens
            # (815 on gpt2), so it varies with the tokenizer and not with the
            # model's representations -- and it was reaching the feature
            # matrix unprefixed (red-team review, 2026-09-05).
            "_meta_unembedding_n_category_tokens": int(n_category_tokens),
            # `_meta_` and RENAMED. As `cross_model_comparable` this read as a
            # statement about the whole result dict; it is not. It describes
            # the PURITY field only -- `unembedding_eff_rank` (rho +0.921 with
            # d_model) is precisely the field it does NOT cover. It is also
            # constant `False` on every model that has any category token, so
            # unprefixed it contributed a zero-variance column.
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
