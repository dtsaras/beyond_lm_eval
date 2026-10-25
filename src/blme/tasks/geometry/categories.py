from ...tasks.base import DiagnosticTask
from ...registry import register_task
import torch
import numpy as np
import json
import os
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import skew
from typing import Dict, List, Tuple
import logging
import hashlib
from .utils import safe_unit_rows as _safe_unit_rows
logger = logging.getLogger("blme")

# Category size used by the MATCHED purity variant. Every model is compared at
# this many tokens per category and this many neighbours, so neither k nor the
# category size can differ across tokenizers.
_MATCHED_CAT_SIZE = 10


@register_task("geometry_categories")
class CategoryGeometryTask(DiagnosticTask):
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Category Geometry Analysis...")
        
        # 1. Load Categories
        categories_path = self.config.get("categories_path")
        if not categories_path:
            # Default to packaged asset
            import blme
            package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(blme.__file__))))
            # This logic depends on where blme is installed. 
            # Better to use pkg_resources or importlib.resources, but simple relative path works for source install
            # src/blme/assets/categories.json
            asset_path = os.path.join(os.path.dirname(blme.__file__), "assets", "categories.json")
            categories_path = asset_path
            
        with open(categories_path, 'r') as f:
            categories = json.load(f)

        # 2. Get Embeddings
        from ..common import get_embeddings as _get_emb
        E_tensor = _get_emb(model)
        if E_tensor is None:
            return {"error": "Could not access embeddings"}
        E = E_tensor.float().cpu().numpy()
             
        # 3. Map Categories to Tokens
        cat_tokens, cat_labels = self._get_category_tokens(tokenizer, categories)
        
        # 4. Compute Metrics
        results = {}
        
        # Separation
        logger.info(" Computing Separation...")
        sep_results = self._compute_separation(E, cat_tokens)
        results.update(sep_results)
        # Tokens claimed by more than one category, dropped from all of them.
        results["_meta_n_ambiguous_tokens"] = int(
            getattr(self, "_n_ambiguous_tokens", 0))
        
        # Purity
        k = self.config.get("k_purity", 20)
        logger.info(f" Computing Purity (k={k})...")
        purity_results = self._compute_purity(E, cat_tokens, cat_labels, k=k)
        results.update(purity_results)
        
        # Hubness
        logger.info(" Computing Category Hubness...")
        hub_results = self._compute_hubness(E, tokenizer, cat_tokens)
        results.update(hub_results)
        
        # Projection (UMAP/t-SNE)
        projection_method = self.config.get("projection_method", None) # "umap", "tsne", "pca"
        if projection_method:
            logger.info(f" Computing {projection_method.upper()} Projection...")
            proj_results = self._compute_projection(E, cat_tokens, method=projection_method)
            results.update(proj_results)

        # Relation Consistency (Singular/Plural, Present/Past)
        logger.info(" Computing Relation Consistency...")
        if hasattr(self, 'relation_pairs') and self.relation_pairs:
            rel_results = self._compute_relation_consistency(E, self.relation_pairs, tokenizer)
            results.update(rel_results)

        n_mapped = sum(len(tids) for tids in cat_tokens.values())
        results["n_category_tokens_mapped"] = int(n_mapped)
        # `_meta_`: a CONSTANT `False` emitted unprefixed alongside ~100
        # numeric keys, so it gated nothing in the analysis pipeline and
        # contributed a zero-variance feature column instead
        # (red-team review, 2026-09-05).
        results["_meta_cross_model_comparable"] = False
        results["_meta_comparability_note"] = (
            "tokenizer_dependent_category_token_mapping"
        )

        return results

    def _get_category_tokens(self, tokenizer, categories):
        cat_tokens = {}
        cat_labels = {}
        self.relation_pairs = {}
        
        for cat_name, items in categories.items():
            cat_tokens[cat_name] = []
            
            # Check if items are pairs (list of lists)
            is_relation = False
            if items and isinstance(items[0], list) and len(items[0]) == 2:
                is_relation = True
                self.relation_pairs[cat_name] = []
                
            flat_words = []
            if is_relation:
                for pair in items:
                    w1, w2 = pair
                    # Verify both are single tokens (or get their IDs)
                    ids1 = tokenizer.encode(w1, add_special_tokens=False)
                    ids2 = tokenizer.encode(w2, add_special_tokens=False)
                    
                    # Try with space prefix too
                    if len(ids1) != 1:
                        ids1_sp = tokenizer.encode(' ' + w1, add_special_tokens=False)
                        if len(ids1_sp) == 1: ids1 = ids1_sp
                    
                    if len(ids2) != 1:
                        ids2_sp = tokenizer.encode(' ' + w2, add_special_tokens=False)
                        if len(ids2_sp) == 1: ids2 = ids2_sp
                        
                    if len(ids1) == 1 and len(ids2) == 1:
                        tid1 = ids1[0]
                        tid2 = ids2[0]
                        self.relation_pairs[cat_name].append((tid1, tid2))
                        
                        # Add to flat collection for other metrics
                        if tid1 not in cat_labels:
                            cat_tokens[cat_name].append(tid1)
                            cat_labels[tid1] = cat_name
                        if tid2 not in cat_labels:
                            cat_tokens[cat_name].append(tid2)
                            cat_labels[tid2] = cat_name
            else:
                # Original logic for simple lists
                for word in items:
                    for w in [word, ' ' + word]:
                        ids = tokenizer.encode(w, add_special_tokens=False)
                        if len(ids) == 1:
                            tid = ids[0]
                            if tid not in cat_labels:
                                cat_tokens[cat_name].append(tid)
                                cat_labels[tid] = cat_name
                                
        # AMBIGUOUS TOKENS ARE DROPPED FROM EVERY CATEGORY, not assigned to
        # whichever one the JSON happened to list first.
        #
        # `if tid not in cat_labels` is a FIRST-WINS rule, so the result
        # depended on asset key order: 14 gpt2 token ids are claimed by two
        # categories, and swapping two keys in the taxonomy file moved
        # `pronouns_purity` -23.6%, `determiners_purity` +45.9% and
        # `pronouns_purity_matched` +120% (independent Opus review,
        # 2026-09-04). A token with two labels has no unambiguous label, so
        # purity for it is not defined at all; excluding it is the only answer
        # that does not depend on file order. The count is reported.
        # RELATION PAIRS ARE CLAIMS TOO.
        #
        # This census skipped any entry that was not a bare string, so a
        # relation category written as pairs -- plural = [[man, men],
        # [woman, women], ...] -- contributed NOTHING to it, and its tokens
        # were labelled first-wins instead of being dropped. The whole point of
        # the census is that the answer must not depend on asset key order, and
        # it did: on unchanged GPT-2 with people = [man, woman, child, person]
        # and that plural list, reversing the two keys moved plural purity
        # 0.083333333333 -> 0.357142857143 and people purity
        # 0.25 -> 0.083333333333. The ambiguous ids are 805, 6259, 8580, 9410,
        # so the correct counts are 4 ambiguous / 8 mapped; both orders emitted
        # 0/12. Dropping every multiply-claimed id gives 0.083333333333 in both
        # orders, which is the declared rule (codex round 15, 2026-09-09).
        _claims = {}
        for _c, _items in categories.items():
            for _w in ([] if not isinstance(_items, list) else _items):
                # Flatten one level so pair/tuple entries are counted.
                if isinstance(_w, str):
                    _words = [_w]
                elif isinstance(_w, (list, tuple)):
                    _words = [_x for _x in _w if isinstance(_x, str)]
                else:
                    _words = []
                for _ww in _words:
                    for _v in (_ww, " " + _ww):
                        _ids = tokenizer.encode(_v, add_special_tokens=False)
                        if len(_ids) == 1:
                            _claims.setdefault(_ids[0], set()).add(_c)
        _ambiguous = {t for t, cs in _claims.items() if len(cs) > 1}
        if _ambiguous:
            for _c in list(cat_tokens):
                cat_tokens[_c] = [t for t in cat_tokens[_c] if t not in _ambiguous]
            for t in _ambiguous:
                cat_labels.pop(t, None)
        self._n_ambiguous_tokens = int(len(_ambiguous))

        return cat_tokens, cat_labels

    def _compute_separation(self, E, cat_tokens):
        n_vocab = len(E)
        results = {}
        
        # Pre-select random tokens for inter-class comparison
        np.random.seed(42)
        all_tids = set([t for tokens in cat_tokens.values() for t in tokens])
        replacement_candidates = [t for t in range(n_vocab) if t not in all_tids]
        
        # If vocab is tiny (test mode), handle gracefully
        if not replacement_candidates:
             replacement_candidates = list(range(n_vocab))

        random_tids = np.random.choice(
            replacement_candidates,
            size=min(100, len(replacement_candidates)),
            replace=False
        )
        
        # SCALE-EXACT unit rows, then 1 - dot. `scipy.spatial.distance.cosine`
        # squares in the input dtype, so on float32 embeddings at magnitude
        # <= 1e-19 it returns exactly 0.0 -- and ALL 42 intra/inter/separation
        # values collapsed to 0.0 where the true cosine distance is 0.9587836
        # (independent Opus review, 2026-09-04). This was the last unfixed site
        # of the round-2 class in this file.
        _U = _safe_unit_rows(E)

        for cat_name, tids in cat_tokens.items():
            if len(tids) < 3: continue
            
            # Intra
            intra_dists = []
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    d = 1.0 - float(_U[tids[i]] @ _U[tids[j]])
                    intra_dists.append(d)
            intra = np.mean(intra_dists) if intra_dists else 0
            
            # Inter
            inter_dists = []
            for tid in tids:
                for rtid in random_tids[:10]:
                    d = 1.0 - float(_U[tid] @ _U[rtid])
                    inter_dists.append(d)
            inter = np.mean(inter_dists) if inter_dists else 0
            
            results[f"{cat_name}_intra"] = float(intra)
            results[f"{cat_name}_inter"] = float(inter)
            results[f"{cat_name}_separation"] = float(inter - intra)
            
        return results

    def _compute_purity(self, E, cat_tokens, cat_labels, k=20):
        # EXACT cosine normalisation -- divide by the norm, never add to it.
        # `norm + 1e-10` biases every small-but-valid row toward the origin and
        # breaks the scale invariance cosine has by definition. Measured: a
        # two-token category with row norms 1e-12/2e-12 returned purity 0.0
        # against an exact-cosine 1.0, and exactly parallel relation
        # differences returned 0.5625 at scale 1e-10 and 0.0008483363 at 1e-12
        # instead of 1.0 (adversarial review, 2026-09-04).
        E_norm = _safe_unit_rows(E)
        results = {}
        
        for cat_name, tids in cat_tokens.items():
            if len(tids) < 2: continue
            
            # ADAPTIVE k. With a fixed k, a category holding m tokens can
            # never exceed a purity of (m-1)/k, because at most m-1 neighbours
            # can share its label. That is a CEILING set by the tokenizer, not
            # by the geometry: the currencies category maps to 3-34 tokens
            # across 16 tokenizers, so its ceiling ranges 0.10 to 1.00 and two
            # models can differ 10x on this feature with identical geometry.
            # Verified on constructed ground truth: four perfectly separated
            # clusters score 1.0000 at k = 9 and 0.4500 at the shipped k = 20.
            #
            # Capping k at m-1 makes the ceiling exactly 1.0 for every category
            # and every tokenizer, so the number means the same thing everywhere.
            k_eff = max(1, min(int(k), len(tids) - 1))

            purities = []
            for tid in tids:
                sims = E_norm @ E_norm[tid]
                sims[tid] = -np.inf
                top_k = np.argsort(sims)[-k_eff:]

                same_cat = sum(1 for t in top_k if cat_labels.get(t) == cat_name)
                purities.append(same_cat / k_eff)

            results[f"{cat_name}_purity"] = float(np.mean(purities))

            # MATCHED variant: the same k AND the same category size on every
            # model. Capping k at m-1 fixes the CEILING at 1.0, but k_eff still
            # varies with the tokenizer, and that alone reverses real orderings
            # -- GPT-2 vs Llama digits read 0.542105 (k=19) against 0.900000
            # (k=9), while at a matched k=9 the same pair reads 0.916667
            # against 0.900000 (adversarial review, 2026-09-04). Subsampling
            # the category to a fixed size with a fixed seed matches m as well,
            # so nothing tokenizer-dependent is left in the comparison.
            _m_fix = _MATCHED_CAT_SIZE
            if len(tids) >= _m_fix:
                # DETERMINISTIC across processes. `hash()` on a str is salted
                # by PYTHONHASHSEED, so the subsample -- and therefore the
                # matched purity -- changed run to run on identical weights:
                # digits_purity_matched read 0.455556 / 0.444444 / 0.455556 /
                # 0.433333 for PYTHONHASHSEED 1..4 (adversarial review round 2,
                # 2026-09-04). blake2b is stable everywhere.
                _seed = int.from_bytes(
                    hashlib.blake2b(cat_name.encode("utf-8"),
                                    digest_size=4).digest(), "big")
                _rng = np.random.default_rng(_seed)
                _sub = sorted(_rng.choice(np.asarray(tids), size=_m_fix,
                                          replace=False).tolist())
                _sub_set = set(_sub)
                _k_m = _m_fix - 1
                _pm = []
                for tid in _sub:
                    sims = E_norm @ E_norm[tid]
                    sims[tid] = -np.inf
                    top_k = np.argsort(sims)[-_k_m:]
                    _pm.append(sum(1 for t in top_k if t in _sub_set) / _k_m)
                # `_meta_`: THIS TWIN FAILS THE CAMPAIGN'S OWN BAR, measured.
                #
                # A normalized twin must show a MEASURED REDUCTION in the
                # architecture association that does not merely relocate it.
                # Over 20 models (vocab 49k-262k, d_model 512-2048) the matched
                # form makes it WORSE: mean |spearman| with vocab_size 0.327
                # raw -> 0.386 matched, and with d_model 0.151 -> 0.241. Per
                # category the worst cases get much worse -- currencies -0.425
                # -> -0.789, units -0.248 -> -0.674, prepositions -0.313 ->
                # -0.511 (independent Opus review, 2026-09-05).
                #
                # The mechanism is understood: fixing m and k removes the
                # CEILING artifact, which was real, but the fixed-size
                # subsample is still drawn from a TOKENIZER-DEPENDENT pool, so
                # which 10 tokens get compared still varies by model. That is
                # the association relocating, not reducing. All 11 previously
                # registered twins were removed on exactly this ground; this
                # one is emitted as a diagnostic, not as a feature.
                results[f"_meta_{cat_name}_purity_matched"] = float(np.mean(_pm))
                results[f"_meta_{cat_name}_k_matched"] = int(_k_m)
            else:
                results[f"_meta_{cat_name}_purity_matched"] = float("nan")
            # The tokenizer-dependent quantities that set the ceiling, so a
            # reader can tell a small category from a poorly clustered one.
            results[f"_meta_{cat_name}_k_eff"] = int(k_eff)
            results[f"_meta_{cat_name}_n_tokens"] = int(len(tids))
            
        return results
        
    def _compute_hubness(self, E, tokenizer, cat_tokens, k_cat=5):
        E_norm = _safe_unit_rows(E)
        results = {}
        
        for cat_name, tids in cat_tokens.items():
            if len(tids) < 5: continue
            
            tids_arr = np.array(tids)
            E_cat = E_norm[tids_arr]
            k = min(k_cat, len(tids) - 1)
            
            n_occ = np.zeros(len(tids))
            for i in range(len(tids)):
                sims = E_cat @ E_cat[i]
                sims[i] = -np.inf
                top_k = np.argsort(sims)[-k:]
                for idx in top_k:
                    n_occ[idx] += 1
            
            # NaN, not 0.0, for a constant occurrence vector -- exactly as
            # `geometry_hubness` already does. S_Nk is the standardized third
            # moment, so a constant vector makes it 0/0 and `scipy.stats.skew`
            # returns NaN; 0.0 asserts "perfectly symmetric", which the data
            # does not support and which is indistinguishable from a genuine
            # symmetric distribution. Unreachable at the shipped taxonomy (the
            # smallest scored category has 10 tokens, and the branch needs
            # m in {5, 6}), but the two tasks must not disagree on a convention
            # (codex round 5 / independent Opus review, 2026-09-05).
            results[f"{cat_name}_hub_skew"] = float(
                float("nan") if n_occ.std() == 0 else skew(n_occ)
            )
            results[f"{cat_name}_hub_max"] = int(n_occ.max())
            
        return results

    def _compute_relation_consistency(self, E, relation_pairs, tokenizer):
        results = {}
        # Normalize E specifically for cosine sim
        E_norm = _safe_unit_rows(E)
        
        for cat_name, pairs in relation_pairs.items():
            if len(pairs) < 2: continue
            
            # 1. Cosine Consistency of Difference Vectors
            # DIFFERENCES IN DOUBLE. `v2 - v1` in the embedding's own dtype
            # OVERFLOWS when the two coordinates have opposite signs near the
            # top of the range: 2.4e38 - (-2.4e38) = 4.8e38 > float32's 3.4e38,
            # so the difference vector becomes inf and `_safe_unit_rows` then
            # returns NaN. On 4 exact constant-relation pairs at
            # 2.406159650500376e38 -- every input finite -- the task returned
            # consistency/top-1/top-5 = NaN/0.0/1.0 where the definition gives
            # 1.0/1.0/1.0, while all six lower rungs were exact (gpt-5.6-sol
            # round 9, 2026-09-07).
            #
            # The relation statistics are cosines of difference vectors, so
            # they are scale-free and this only ever removes error. Promoting a
            # float32/bf16 embedding to double is exact.
            _E64 = np.asarray(E, dtype=np.float64)
            diff_vectors = []
            for t1, t2 in pairs:
                diff_vectors.append(_E64[t2] - _E64[t1])
            
            diff_vectors = np.array(diff_vectors)
            # Normalize diff vectors to compute their cosine similarity
            diff_norm = _safe_unit_rows(diff_vectors)
            
            # Compute pairwise similarities
            sim_matrix = diff_norm @ diff_norm.T
            
            # Average off-diagonal elements
            n = len(pairs)
            if n > 1:
                sum_sim = np.sum(sim_matrix) - np.trace(sim_matrix)
                avg_sim = sum_sim / (n * (n - 1))
            else:
                avg_sim = 0.0
                
            results[f"{cat_name}_consistency"] = float(avg_sim)
            
            # 2. Analogy Accuracy (a:b :: c:?)
            # Target: b - a + c
            # We check if the nearest neighbor to (b - a + c) is d
            # We must exclude a, b, c from candidates
            
            # Optimization: limit validation set size to avoid O(N^2) if pairs are many
            # But here N=50 is small.
            
            correct_top1 = 0
            correct_top5 = 0
            total = 0
            
            # Create a localized vocab of just the relevant tokens to speed up? 
            # No, analogy needs full vocab search usually to be rigorous, 
            # but usually restricted to the candidate set or a subset. 
            # Let's simple search over the full vocab E_norm.
            
            # Limit number of analogy checks to avoid slow run if pairs are huge
            # e.g. check max 100 analogies
            
            pair_indices = list(range(n))
            np.random.seed(42)
            # Generate random quartets (pair i, pair j)
            # We treat pair i as (a, b) and pair j as (c, d)
            # We want to predict d from a, b, c
            
            # check all pairs if small, else sample
            check_indices = []
            if n < 20:
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            check_indices.append((i, j))
            else:
                 # Sample 200 random pairs
                 for _ in range(200):
                     i, j = np.random.choice(n, 2, replace=False)
                     check_indices.append((i, j))
            
            for i, j in check_indices:
                a_id, b_id = pairs[i]
                c_id, d_id = pairs[j]
                
                # Center calculation: d_pred = b - a + c
                # In normalized space: usually analogies are done on unnormalized vectors, 
                # then similarity is cosine.
                # Let's use E (unnormalized) for arithmetic, and E_norm for search.
                
                vec_a = E[a_id]
                vec_b = E[b_id]
                vec_c = E[c_id]
                
                target_vec = vec_b - vec_a + vec_c
                
                # Normalize target for cosine search
                target_norm = _safe_unit_rows(target_vec[None, :])[0]
                
                # Search in E_norm
                sims = E_norm @ target_norm
                
                # Exclude a, b, c from search (set to -inf)
                sims[a_id] = -np.inf
                sims[b_id] = -np.inf
                sims[c_id] = -np.inf
                
                # Retrieve top k
                top_k_ids = np.argsort(sims)[-5:] # Top 5
                top_1_id = top_k_ids[-1]
                
                if d_id == top_1_id:
                    correct_top1 += 1
                if d_id in top_k_ids:
                    correct_top5 += 1
                
                total += 1
            
            if total > 0:
                # `_meta_`: ANALOGY ACCURACY IS A CAPABILITY MEASURE -- "can
                # this model complete word analogies" -- so it is exactly what
                # the study is trying to PREDICT, not a structural predictor of
                # it. The analysis pipeline blocks performance-like columns by
                # regex, but the pattern requires `.acc` and these spell
                # `_acc`, so all four returned KEPT (rho -0.665 with
                # vocab_size; independent Opus review, 2026-09-05). The regex
                # is being widened too; a task declaring its own non-features
                # is the half that does not depend on the consumer.
                results[f"_meta_{cat_name}_analogy_acc_top1"] = float(correct_top1 / total)
                results[f"_meta_{cat_name}_analogy_acc_top5"] = float(correct_top5 / total)
            else:
                results[f"_meta_{cat_name}_analogy_acc_top1"] = 0.0
                results[f"_meta_{cat_name}_analogy_acc_top5"] = 0.0

        return results

    def _compute_projection(self, E, cat_tokens, method="umap"):
        """
        Project category tokens + random subset of other tokens to 2D.
        Returns coordinates.
        """
        results = {}
        target_tids = []
        labels = [] # category name or "other"
        
        # Collect target tokens
        for cat, tids in cat_tokens.items():
            target_tids.extend(tids)
            labels.extend([cat] * len(tids))
            
        # Add background tokens (random sample)
        target_set = set(target_tids)
        n_vocab = len(E)
        n_background = min(500, n_vocab - len(target_set))
        
        if n_background > 0:
            np.random.seed(42)
            candidates = [i for i in range(n_vocab) if i not in target_set]
            bg_tids = np.random.choice(candidates, n_background, replace=False)
            target_tids.extend(bg_tids)
            labels.extend(["other"] * len(bg_tids))
            
        target_tids = np.array(target_tids)
        X = E[target_tids]
        
        # Project
        coords = None
        try:
            if method.lower() == "umap":
                try:
                    import umap
                    reducer = umap.UMAP(n_components=2, random_state=42)
                    coords = reducer.fit_transform(X)
                except ImportError:
                    logger.info("  Warning: umap-learn not installed. Skipping UMAP.")
                    return {"projection_error": "umap-learn not installed"}
                    
            elif method.lower() == "tsne":
                try:
                    from sklearn.manifold import TSNE
                    # perplexity must be < n_samples
                    perp = min(30, len(X) - 1)
                    reducer = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
                    coords = reducer.fit_transform(X)
                except ImportError:
                    logger.info("  Warning: sklearn not installed. Skipping t-SNE.")
                    return {"projection_error": "scikit-learn not installed"}
                    
            elif method.lower() == "pca":
                try:
                    from sklearn.decomposition import PCA
                    reducer = PCA(n_components=2)
                    coords = reducer.fit_transform(X)
                except ImportError:
                    return {"projection_error": "scikit-learn not installed"}
                    
            if coords is not None:
                # Format results
                points = []
                for i in range(len(target_tids)):
                    points.append({
                        "tid": int(target_tids[i]),
                        "category": labels[i],
                        "x": float(coords[i, 0]),
                        "y": float(coords[i, 1])
                    })
                results["projection_points"] = points
                
        except Exception as e:
            logger.info(f"  Projection failed: {e}")
            results["projection_error"] = str(e)
            
        return results
