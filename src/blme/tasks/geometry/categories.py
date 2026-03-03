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
logger = logging.getLogger("blme")

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
        
        for cat_name, tids in cat_tokens.items():
            if len(tids) < 3: continue
            
            # Intra
            intra_dists = []
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    d = cosine_dist(E[tids[i]], E[tids[j]])
                    intra_dists.append(d)
            intra = np.mean(intra_dists) if intra_dists else 0
            
            # Inter
            inter_dists = []
            for tid in tids:
                for rtid in random_tids[:10]:
                    d = cosine_dist(E[tid], E[rtid])
                    inter_dists.append(d)
            inter = np.mean(inter_dists) if inter_dists else 0
            
            results[f"{cat_name}_intra"] = float(intra)
            results[f"{cat_name}_inter"] = float(inter)
            results[f"{cat_name}_separation"] = float(inter - intra)
            
        return results

    def _compute_purity(self, E, cat_tokens, cat_labels, k=20):
        # Normalize E for fast cosine sim
        E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)
        results = {}
        
        for cat_name, tids in cat_tokens.items():
            if len(tids) < 2: continue
            
            purities = []
            for tid in tids:
                sims = E_norm @ E_norm[tid]
                sims[tid] = -np.inf
                top_k = np.argsort(sims)[-k:]
                
                same_cat = sum(1 for t in top_k if cat_labels.get(t) == cat_name)
                purities.append(same_cat / k)
                
            results[f"{cat_name}_purity"] = float(np.mean(purities))
            
        return results
        
    def _compute_hubness(self, E, tokenizer, cat_tokens, k_cat=5):
        E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)
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
            
            results[f"{cat_name}_hub_skew"] = float(skew(n_occ))
            results[f"{cat_name}_hub_max"] = int(n_occ.max())
            
        return results

    def _compute_relation_consistency(self, E, relation_pairs, tokenizer):
        results = {}
        # Normalize E specifically for cosine sim
        E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)
        
        for cat_name, pairs in relation_pairs.items():
            if len(pairs) < 2: continue
            
            # 1. Cosine Consistency of Difference Vectors
            diff_vectors = []
            for t1, t2 in pairs:
                v1 = E[t1]
                v2 = E[t2]
                diff = v2 - v1
                diff_vectors.append(diff)
            
            diff_vectors = np.array(diff_vectors)
            # Normalize diff vectors to compute their cosine similarity
            diff_norm = diff_vectors / (np.linalg.norm(diff_vectors, axis=1, keepdims=True) + 1e-10)
            
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
                target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-10)
                
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
                results[f"{cat_name}_analogy_acc_top1"] = float(correct_top1 / total)
                results[f"{cat_name}_analogy_acc_top5"] = float(correct_top5 / total)
            else:
                results[f"{cat_name}_analogy_acc_top1"] = 0.0
                results[f"{cat_name}_analogy_acc_top5"] = 0.0

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
