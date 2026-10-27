"""
Contextualization metrics — Ethayarajh 2019 ("How Contextual are Contextualized
Word Representations?", arXiv:1909.00512).

Three complementary measurements per layer:

  1. **Self-similarity**: average cosine similarity of representations of the
     same word across different contexts. Lower = more contextualized.
     Anisotropy-corrected: subtract the random-pair baseline.

  2. **Intra-sentence similarity**: average cosine similarity of all token
     representations within the same sentence. High intra-sentence similarity
     means the model collapses sentence content to a single vector;
     low means each token retains distinct meaning. Anisotropy-corrected.

  3. **Maximum Explainable Variance (MEV)**: for each word that appears in
     multiple contexts, the fraction of variance captured by the top PC of
     its contextual representations. High MEV = the model encodes the word
     in a (mostly) static direction; low MEV = its meaning is highly
     context-dependent.

The bare versions (`*_raw`) are reported alongside the
anisotropy-corrected versions (`*_corrected`), which subtract the average
cosine similarity of random pairs of tokens at the same layer.

MEV is reported both raw (matching the official variance_explained_by_pc)
and anisotropy-adjusted. The paper's headline MEV figures are
anisotropy-adjusted (Sec. 3.4) and are NOT directly comparable.
"""

import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np
from .utils import safe_unit_rows
import torch
from tqdm import tqdm

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


def _cosine_pairwise_mean(vectors: np.ndarray, max_pairs: int = 5000) -> float:
    """Mean cosine similarity over distinct pairs from `vectors` (n, d)."""
    n = vectors.shape[0]
    if n < 2:
        return float("nan")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0, norms, 1.0)
    unit = vectors / safe_norms
    if n * (n - 1) // 2 <= max_pairs:
        sim = unit @ unit.T
        triu = sim[np.triu_indices(n, k=1)]
        return float(triu.mean())
    rng = np.random.default_rng(0)
    a = rng.integers(0, n, size=max_pairs)
    b = rng.integers(0, n, size=max_pairs)
    diff = a != b
    a, b = a[diff], b[diff]
    return float(np.einsum("ij,ij->i", unit[a], unit[b]).mean())


def _intra_sentence_mean_cosine(vectors: np.ndarray) -> float:
    """Ethayarajh 2019 IntraSim: mean cosine of each token vector to the
    SENTENCE-MEAN vector.

    IntraSim_l(s) = (1/n) Σ_i cos(f_l(s, i), s̄_l), where s̄_l is the
    average of the token vectors. This is distinct from the pairwise-mean
    cosine used for SelfSim — the reference (kawine/contextual analyze.py)
    averages token-to-sentence-mean similarities, not token-to-token pairs.
    """
    n = vectors.shape[0]
    if n < 2:
        return float("nan")
    # Scale-exact unit rows and a float64 mean: Definitions 1-2 are invariant
    # to a common rescale, which a raw norm (underflow) or a float32 sum
    # (overflow) breaks (tests/tasks/test_reference_parity_formulas.py).
    mean = np.asarray(vectors, dtype=np.float64).mean(axis=0, keepdims=True)
    if not np.any(mean):
        return float("nan")
    unit = safe_unit_rows(vectors)
    unit_mean = safe_unit_rows(mean)
    return float((unit @ unit_mean.T).mean())


def _max_explainable_variance(vectors: np.ndarray) -> float:
    """Ethayarajh 2019 MEV: variance explained by the first principal
    component of a word's occurrence matrix.

    CENTERED, matching the official implementation
    (kawine/contextual `analyze.py::variance_explained_by_pc`, sklearn
    ``PCA(n_components=1).explained_variance_ratio_[0]``). The released
    gpt2/variance_explained.csv (raw MEV ~0.29-0.35 over layers 2-11) settles
    the convention; the uncentred form reads ~0.99 there.
    """
    n = vectors.shape[0]
    if n < 2:
        return float("nan")
    centered = np.asarray(vectors, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    try:
        # `safe_svdvals` factors out the max, so the cast back cannot overflow.
        from .utils import safe_svdvals
        S = safe_svdvals(centered)
    except (np.linalg.LinAlgError, torch._C._LinAlgError):
        # Both types: the torch error and numpy's share no ancestor.
        return float("nan")
    eig = S ** 2
    total = eig.sum()
    if total <= 0:
        return float("nan")
    return float(eig[0] / total)


def _baseline_anisotropy(all_vectors: np.ndarray, n_pairs: int = 2000) -> float:
    """Random-pair cosine similarity baseline used to anisotropy-correct."""
    n = all_vectors.shape[0]
    if n < 2:
        return 0.0
    # Scale-exact unit rows, as in `_intra_sentence_mean_cosine`.
    unit = safe_unit_rows(all_vectors)
    rng = np.random.default_rng(1)
    a = rng.integers(0, n, size=n_pairs)
    b = rng.integers(0, n, size=n_pairs)
    diff = a != b
    a, b = a[diff], b[diff]
    if len(a) == 0:
        return 0.0
    return float(np.einsum("ij,ij->i", unit[a], unit[b]).mean())


def _max_explainable_variance_truncsvd(vectors) -> float:
    """MEV under the OFFICIAL baseline: sklearn TruncatedSVD's
    ``explained_variance_ratio_[0]``.

    `kawine/contextual` analyze.py::explore_embedding_space reports
    ``TruncatedSVD(n_components=100).explained_variance_ratio_[0]``: the
    direction v0 from an UNcentred SVD, but both variances centred,
    Var(X @ v0) / sum_j Var(X_j) -- neither the uncentred energy ratio nor PCA.
    """
    X = np.asarray(vectors, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return float("nan")
    # Direction from the UNCENTERED SVD, as TruncatedSVD does.
    try:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan")
    if Vt.shape[0] < 1:
        return float("nan")
    v0 = Vt[0]
    # Both variances CENTERED, as sklearn does (ddof=1, matching np.var of the
    # transformed data in sklearn's implementation).
    proj = X @ v0
    num = float(np.var(proj, ddof=1))
    den = float(np.var(X, axis=0, ddof=1).sum())
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return float("nan")
    return num / den


def _baseline_mev(all_vectors: np.ndarray, occ_counts=None, n_draws: int = 200,
                  seed: int = 2, size_matched: bool = True) -> float:
    """Anisotropy baseline for MEV (Ethayarajh 2019, Sec. 3.4 / 4.3).

    Raw MEV is not comparable across models: in a highly anisotropic space ANY
    set of vectors -- including vectors of unrelated words -- has a large share
    of its variance on one principal component. The correction subtracts the MEV
    of randomly sampled words.

    ``size_matched=True`` (the default) draws random groups whose sizes match
    the real per-word occurrence counts: MEV depends strongly on n, and a
    tracked word has a median of ~9 occurrences against the pool's 1000.
    ``size_matched=False`` is the literal Sec. 3.4 reading (1K uniformly
    sampled words, uncentred TruncatedSVD), reported as ``mev_baseline_pool1k``.
    """
    all_vectors = np.asarray(all_vectors)
    n = all_vectors.shape[0]
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    if not size_matched:
        # The literal reading uses the paper's algorithm: uncentred TruncatedSVD.
        return _max_explainable_variance_truncsvd(all_vectors[
            rng.choice(n, size=int(min(1000, n)), replace=False)])
    counts = [int(c) for c in (occ_counts or []) if c >= 2]
    if not counts:
        return float("nan")
    vals = []
    for _ in range(min(n_draws, len(counts) * 20)):
        k = min(counts[rng.integers(0, len(counts))], n)
        if k < 2:
            continue
        v = _max_explainable_variance(all_vectors[rng.choice(n, size=k, replace=False)])
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


@register_task("geometry_contextualization")
class ContextualizationTask(DiagnosticTask):
    """
    Ethayarajh 2019 contextualization profile (self-similarity,
    intra-sentence similarity, MEV) per layer, with anisotropy correction.

    Designed to be cheap: requires only `output_hidden_states=True` and a
    handful of frequent target words that appear in multiple contexts.
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Contextualization Analysis (Ethayarajh 2019)...")

        # Ethayarajh 2019 uses ~1000 sentences (Sec. 3.4).
        num_samples = self.config.get("num_samples", 1000)
        # Words to track must appear ≥ min_word_occurrences contexts to
        # be included in the self-similarity / MEV averages.
        min_word_occurrences = self.config.get("min_word_occurrences", 5)
        # Layers to report. None ⇒ ~5 evenly-spaced layers; "all" ⇒ all.
        layer_subset = self.config.get("layers", None)

        if dataset is None:
            # Distinct sentences reusing common words in different contexts
            # (identical sentences would make SelfSim and MEV exactly 1.0).
            subjects = ["fox", "dog", "bank", "river", "market", "story",
                        "model", "system", "garden", "player"]
            frames = [
                "The {w} was near the old {v} before dawn.",
                "Every {w} in the {v} looked different that day.",
                "She described the {w} while walking past the {v}.",
                "A {w} can change how the {v} is understood.",
                "Nobody expected the {w} to matter for the {v}.",
            ]
            dataset = [
                {"text": f.format(w=subjects[i % len(subjects)],
                                  v=subjects[(i + 3) % len(subjects)])}
                for i, f in enumerate(
                    [fr for fr in frames for _ in range(10)])
            ]

        device = next(model.parameters()).device

        # Step 1: collect per-token hidden states grouped by token id, per layer.
        # We also keep all-token vectors per layer to compute the random-pair
        # baseline.
        token_states: Dict[int, Dict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
        # token_states[layer_idx][token_id] = list of vectors

        all_states_per_layer: Dict[int, List[np.ndarray]] = defaultdict(list)
        # For intra-sentence similarity: list of (layer -> per-sample vectors).
        intra_per_layer: Dict[int, List[np.ndarray]] = defaultdict(list)

        n_layers_seen = 0
        # Seeded ONCE per evaluate: picks which token of each sentence joins
        # the anisotropy baseline pool. Seeded for reproducibility, drawn
        # per sentence so the pool is not positionally biased.
        rng_pool = np.random.default_rng(20260902)

        # post-final-norm=False, DELIBERATELY: kawine/contextual's vendored GPT-2
        # pops the pre-norm last layer and appends ln_f(...)
        # (modeling_gpt2.py:562-565), so the deepest layer is post-norm.
        with torch.no_grad():
            for i, sample in enumerate(tqdm(dataset, desc="Contextualization")):
                if i >= num_samples:
                    break

                text = sample["text"] if isinstance(sample, dict) else str(sample)
                inputs = tokenizer(text, return_tensors="pt",
                                   truncation=True, max_length=128).to(device)
                if inputs["input_ids"].shape[1] < 4:
                    continue

                # post-final-norm=False, DELIBERATELY (see above).
                outputs = model(**inputs, output_hidden_states=True)
                hidden = list(outputs.hidden_states)  # (L+1) of (1, T, D)
                token_ids = inputs["input_ids"][0].cpu().tolist()
                # Special tokens are excluded (the reference strips them; a
                # context-free BOS would read SelfSim/MEV = 1.0).
                special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
                # Skip the embedding layer (index 0); keep block outputs 1..L.
                per_layer = hidden[1:]
                if not n_layers_seen:
                    n_layers_seen = len(per_layer)

                # Baseline pool: one token per sentence, chosen uniformly over
                # the kept positions and reused for every layer (random pairs
                # must not share a context).
                non_special = [p_ for p_, t_ in enumerate(token_ids)
                               if t_ not in special_ids]
                non_special_all = non_special
                pool_pos = (int(rng_pool.choice(non_special))
                            if non_special else None)

                # IntraSim sees the same positions as SelfSim and MEV.
                keep_pos = np.asarray(non_special_all, dtype=np.int64)

                for li, h in enumerate(per_layer):
                    h_np = h[0].float().cpu().numpy()  # (T, D)
                    if keep_pos.size >= 2:
                        intra_per_layer[li].append(h_np[keep_pos])
                    if pool_pos is not None:
                        all_states_per_layer[li].append(h_np[pool_pos])
                    for tok_pos, tok_id in enumerate(token_ids):
                        if tok_id in special_ids:
                            continue
                        token_states[li][tok_id].append(h_np[tok_pos])

        if n_layers_seen == 0:
            return {"error": "No samples processed"}

        # Pick which layers to report on.
        if layer_subset == "all":
            layers_to_report = list(range(n_layers_seen))
        elif isinstance(layer_subset, list):
            layers_to_report = [i for i in layer_subset if 0 <= i < n_layers_seen]
        else:
            # Default: ~5 evenly spaced layers including first and last.
            n_pick = min(5, n_layers_seen)
            layers_to_report = list(np.linspace(0, n_layers_seen - 1, n_pick).astype(int))

        # Step 2: per-layer metrics.
        per_layer_metrics: Dict[str, Dict[str, float]] = {}

        for li in layers_to_report:
            # Random-pair baseline (anisotropy)
            all_vecs = np.asarray(all_states_per_layer[li])
            baseline = _baseline_anisotropy(all_vecs)

            # 1. Self-similarity per word, then average across words.
            self_sims = []
            mevs = []
            occ_counts = []
            for tok_id, vecs in token_states[li].items():
                if len(vecs) < min_word_occurrences:
                    continue
                arr = np.asarray(vecs)
                self_sims.append(_cosine_pairwise_mean(arr))
                mevs.append(_max_explainable_variance(arr))
                occ_counts.append(len(vecs))
            self_sim_raw = float(np.nanmean(self_sims)) if self_sims else float("nan")
            self_sim_corr = (self_sim_raw - baseline) if not np.isnan(self_sim_raw) else float("nan")
            mev_mean = float(np.nanmean(mevs)) if mevs else float("nan")
            mev_baseline = _baseline_mev(all_vecs, occ_counts, size_matched=True)
            mev_baseline_1k = _baseline_mev(all_vecs, occ_counts, size_matched=False)
            # Both MEV corrections are emitted and neither is privileged: they
            # disagree in sign on this corpus, so there is no unqualified
            # `mev_corrected` key.
            mev_corr_sm = (mev_mean - mev_baseline
                           if not (np.isnan(mev_mean) or np.isnan(mev_baseline))
                           else float("nan"))
            mev_corr_1k = (mev_mean - mev_baseline_1k
                           if not (np.isnan(mev_mean) or np.isnan(mev_baseline_1k))
                           else float("nan"))

            # 2. IntraSim: mean cosine of each token to its sentence mean.
            intra_sims = []
            for h_np in intra_per_layer[li]:
                if h_np.shape[0] < 2:
                    continue
                intra_sims.append(_intra_sentence_mean_cosine(h_np))
            intra_raw = float(np.nanmean(intra_sims)) if intra_sims else float("nan")
            intra_corr = (intra_raw - baseline) if not np.isnan(intra_raw) else float("nan")

            per_layer_metrics[f"layer{li}"] = {
                "self_similarity_raw": self_sim_raw,
                "self_similarity_corrected": self_sim_corr,
                "intra_sentence_similarity_raw": intra_raw,
                "intra_sentence_similarity_corrected": intra_corr,
                "mev": mev_mean,
                "mev_corrected_sizematched": mev_corr_sm,
                "mev_corrected_pool1k": mev_corr_1k,
                "mev_baseline": mev_baseline,
                "mev_baseline_pool1k": mev_baseline_1k,
                "anisotropy_baseline": baseline,
                # A tokenizer statistic (r = -0.905 with log vocab), not geometry.
                "_meta_n_words_tracked": len(self_sims),
            }

        # Compact aggregate summary across the reported layers.
        def _agg(key):
            xs = [v[key] for v in per_layer_metrics.values() if not np.isnan(v[key])]
            return float(np.mean(xs)) if xs else float("nan")

        return {
            # False: the deepest layer is post-ln_f, as the reference defines it.
            "_meta_last_entry_is_block_output": False,
            "mean_self_similarity_corrected": _agg("self_similarity_corrected"),
            "mean_intra_sentence_similarity_corrected": _agg("intra_sentence_similarity_corrected"),
            "mean_mev": _agg("mev"),
            "mean_mev_corrected_sizematched": _agg("mev_corrected_sizematched"),
            "mean_mev_corrected_pool1k": _agg("mev_corrected_pool1k"),
            "mean_mev_baseline": _agg("mev_baseline"),
            "mean_mev_baseline_pool1k": _agg("mev_baseline_pool1k"),
            "mean_anisotropy_baseline": _agg("anisotropy_baseline"),
            "per_layer": per_layer_metrics,
            # Python ints: numpy scalars break strict JSON encoders.
            "layers_reported": [int(i) for i in layers_to_report],
        }
