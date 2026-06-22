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
"""

import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np
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
    mean = vectors.mean(axis=0, keepdims=True)
    m_norm = np.linalg.norm(mean)
    if m_norm == 0:
        return float("nan")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms > 0, norms, 1.0)
    unit_mean = mean / m_norm
    return float((unit @ unit_mean.T).mean())


def _max_explainable_variance(vectors: np.ndarray) -> float:
    """Fraction of variance captured by the top PC of `vectors` (n, d)."""
    n = vectors.shape[0]
    if n < 2:
        return float("nan")
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    try:
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
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
    norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)
    unit = all_vectors / np.where(norms > 0, norms, 1.0)
    rng = np.random.default_rng(1)
    a = rng.integers(0, n, size=n_pairs)
    b = rng.integers(0, n, size=n_pairs)
    diff = a != b
    a, b = a[diff], b[diff]
    if len(a) == 0:
        return 0.0
    return float(np.einsum("ij,ij->i", unit[a], unit[b]).mean())


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

        num_samples = self.config.get("num_samples", 200)
        # Words to track must appear ≥ min_word_occurrences contexts to
        # be included in the self-similarity / MEV averages.
        min_word_occurrences = self.config.get("min_word_occurrences", 5)
        # Layers to report. None ⇒ ~5 evenly-spaced layers; "all" ⇒ all.
        layer_subset = self.config.get("layers", None)

        if dataset is None:
            dataset = [
                {"text": "The quick brown fox jumps over the lazy dog."}
                for _ in range(50)
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

        with torch.no_grad():
            for i, sample in enumerate(tqdm(dataset, desc="Contextualization")):
                if i >= num_samples:
                    break

                text = sample["text"] if isinstance(sample, dict) else str(sample)
                inputs = tokenizer(text, return_tensors="pt",
                                   truncation=True, max_length=128).to(device)
                if inputs["input_ids"].shape[1] < 4:
                    continue

                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states  # (L+1) tuple of (1, T, D)
                token_ids = inputs["input_ids"][0].cpu().tolist()
                # Skip the embedding layer (index 0); keep block outputs 1..L.
                per_layer = hidden[1:]
                if not n_layers_seen:
                    n_layers_seen = len(per_layer)

                for li, h in enumerate(per_layer):
                    h_np = h[0].float().cpu().numpy()  # (T, D)
                    intra_per_layer[li].append(h_np)
                    for tok_pos, tok_id in enumerate(token_ids):
                        token_states[li][tok_id].append(h_np[tok_pos])
                        all_states_per_layer[li].append(h_np[tok_pos])

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
            for tok_id, vecs in token_states[li].items():
                if len(vecs) < min_word_occurrences:
                    continue
                arr = np.asarray(vecs)
                self_sims.append(_cosine_pairwise_mean(arr))
                mevs.append(_max_explainable_variance(arr))
            self_sim_raw = float(np.nanmean(self_sims)) if self_sims else float("nan")
            self_sim_corr = (self_sim_raw - baseline) if not np.isnan(self_sim_raw) else float("nan")
            mev_mean = float(np.nanmean(mevs)) if mevs else float("nan")

            # 2. Intra-sentence similarity (Ethayarajh 2019 IntraSim):
            # mean cosine of each token to the sentence-mean vector, then
            # average across sentences. (Token-to-mean, NOT pairwise — the
            # pairwise form is SelfSim, used above.)
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
                "anisotropy_baseline": baseline,
                "n_words_tracked": len(self_sims),
            }

        # Compact aggregate summary across the reported layers.
        def _agg(key):
            xs = [v[key] for v in per_layer_metrics.values() if not np.isnan(v[key])]
            return float(np.mean(xs)) if xs else float("nan")

        return {
            "mean_self_similarity_corrected": _agg("self_similarity_corrected"),
            "mean_intra_sentence_similarity_corrected": _agg("intra_sentence_similarity_corrected"),
            "mean_mev": _agg("mev"),
            "mean_anisotropy_baseline": _agg("anisotropy_baseline"),
            "per_layer": per_layer_metrics,
            "layers_reported": layers_to_report,
        }
