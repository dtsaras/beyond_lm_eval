"""
Neural Collapse — Papyan, Han & Donoho 2020 (arXiv:2008.03465).

Measures the geometric properties of last-layer representations grouped by
class label. Three metrics from the Neural Collapse phenomenon:

  - **NC1 — within-class variability collapse**: tr(Σ_W Σ_B^†) / K, where
    Σ_W is the within-class covariance, Σ_B is the between-class
    covariance, K is the number of classes, and (·)^† is the pseudo-inverse.
    Lower = more collapsed (each class has converged to a point).

  - **NC2-equinorm**: coefficient of variation of class-mean norms
    ||μ_k - μ_global||. Lower = more equal class-mean norms.

  - **NC2-equiangularity**: standard deviation of cosine similarities
    between every pair of (μ_k - μ_global) vectors, vs. the ETF target of
    -1/(K-1). Lower = closer to a simplex equiangular tight frame.

Because BLME is unsupervised, we use a small bundled topic-classification
dataset. The user can override `dataset` with their own
`{text, label}` items if they want to compute NC on a different task.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_final_norm

logger = logging.getLogger("blme")


_TOPIC_BUNDLE: List[Tuple[str, str]] = [
    # weather
    ("Heavy rain is expected throughout the weekend.", "weather"),
    ("The forecast calls for sunshine and warm temperatures.", "weather"),
    ("A blizzard hit the northern states overnight.", "weather"),
    ("Hurricane winds knocked down trees across the coast.", "weather"),
    ("Today will be cloudy with a chance of thunderstorms.", "weather"),
    ("The drought has lasted for three months now.", "weather"),
    ("Humidity is unusually high for this time of year.", "weather"),
    ("A cold front is moving through the central plains.", "weather"),
    # sports
    ("The team won the championship in overtime.", "sports"),
    ("She scored the winning goal with seconds to spare.", "sports"),
    ("The marathon was held under perfect running conditions.", "sports"),
    ("The pitcher threw a perfect game last night.", "sports"),
    ("Their star player suffered a season-ending injury.", "sports"),
    ("The Olympic athletes trained for years to qualify.", "sports"),
    ("The coach was fired after a losing streak.", "sports"),
    ("Tennis fans cheered as the underdog won the match.", "sports"),
    # music
    ("The symphony performed Beethoven's ninth last night.", "music"),
    ("Her new album debuted at number one on the charts.", "music"),
    ("The jazz quartet played a sold-out show downtown.", "music"),
    ("He plays guitar in a heavy metal band.", "music"),
    ("The opera singer hit a high C without effort.", "music"),
    ("The recording studio booked sessions for next month.", "music"),
    ("Classical music has a calming effect on listeners.", "music"),
    ("Their world tour will visit forty cities.", "music"),
    # food
    ("The restaurant serves the best pasta in town.", "food"),
    ("She baked a chocolate cake for her birthday.", "food"),
    ("Fresh vegetables are available at the farmers market.", "food"),
    ("The chef recommends the seafood special tonight.", "food"),
    ("He grilled steaks and corn for the barbecue.", "food"),
    ("The bakery sells fresh bread every morning.", "food"),
    ("Ice cream is the perfect summer dessert.", "food"),
    ("The soup needs more salt and pepper.", "food"),
    # technology
    ("The new smartphone has an improved camera.", "technology"),
    ("Artificial intelligence is transforming many industries.", "technology"),
    ("She wrote the software using Python and Rust.", "technology"),
    ("The data center went offline for two hours.", "technology"),
    ("Cloud computing has changed how businesses operate.", "technology"),
    ("The startup launched their product on the app store.", "technology"),
    ("Quantum computers may eventually break current encryption.", "technology"),
    ("He fixed the bug in the production database.", "technology"),
]


def _neural_collapse_metrics(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute NC1 + NC2 metrics from a feature matrix and class labels."""
    classes = np.unique(labels)
    K = len(classes)
    if K < 2:
        return {"error": "Need at least 2 classes for Neural Collapse"}
    n, d = features.shape
    if n < K + 1:
        return {"error": "Need at least K+1 samples"}

    global_mean = features.mean(axis=0)
    class_means = np.zeros((K, d), dtype=np.float64)
    for ki, c in enumerate(classes):
        class_means[ki] = features[labels == c].mean(axis=0)

    # Sigma_W: within-class scatter (covariance)
    sigma_w = np.zeros((d, d), dtype=np.float64)
    for ki, c in enumerate(classes):
        diffs = features[labels == c] - class_means[ki]
        sigma_w += diffs.T @ diffs
    sigma_w /= n

    # Sigma_B: between-class scatter
    centered = class_means - global_mean
    sigma_b = (centered.T @ centered) / K

    # NC1: tr(Sigma_W @ pinv(Sigma_B)) / K
    try:
        sigma_b_pinv = np.linalg.pinv(sigma_b, rcond=1e-10)
        nc1 = float(np.trace(sigma_w @ sigma_b_pinv) / K)
    except np.linalg.LinAlgError:
        nc1 = float("nan")

    # NC2 — equinorm
    M = class_means - global_mean
    norms = np.linalg.norm(M, axis=1)
    if norms.mean() > 0:
        nc2_equinorm_cv = float(np.std(norms) / norms.mean())
    else:
        nc2_equinorm_cv = float("nan")

    # NC2 — equiangularity
    # Pairwise cosine similarities between centered class means.
    M_unit = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    cos = M_unit @ M_unit.T
    iu = np.triu_indices(K, k=1)
    pair_cos = cos[iu]
    if len(pair_cos) > 0:
        target = -1.0 / (K - 1)  # ETF target
        nc2_equiangularity_dev = float(np.mean(np.abs(pair_cos - target)))
    else:
        nc2_equiangularity_dev = float("nan")

    return {
        "nc1_within_class_collapse": nc1,
        "nc2_equinorm_cv": nc2_equinorm_cv,
        "nc2_equiangularity_dev": nc2_equiangularity_dev,
        "n_classes": int(K),
        "n_samples": int(n),
    }


@register_task("geometry_neural_collapse")
class NeuralCollapseTask(DiagnosticTask):
    """Neural Collapse NC1 + NC2 on a small bundled topic dataset."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Neural Collapse Analysis...")

        # Allow override; otherwise use bundled topic dataset.
        if dataset is not None and isinstance(dataset, list) and dataset and (
            isinstance(dataset[0], dict) and {"text", "label"} <= set(dataset[0])
        ):
            samples = [(d["text"], d["label"]) for d in dataset]
        else:
            samples = list(_TOPIC_BUNDLE)

        device = next(model.parameters()).device
        final_norm = get_final_norm(model)

        # Mean-pool the final-layer hidden states for each sample. We
        # apply final_norm if present so the features live in the same
        # space as the LM-head input — this is the natural "last-layer
        # representation" for Neural Collapse.
        feats = []
        labs = []
        with torch.no_grad():
            for text, label in tqdm(samples, desc="Neural collapse"):
                inputs = tokenizer(text, return_tensors="pt",
                                   truncation=True, max_length=128).to(device)
                if inputs["input_ids"].shape[1] < 1:
                    continue
                out = model(**inputs, output_hidden_states=True)
                h = out.hidden_states[-1][0]  # (T, D)
                if final_norm is not None:
                    try:
                        norm_dtype = next(final_norm.parameters()).dtype
                    except StopIteration:
                        norm_dtype = h.dtype
                    h = final_norm(h.to(norm_dtype))
                # Mean-pool, ignoring padding (causal LMs typically don't
                # have padding here since we tokenize one text at a time).
                pooled = h.float().mean(dim=0).cpu().numpy()
                feats.append(pooled)
                labs.append(label)

        if len(feats) < 4:
            return {"error": "Too few samples for Neural Collapse"}

        feats_arr = np.stack(feats, axis=0)
        # Map string labels to integer ids
        label_to_id: Dict[str, int] = {}
        labs_int = []
        for lb in labs:
            if lb not in label_to_id:
                label_to_id[lb] = len(label_to_id)
            labs_int.append(label_to_id[lb])
        labs_arr = np.array(labs_int, dtype=np.int64)

        return _neural_collapse_metrics(feats_arr, labs_arr)
