"""
Neural Collapse — Papyan, Han & Donoho 2020 (arXiv:2008.08186).

Measures the geometric properties of last-layer representations grouped by
class label. Three metrics from the Neural Collapse phenomenon:

  - **NC1 — within-class variability collapse**: tr(Σ_W Σ_B^†) / K, where
    Σ_W is the within-class covariance, Σ_B is the between-class
    covariance, K is the number of classes, and (·)^† is the pseudo-inverse.
    Lower = more collapsed (each class has converged to a point).

  - **NC2-equinorm**: coefficient of variation of class-mean norms
    ||μ_k - μ_global||. Lower = more equal class-mean norms.

  - **NC2-ETF cosine deviation proxy**: MEAN ABSOLUTE deviation of
    pairwise cosine similarities between centred class-mean vectors from
    the ETF target ``-1/(K-1)`` (the Papyan et al. Fig. 4 maximal-angle
    equiangularity measure; their Fig. 3 std-of-cosines is a different
    statistic). This is a lightweight ETF-alignment proxy,
    not the full normalized Gram/Frobenius NC2 measure from Papyan et al.

The task follows the LLM adaptation of Wu & Papyan 2024 (arXiv:2405.17767):
next-token classes over a corpus, token-level post-norm features, CDNV. The
Papyan statistics above are computed on the old bundled topic proxy and on a
user-supplied ``{text, label}`` dataset.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ...registry import register_task
from ...tasks.base import DiagnosticTask

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
    # n == K is defined (Sigma_W = 0); only n < K is an error.
    if n < K:
        return {"error": "Need at least K samples"}

    # Promote before the means, so a common offset cancels exactly.
    features = np.asarray(features, dtype=np.float64)
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

    # Sigma_B: between-class scatter, rank <= K-1.
    centered = class_means - global_mean
    sigma_b = (centered.T @ centered) / K

    # NC1: tr(Sigma_W Sigma_B^+) / K in the top-(K-1) Sigma_B eigenspace
    # (Papyan et al. 2020); a naive pinv inverts noise eigenvalues.
    try:
        eigvals, eigvecs = np.linalg.eigh(sigma_b)
        # eigh returns ascending — take the top K-1.
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        k_eff = min(K - 1, d)
        lam = eigvals[:k_eff]
        V = eigvecs[:, :k_eff]

        # Drop numerical-zero eigenvalues that leak through eigh.
        top = lam.max() if lam.size else 0.0
        if top <= 0:
            # Sigma_B == 0: its pseudo-inverse is 0, so NC1 = 0 (as numpy pinv).
            nc1 = 0.0
            subspace_rank = 0
        else:
            # numpy's pinv rank tolerance (max(shape) * eps * top), the
            # reference convention (tests/tasks/parity/test_neural_collapse_parity.py).
            _rcond = max(sigma_b.shape) * np.finfo(lam.dtype).eps
            keep = lam > top * _rcond
            lam = lam[keep]
            V = V[:, keep]
            subspace_rank = int(keep.sum())
            # Project Σ_W into this subspace and divide by the
            # diagonal of Σ_B restricted to the same directions.
            sigma_w_proj = V.T @ sigma_w @ V          # (k_eff, k_eff)
            nc1 = float(np.trace(sigma_w_proj / lam[:, None]) / K)
    except np.linalg.LinAlgError:
        nc1 = float("nan")
        subspace_rank = 0

    # NC2 — equinorm
    M = class_means - global_mean
    norms = np.linalg.norm(M, axis=1)
    if norms.mean() > 0:
        # ddof=1, as the released notebook's `torch.std`.
        nc2_equinorm_cv = float(np.std(norms, ddof=1) / norms.mean())
    else:
        nc2_equinorm_cv = float("nan")

    # NC2 — ETF cosine deviation proxy (not full normalized Gram NC2).
    # Pairwise cosine similarities between centered class means.
    # Scale-exact cosine; an exactly zero centred class mean stays undefined.
    from .utils import safe_unit_rows as _safe_unit_rows
    M_unit = _safe_unit_rows(M.astype(np.float64), eps_zero=False)
    _degenerate_mean = np.ptp(np.abs(M), axis=1) == 0
    _degenerate_mean &= (np.abs(M).max(axis=1) == 0)
    if np.any(_degenerate_mean):
        M_unit = M_unit.copy()
        M_unit[_degenerate_mean] = np.nan
    cos = M_unit @ M_unit.T
    iu = np.triu_indices(K, k=1)
    pair_cos = cos[iu]
    if len(pair_cos) > 0:
        target = -1.0 / (K - 1)  # ETF target
        nc2_etf_cosine_deviation_proxy = float(np.mean(np.abs(pair_cos - target)))
        # Std of the pairwise cosines (Papyan Fig. 3): equiangularity spread.
        nc2_cosine_std = float(np.std(pair_cos))
    else:
        nc2_etf_cosine_deviation_proxy = float("nan")
        nc2_cosine_std = float("nan")

    return {
        "nc1_within_class_collapse": nc1,
        "nc1_subspace_rank": int(subspace_rank),
        # NC2 statistics are _meta_: all three sit inside their own 200-draw
        # label-permutation null at 8 sentences per class (underpowered).
        "_meta_nc2_equinorm_cv": nc2_equinorm_cv,
        "_meta_nc2_etf_cosine_deviation_proxy": nc2_etf_cosine_deviation_proxy,
        "_meta_nc2_cosine_std": nc2_cosine_std,
        "_meta_n_per_topic": int(round(n / K)) if K else 0,
        "_meta_n_classes": int(K),
        "_meta_n_samples": int(n),
    }


def _cdnv(means: np.ndarray, variances: np.ndarray, dist_exp: float,
          tile: int = 1024) -> float:
    """Mean over class pairs c < c' of ((s_c + s_c') / 2) / (||mu_c - mu_c'||^2)^dist_exp.

    ``neural_collapse.measure.variability_cdnv`` with ``kernels.
    class_dist_norm_vars``: dist_exp = 1 is Galanti et al.'s CDNV; Wu & Papyan
    (2024, Eq. 4) divide by one more power of the squared distance, which
    linguistic-collapse/analyze.py calls with dist_exp = 2. Distances come
    from the float64 Gram, tiled by rows; a pair with zero distance is
    skipped (the reference would divide by zero).
    """
    M = np.asarray(means, dtype=np.float64)
    v = np.asarray(variances, dtype=np.float64)
    C = M.shape[0]
    if C < 2:
        return float("nan")
    sq = np.einsum("ij,ij->i", M, M)
    total, count = 0.0, 0
    for a in range(0, C, tile):
        b = min(C, a + tile)
        d2 = sq[a:b, None] + sq[None, :] - 2.0 * (M[a:b] @ M.T)
        # Gram form, clamped at 0 against cancellation; upper triangle only.
        rows = np.arange(a, b)
        upper = np.arange(C)[None, :] > rows[:, None]
        d2 = np.where(upper, np.maximum(d2, 0.0), 0.0)
        num = 0.5 * (v[a:b, None] + v[None, :])
        ok = upper & (d2 > 0)
        total += float(np.sum(num[ok] / d2[ok] ** dist_exp))
        count += int(ok.sum())
    return total / count if count else float("nan")


def _next_token_class_stats(model, tokenizer, texts, min_per_class: int,
                            max_length: int = 512):
    """Per next-token class: count, feature mean and mean squared deviation.

    Wu & Papyan (linguistic-collapse lib/collection.py:103-112): features are
    the post-final-norm hidden states at every position t < T-1 (position 0
    included), labelled by the token at t+1. Classes with fewer than
    ``min_per_class`` samples are dropped before the forward pass (a token
    count needs only the tokenizer), so only kept classes are accumulated.
    """
    enc = []
    counts: Dict[int, int] = defaultdict(int)
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", truncation=True,
                        max_length=max_length)
        if ids["input_ids"].shape[1] < 2:
            continue
        enc.append(ids)
        for y in ids["input_ids"][0, 1:].tolist():
            counts[int(y)] += 1
    keep = sorted(c for c, n in counts.items() if n >= int(min_per_class))
    index = {c: i for i, c in enumerate(keep)}
    device = next(model.parameters()).device
    sums = sumsq = n = None
    n_tokens = 0
    with torch.no_grad():
        for ids in tqdm(enc, desc="Neural collapse (next-token)"):
            ids = ids.to(device)
            out = model(**ids, output_hidden_states=True)
            # post-final-norm: DELIBERATELY post-norm -- the classifier input
            # (the reference replaces lm_head by Identity, model.py:261-268).
            h = out.hidden_states[-1][0, :-1].detach().to("cpu", torch.float64).numpy()
            y = ids["input_ids"][0, 1:].tolist()
            n_tokens += len(y)
            sel = [(r, index[c]) for r, c in enumerate(y) if c in index]
            if not sel:
                continue
            if sums is None:
                sums = np.zeros((len(keep), h.shape[1]), dtype=np.float64)
                sumsq = np.zeros(len(keep), dtype=np.float64)
                n = np.zeros(len(keep), dtype=np.int64)
            rows = np.fromiter((r for r, _ in sel), dtype=np.int64)
            cls = np.fromiter((c for _, c in sel), dtype=np.int64)
            np.add.at(sums, cls, h[rows])
            np.add.at(sumsq, cls, np.einsum("ij,ij->i", h[rows], h[rows]))
            np.add.at(n, cls, 1)
    if sums is None:
        return None
    means = sums / n[:, None]
    # Mean squared deviation (the reference's VarNormAccumulator, 1/n_c).
    variances = np.maximum(sumsq / n - np.einsum("ij,ij->i", means, means), 0.0)
    return {"means": means, "variances": variances, "counts": n,
            "n_tokens": n_tokens, "n_docs": len(enc), "n_classes_seen": len(counts)}


def _topic_proxy(model, tokenizer, samples):
    """The previous BLME construct: NC1 of mean-pooled sentences by topic."""
    device = next(model.parameters()).device
    feats, labs = [], []
    with torch.no_grad():
        for text, label in tqdm(samples, desc="Neural collapse (topic proxy)"):
            inputs = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=128).to(device)
            if inputs["input_ids"].shape[1] < 1:
                continue
            out = model(**inputs, output_hidden_states=True)
            # post-final-norm: DELIBERATELY post-norm -- neural collapse
            # concerns the features the LM head reads.
            h = out.hidden_states[-1][0]  # (T, D) — already normed
            # Mean in float64, kept in float64 (no narrowing cast).
            feats.append(h.double().mean(dim=0).cpu().numpy())
            labs.append(label)
    if len(feats) < 4:
        return {"error": "Too few samples for Neural Collapse"}
    label_to_id: Dict[str, int] = {}
    labs_int = []
    for lb in labs:
        if lb not in label_to_id:
            label_to_id[lb] = len(label_to_id)
        labs_int.append(label_to_id[lb])
    return _neural_collapse_metrics(np.stack(feats, axis=0),
                                    np.array(labs_int, dtype=np.int64))


@register_task("geometry_neural_collapse")
class NeuralCollapseTask(DiagnosticTask):
    """Neural collapse of an LM's classifier features (Wu & Papyan 2024).

    Classes are next tokens: every position t < T-1 of every corpus text is a
    sample, its feature the post-final-norm hidden state and its label the
    token at t+1 (linguistic-collapse lib/collection.py). NC1 is reported as
    Wu & Papyan's CDNV (Eq. 4: Galanti's CDNV with one more power of the
    squared class-mean distance; analyze.py dist_exp=2) and, beside it, the
    scale-free Galanti CDNV. The previous topic-label construct (5 bundled
    topics x 8 sentences, Papyan NC1) is kept as ``_meta_topic_*``; a
    dataset of ``{text, label}`` items still selects that supervised path.
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Neural Collapse Analysis...")

        # A labelled dataset selects the supervised (Papyan) path.
        if dataset is not None and isinstance(dataset, list) and dataset and (
            isinstance(dataset[0], dict) and {"text", "label"} <= set(dataset[0])
        ):
            return _topic_proxy(model, tokenizer,
                                [(d["text"], d["label"]) for d in dataset])

        samples = list(_TOPIC_BUNDLE)
        # Optional cap on sentences per topic; _meta_n_per_topic records it.
        _n_per = self.config.get("n_sentences_per_topic")
        if isinstance(_n_per, int) and _n_per > 0:
            _by = defaultdict(list)
            for _t, _lab in samples:
                _by[_lab].append((_t, _lab))
            samples = [x for _lab in _by for x in _by[_lab][:_n_per]]
        topic = _topic_proxy(model, tokenizer, samples)
        out = {f"_meta_topic_{k.replace('_meta_', '')}": v for k, v in topic.items()}

        num_samples = int(self.config.get("num_samples", 1000))
        min_per_class = int(self.config.get("min_per_class", 10))
        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)
        texts = [d["text"] if isinstance(d, dict) else str(d)
                 for d in list(dataset)[:num_samples]]
        st = _next_token_class_stats(model, tokenizer, texts, min_per_class)
        if st is None or len(st["counts"]) < 2:
            out.update({"nc1_cdnv": float("nan"), "nc1_cdnv_galanti": float("nan"),
                        "_meta_nc_reason": "fewer than 2 next-token classes "
                                           f"with >= {min_per_class} samples"})
            return out
        out.update({
            "nc1_cdnv": _cdnv(st["means"], st["variances"], dist_exp=2.0),
            "nc1_cdnv_galanti": _cdnv(st["means"], st["variances"], dist_exp=1.0),
            "_meta_nc_n_classes": int(len(st["counts"])),
            "_meta_nc_n_classes_seen": int(st["n_classes_seen"]),
            "_meta_nc_n_samples": int(st["counts"].sum()),
            "_meta_nc_n_tokens": int(st["n_tokens"]),
            "_meta_nc_n_docs": int(st["n_docs"]),
            "_meta_nc_min_per_class": min_per_class,
            "_meta_nc_mean_within_class_variance": float(np.mean(st["variances"])),
        })
        return out
