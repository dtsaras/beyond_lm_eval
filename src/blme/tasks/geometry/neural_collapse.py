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
    # n == K (one example per class) is a DEFINED case, not an error. Sigma_W
    # is then exactly zero and every NC statistic still has a value: on the
    # explicit K=3 centres (1,0), (0,1), (-1,-1) the paper formulas give
    # NC1 = 0.0, equinorm CV = 0.171572875254, ETF deviation = 0.304737854124
    # and cosine std = 0.333333333333, while BLME returned only an error
    # (adversarial review round 2, 2026-09-04).
    if n < K:
        return {"error": "Need at least K samples"}

    # PROMOTE BEFORE THE MEANS, not after.
    #
    # Every NC statistic is defined on CENTRED quantities, so a common offset
    # must cancel exactly -- but reducing float32 features before promoting
    # loses it. On a balanced K=3, n=24 fixture, adding the common offset 1.0
    # (every stored delta exact to +/-2.384185791015625e-07) collapsed
    # NC1/rank/CV/ETF to 0.0/0/NaN/NaN (codex round 5, 2026-09-05). Same defect
    # class as the IsoScore translation loss fixed earlier today.
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

    # Sigma_B: between-class scatter. By construction it has rank
    # ≤ K−1 (the centred class means span a (K-1)-dim affine subspace
    # through the origin), so its full-rank inverse is numerically
    # ill-defined in D ≫ K dimensions.
    centered = class_means - global_mean
    sigma_b = (centered.T @ centered) / K

    # NC1: tr(Σ_W · Σ_B⁺) / K, measured in the (K-1)-dim *Σ_B principal
    # subspace* (Papyan, Han & Donoho 2020 Eq. 3-4 / §2.9, Fig. 6). Naïve
    # ``np.linalg.pinv(Σ_B, rcond=1e-10)`` keeps any eigenvalue above
    # the tiny threshold and inverts it — on real LLM hidden states
    # that promotes near-zero noise eigenvalues to ≈ 1/rcond and makes
    # NC1 explode (3×10⁷ for pythia-70m in the study). We instead
    # eigendecompose Σ_B, keep only the top K−1 eigenvalues (the
    # rest are structural zeros), and rotate Σ_W into that subspace.
    try:
        eigvals, eigvecs = np.linalg.eigh(sigma_b)
        # eigh returns ascending — take the top K-1.
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        k_eff = min(K - 1, d)
        lam = eigvals[:k_eff]
        V = eigvecs[:, :k_eff]

        # Drop directions whose Σ_B eigenvalue is below a
        # relative-tolerance threshold of the top eigenvalue — those
        # are numerical zeros that leaked through eigh.
        top = lam.max() if lam.size else 0.0
        if top <= 0:
            # Sigma_B == 0 is DEFINED, not undefined. NC1 is
            # trace(Sigma_W Sigma_B^+)/K, and the Moore-Penrose inverse of the
            # zero matrix is the zero matrix, so NC1 = 0 exactly -- which is
            # what NumPy pinv, Torch pinv and the released covariance algebra
            # all return for three balanced classes with identical means and
            # nonzero Sigma_W. Returning NaN discarded a real measurement
            # (adversarial review round 2, 2026-09-04).
            nc1 = 0.0
            subspace_rank = 0
        else:
            # Use NUMPY'S PINV RANK TOLERANCE, not a hand-picked 1e-6.
            # NC1 is trace(Sigma_W Sigma_B^+)/K, and 1e-6 relative deletes
            # genuine directions: on the balanced analytic fixture with
            # Sigma_B eigenvalues (1, 5e-7, 0) it returned NC1=0.0/rank=1
            # where numpy.pinv and torch.pinv both give 0.333333333333 and
            # the released top-(K-1) SVD algebra gives 0.333333365984
            # (adversarial review, 2026-09-04).
            # AMBIENT-d DEPENDENCE, examined 2026-09-07 and kept deliberately.
            #
            # A round-9 review reported that appending ONE zero coordinate
            # (d=47 -> 48) takes NC1 from 0.6666666666666666 to 0.0, since NC1
            # is invariant under zero-padding. Investigated:
            #
            #  * It does NOT reproduce on ordinary data. On a non-degenerate
            #    balanced K=3, n=20, d=47 fixture, NC1 = 0.0033624825287505047
            #    is unchanged at d=48/49/52 and across scales 1e-30 .. 1e25.
            #  * The mechanism is real but measure-zero: `_rcond` scales with
            #    `max(sigma_b.shape)`, so d=47 -> 48 moves the keep-threshold
            #    from 1.043610e-14 to 1.065814e-14. Only an eigenvalue inside
            #    that window -- relative width one eps, 2.2e-16 -- changes its
            #    keep decision. A fixture has to be engineered onto it.
            #  * `numpy.linalg.pinv` uses this exact convention
            #    (`max(M.shape) * eps * largest_singular_value`), and matching
            #    numpy is why the previous hand-picked 1e-6 was replaced. So
            #    the alternative is not "be invariant", it is "diverge from the
            #    reference tolerance in order to be invariant on a set of
            #    measure zero".
            #
            # Left as numpy's. Changing it to depend on the rank-relevant
            # dimension instead of the ambient one would buy zero-padding
            # invariance at the cost of the reference convention, which is the
            # wrong trade under this campaign's own rule.
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
        # ddof=1, matching the released Papyan notebook's `torch.std`.
        # `np.std` defaults to ddof=0, which is low by exactly sqrt(K/(K-1)):
        # the ratio is 1.154701 = sqrt(4/3) on all five K=4 fixtures, and on
        # gpt2 BLME read 0.155268 against the official 0.173595, -10.56%
        # (independent Opus review, 2026-09-04).
        nc2_equinorm_cv = float(np.std(norms, ddof=1) / norms.mean())
    else:
        nc2_equinorm_cv = float("nan")

    # NC2 — ETF cosine deviation proxy (not full normalized Gram NC2).
    # Pairwise cosine similarities between centered class means.
    # Exact cosine: an ETF is an ETF at every positive scale alpha, which the
    # paper states explicitly. The +1e-12 form reported deviation 9.999e-13
    # for an exact K=3 ETF at alpha=1 but 0.375 at alpha=1e-12 and 0.499999501
    # at alpha=1e-15, instead of 0 throughout (adversarial review, 2026-09-04).
    # Scale-exact normalisation (see utils.safe_unit_rows), and an EXACTLY
    # zero centred class mean stays undefined rather than becoming the zero
    # vector: with all three class means coincident, dividing by finfo.tiny
    # reported ETF deviation 0.5 and cosine std 0.0 when every paper cosine is
    # 0/0 (adversarial review round 2, 2026-09-04).
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
        # Papyan et al. report THREE NC2 statistics; this is the third and it
        # was missing. Std of the pairwise cosines (their Fig. 3) measures
        # equiangularity independently of the mean deviation above: a set can
        # sit at the ETF target on average while being highly variable, and
        # only this statistic sees that.
        nc2_cosine_std = float(np.std(pair_cos))
    else:
        nc2_etf_cosine_deviation_proxy = float("nan")
        nc2_cosine_std = float("nan")

    return {
        "nc1_within_class_collapse": nc1,
        "nc1_subspace_rank": int(subspace_rank),
        # GATED OUT OF THE FEATURE MATRIX (_meta_ prefix).
        #
        # All THREE NC2 statistics fail a 200-draw label-permutation test on
        # every model measured -- 9 tests (3 statistics x 3 models), none below
        # p = 0.05, best 19/200 = 0.095, and nc2_equinorm_cv on pythia-70m at
        # 196/200 is in the WRONG DIRECTION. A feature whose observed value
        # lies inside its own label-permutation null cannot support a claim
        # about the model: the same value arises from random labels. NC1 by
        # contrast passes at p = 0/200 on all three and stays in the matrix.
        #
        # The formulas are exact (verified 0.0e+00 to 8.5e-14 against Papyan's
        # own definitions) -- this is a POWER problem, not a correctness one:
        # 5 classes x 8 sentences at d = 512-1280 cannot resolve NC2. The real
        # fix is a larger bundle (30-50 sentences per topic), which
        # `n_sentences_per_topic` now exposes; until the bundle grows these
        # columns must not enter the analysis (adversarial review, 2026-09-04).
        "_meta_nc2_equinorm_cv": nc2_equinorm_cv,
        "_meta_nc2_etf_cosine_deviation_proxy": nc2_etf_cosine_deviation_proxy,
        "_meta_nc2_cosine_std": nc2_cosine_std,
        "_meta_n_per_topic": int(round(n / K)) if K else 0,
        "_meta_n_classes": int(K),
        "_meta_n_samples": int(n),
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
            # Expose the bundle size. The NC2 statistics are underpowered at
            # the shipped 8 sentences per topic -- all three sit inside their
            # own label-permutation null -- and the fix is more sentences per
            # class, not a different formula. Capping here lets a caller shrink
            # the bundle for a fast run; growing it requires extending
            # _TOPIC_BUNDLE itself, and `_meta_n_per_topic` records what was
            # actually used so an underpowered run is identifiable.
            _n_per = self.config.get("n_sentences_per_topic")
            if isinstance(_n_per, int) and _n_per > 0:
                from collections import defaultdict as _dd
                _by = _dd(list)
                for _t, _lab in samples:
                    _by[_lab].append((_t, _lab))
                samples = [x for _lab in _by for x in _by[_lab][:_n_per]]

        device = next(model.parameters()).device

        # Mean-pool the final-layer hidden states for each sample.
        # ``hidden_states[-1]`` is already post-final_norm in
        # transformers 5.x, so it already lives in the LM-head input
        # space — no need to apply ``final_norm`` again.
        feats = []
        labs = []
        with torch.no_grad():
            for text, label in tqdm(samples, desc="Neural collapse"):
                inputs = tokenizer(text, return_tensors="pt",
                                   truncation=True, max_length=128).to(device)
                if inputs["input_ids"].shape[1] < 1:
                    continue
                out = model(**inputs, output_hidden_states=True)
                # transformers 5.x: ``hidden_states[-1]`` is already
                # post-final_norm (verified in logit_lens where
                # ``lm_head(hidden_states[-1]) == outputs.logits``). The
                # previous code re-applied ``final_norm`` here which
                # double-normalised the features and gave spuriously
                # small NC1 for RMSNorm architectures.
                # post-final-norm: DELIBERATELY post-norm. Neural collapse is a
                # statement about the FEATURE LAYER FEEDING THE CLASSIFIER, which for
                # an LM is the post-`ln_f` vector the head reads.
                h = out.hidden_states[-1][0]  # (T, D) — already normed
                # Mean-pool, ignoring padding (causal LMs typically don't
                # have padding here since we tokenize one text at a time).
                # The trailing .float() DISCARDED the precision the .double() was added to
                # gain: promote, average, then narrow back is self-defeating. Removing only
                # this cast moved geometry_correlation_dimension from 7.486064047571674 to
                # 15.295160200026809 on native Pythia-70m hidden states (+104%), matching an
                # independently reconstructed Grassberger-Procaccia value, and restored a
                # lossless translation from 0.0 to 2.3540184486398976 (codex round 15,
                # 2026-09-09). Swept as a class: 7 sites carried this exact pattern.
                pooled = h.double().mean(dim=0).cpu().numpy()
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
