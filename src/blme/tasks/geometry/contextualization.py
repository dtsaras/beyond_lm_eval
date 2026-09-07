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
    # `safe_unit_rows`, not a raw norm.
    #
    # Ethayarajh's Definitions 1-2 are exactly invariant under any positive
    # common rescaling, and this was not: the corrected SelfSim/IntraSim pair
    # -0.126126798 / 0.283469260 became 0.0 / NaN at 1e-30 and 0.0 / 0.0 at
    # 1e25 and 2.406159650500376e38 (gpt-5.6-sol round 9, 2026-09-07).
    #
    # `np.linalg.norm` SQUARES before summing, so it underflows to 0 long
    # before the data does -- and then `np.where(norms > 0, norms, 1.0)`
    # divides the row by *one*, leaving the raw tiny vector in place and
    # driving every cosine to 0. The guard reads like a zero-row guard but
    # fires on rows that are merely small, which is the failure it produces.
    # The MEAN in float64. Summing n rows near the top of the float32 range
    # overflows the accumulator before the division ever happens -- 24 rows at
    # 2.4e38 sum to 5.8e39 and the sentence mean becomes inf, which is the last
    # surviving NaN on the ladder after the normalisation above was fixed. This
    # is the third instance of "a float32 reduction overflows" found on
    # 2026-09-07 (see also `matrix_entropy` and `collapse`); it is filed as a
    # population in PENDING_FIXES.md item 15b.
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
    (kawine/contextual `analyze.py::variance_explained_by_pc`, which is
    `sklearn.decomposition.PCA(n_components=1).explained_variance_ratio_[0]`
    -- and PCA centers).

    On 2026-09-03 this was briefly changed to an uncentered
    sigma_1^2 / sum sigma_i^2, on the reading that Definition 3 states no mean
    subtraction. That was WRONG, and the released artifacts settle it: the
    repo's shipped gpt2/variance_explained.csv gives a mean raw MEV of
    0.286-0.351 over layers 2-11, reproducing the paper's own sentence that
    raw MEV is "around 30% on average for layers 2 to 11" for GPT-2. The
    uncentered convention reads ~0.99 there. Centering is the only convention
    in the published regime, and it also matched the official code to 3.3e-16
    before the change.

    Not centering additionally turns the kernel into a norm-outlier detector:
    on gpt2's mid layers the top 10 of 1000 pooled vectors carry 60-64% of
    total energy, which is why the uncentered baseline sits at 0.94-0.97 there.
    """
    n = vectors.shape[0]
    if n < 2:
        return float("nan")
    centered = np.asarray(vectors, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    try:
        # `safe_svdvals`: numpy/torch compute the SVD internally and then CAST
        # the singular values back to the input dtype, and that cast overflows
        # to inf near 3.4e38 -- so a scale-FREE consumer downstream (effective
        # rank, RankMe, IsoScore, alpha) returns NaN on data it should handle
        # exactly. The helper factors out the max first, which only shifts
        # exponents (2026-09-07).
        from .utils import safe_svdvals
        S = safe_svdvals(centered)
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
    # `safe_unit_rows` -- same underflow as `_intra_sentence_mean_cosine`
    # above; the anisotropy baseline is subtracted from both statistics, so a
    # collapse here corrupts the corrected values even when the raw ones survive.
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
    ``TruncatedSVD(n_components=100).explained_variance_ratio_[0]``. That is
    NOT the uncentered energy ratio S0^2 / sum(S^2): sklearn defines it as

        Var(X @ v0) / sum_j Var(X_j)

    where the DIRECTION v0 comes from an uncentered SVD (that is what makes it
    TruncatedSVD rather than PCA) but the numerator and denominator are both
    CENTERED variances. Three distinct quantities are in play here, and an
    earlier version of this function returned the wrong third one: on BLME's own
    gpt2 pools it read 0.5796 against the official 0.0107 at the first block
    output -- a factor of 54 -- and the centred-PCA helper it replaced was
    closer at all five reported layers (adversarial review, 2026-09-04).

    The per-word MEV path is unrelated and unchanged: the official uses sklearn
    PCA there, which centres, and it reproduces the released CSVs.
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

    ``size_matched=True`` (the default) draws random groups whose sizes MATCH the
    real per-word occurrence counts. This is required for the subtraction to
    mean anything, because MEV is strongly n-dependent and the two terms are
    otherwise estimated at wildly different n: in our corpus a tracked word has
    a MEDIAN of 9 occurrences while the pool has 1000. Measured on the same gpt2
    pool, the size-matched baseline is 0.373 against 0.975 for the flat 1K pool,
    and using the 1K pool makes the "adjusted" MEV +0.516 at the reference's
    layer 1 where the authors' own released file gives -0.138 -- a sign flip and
    10x the paper's <5% bound (adversarial review, 2026-09-03).

    ``size_matched=False`` reproduces the literal reading of Sec. 3.4 ("both the
    raw measure and baseline are estimated with 1K uniformly randomly sampled
    word representations"), which is self-consistent in the paper's own setting
    because its SemEval-STS words have far more contexts each than ours do. It
    is reported separately as ``mev_baseline_pool1k`` so the deviation is visible
    rather than silent.
    """
    all_vectors = np.asarray(all_vectors)
    n = all_vectors.shape[0]
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    if not size_matched:
        # The pool-1K baseline exists to reproduce the paper's literal reading,
        # so it uses the paper's algorithm: uncentered TruncatedSVD.
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

        # Ethayarajh 2019 uses ~1000 sentences; 200 was a speed default that
        # left per-word occurrence counts too thin for a stable top-PC
        # estimate in MEV (adversarial review, 2026-09-02).
        num_samples = self.config.get("num_samples", 1000)
        # Words to track must appear ≥ min_word_occurrences contexts to
        # be included in the self-similarity / MEV averages.
        min_word_occurrences = self.config.get("min_word_occurrences", 5)
        # Layers to report. None ⇒ ~5 evenly-spaced layers; "all" ⇒ all.
        layer_subset = self.config.get("layers", None)

        if dataset is None:
            # Distinct sentences that reuse common words across DIFFERENT
            # contexts (audit 2026-07-16: the previous 50-identical-sentence
            # fallback made SelfSim and MEV identically 1.0 — every tracked
            # occurrence was the same vector — and drew the anisotropy
            # baseline entirely from same-context pairs).
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

        # POST-NORM AT THE DEEPEST LAYER, BECAUSE THE REFERENCE DEFINES IT SO.
        #
        # The 2026-09-05 sweep hooked `layers[-1]` here on the general rule that
        # a per-layer PROFILE must be homogeneous. That rule has a third clause
        # this task is the case for: **the released reference wins**.
        # `kawine/contextual`'s own vendored GPT-2
        # (`pytorch_pretrained/modeling_gpt2.py:562-565`) does
        #
        #     all_hidden_states.pop()   # ignore the last layer prior to layer norm
        #     all_hidden_states.append(self.ln_f(hidden_states))
        #
        # i.e. Ethayarajh's deepest layer IS the post-LayerNorm state, by an
        # explicit line of code with a comment saying why. Substituting the raw
        # block there stops computing the reference's quantity: on gpt2 the
        # layer-11 `self_similarity_corrected` moves 0.004359 -> 0.070297 and
        # pythia-160m's 0.013265 -> 0.025363 (independent Opus review,
        # 2026-09-05).
        #
        # So the hook is NOT installed. The mixed-convention profile that
        # results -- entries 1..L-1 block outputs, entry L a LayerNorm output --
        # is the reference's own shape, and the concern about it belongs on
        # `comparable`, not on faithfulness. `_meta_last_entry_is_block_output`
        # reports False so nothing downstream has to guess.
        #
        # This is the second task (with `geometry_matrix_entropy`) where the
        # sweep's blanket rule was wrong because a released reference had
        # already made the choice.
        with torch.no_grad():
            for i, sample in enumerate(tqdm(dataset, desc="Contextualization")):
                if i >= num_samples:
                    break

                text = sample["text"] if isinstance(sample, dict) else str(sample)
                inputs = tokenizer(text, return_tensors="pt",
                                   truncation=True, max_length=128).to(device)
                if inputs["input_ids"].shape[1] < 4:
                    continue

                # post-final-norm=False, DELIBERATELY -- the reference's own
                # vendored model pops the pre-norm final layer and appends
                # ln_f(...). See the note above.
                outputs = model(**inputs, output_hidden_states=True)
                hidden = list(outputs.hidden_states)  # (L+1) of (1, T, D)
                token_ids = inputs["input_ids"][0].cpu().tolist()
                # Exclude special tokens (audit 2026-07-16: on BOS-adding
                # tokenizers the position-0 BOS state is context-independent
                # — SelfSim/MEV exactly 1.0 — inflating the averages; the
                # reference tracks real words only).
                special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
                # Skip the embedding layer (index 0); keep block outputs 1..L.
                per_layer = hidden[1:]
                if not n_layers_seen:
                    n_layers_seen = len(per_layer)

                # Baseline pool: ONE token vector per sentence per layer
                # (reference protocol: random pairs must not share a context;
                # audit 2026-07-16). The position is chosen ONCE per sentence,
                # UNIFORMLY AT RANDOM over the non-special tokens, and reused
                # for every layer.
                #
                # It used to be non_special[len//2] -- the positional MIDDLE of
                # every sentence (adversarial review, 2026-09-02). Token
                # representations depend strongly on position, so a fixed
                # relative position makes the "random pair" pool a sample from
                # one narrow positional slice rather than from the token
                # distribution, biasing the anisotropy baseline that is then
                # SUBTRACTED from SelfSim and IntraSim. Choosing per sentence
                # rather than per (sentence, layer) also keeps the pool
                # consistent across layers, so the layer profile compares like
                # with like.
                non_special = [p_ for p_, t_ in enumerate(token_ids)
                               if t_ not in special_ids]
                non_special_all = non_special
                pool_pos = (int(rng_pool.choice(non_special))
                            if non_special else None)

                # Positions IntraSim is allowed to see. SelfSim and MEV both
                # exclude special tokens; IntraSim was appending the FULL (T, D)
                # sequence, so on BOS-adding tokenizers it averaged a
                # context-independent BOS state into every sentence. Measured on
                # Llama-3.2-1B layer 3: 0.1852 as shipped vs 0.5040 correct, a
                # 2.7x error -- and exactly 0.0000 on gpt2, which adds no BOS.
                # That is an architecture-family step change, not noise.
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
            # BOTH corrections are emitted and NEITHER is privileged. The two
            # protocols disagree in SIGN on our corpus (raw 0.562, size-matched
            # baseline 0.486 -> +0.076; flat 1K pool 0.782 -> -0.220), against
            # the authors' released adjusted values of -0.138 / -0.116.
            #
            # The principled argument favours size matching: subtracting a
            # baseline estimated at n=1000 from a raw measure estimated at a
            # MEDIAN of n=9 is not a valid subtraction, because MEV is strongly
            # n-dependent. The textual argument favours the flat pool: Sec. 3.4
            # says both terms use 1K samples -- which is self-consistent in the
            # paper's setting because its SemEval-STS words have far more
            # contexts each than our corpus gives.
            #
            # We cannot settle it from here, so we do not pretend to: the
            # unqualified `mev_corrected` key is deliberately NOT emitted, and a
            # caller must name the protocol it wants. The adjusted MEV should
            # not be used as evidence for or against Ethayarajh's <5% claim
            # under either convention (adversarial review, 2026-09-03).
            mev_corr_sm = (mev_mean - mev_baseline
                           if not (np.isnan(mev_mean) or np.isnan(mev_baseline))
                           else float("nan"))
            mev_corr_1k = (mev_mean - mev_baseline_1k
                           if not (np.isnan(mev_mean) or np.isnan(mev_baseline_1k))
                           else float("nan"))

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
                "mev_corrected_sizematched": mev_corr_sm,
                "mev_corrected_pool1k": mev_corr_1k,
                "mev_baseline": mev_baseline,
                "mev_baseline_pool1k": mev_baseline_1k,
                "anisotropy_baseline": baseline,
                # _meta_: this is a pure tokenizer statistic (r = -0.905 with
                # log(vocab) over 13 tokenizers) and is computable with no model
                # weights at all, yet it ranked #2 by partial correlation in
                # TOP_PREDICTORS. It must not enter the feature matrix.
                "_meta_n_words_tracked": len(self_sims),
            }

        # Compact aggregate summary across the reported layers.
        def _agg(key):
            xs = [v[key] for v in per_layer_metrics.values() if not np.isnan(v[key])]
            return float(np.mean(xs)) if xs else float("nan")

        return {
            # False, and deliberately so: the deepest reported layer is the
            # post-`ln_f` state, which is what the released reference defines.
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
            # `int`, not `np.int64`: `np.linspace(...).astype(int)`
            # yields numpy scalars, which `json.dump` rejects under
            # strict encoders (independent Opus review, 2026-09-05).
            "layers_reported": [int(i) for i in layers_to_report],
        }
