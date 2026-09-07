"""
Latent Mahalanobis OOD Distance Task
──────────────────────────────────────────────────────────────────────
Evaluates Out-Of-Distribution (OOD) detection capabilities by measuring
the Mahalanobis distance of hidden states from a reference text distribution.

A robust model encodes OOD/anomalous text geometrically far from the
centroid of its normal distribution, normalized by the covariance
(Mahalanobis distance).

References:
- "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks" (Lee et al., NeurIPS 2018)
  ADAPTED, not implemented: this task uses a SINGLE centroid (C=1) with a
  Ledoit-Wolf shrinkage covariance, where Lee et al. Eq. 2 uses per-class
  means, a pooled within-class covariance and a max over classes. The
  difference is not cosmetic: on a bimodal in-distribution cloud with the
  OOD points at the global mean, this estimator scores AUROC 0.021 (mean
  over 20 seeds) where the two-class Lee form scores 1.000. Ledoit-Wolf
  also breaks the affine invariance that gives a Mahalanobis distance its
  meaning (41.9% distance change and Spearman 0.270 under a diagonal
  rescale, against 9.4e-12 for an empirical covariance at n > d).
"""

import numpy as np
import torch
from scipy.spatial.distance import mahalanobis

try:
    from sklearn.covariance import LedoitWolf
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from ...tasks.base import DiagnosticTask
from ...registry import register_task
import logging
logger = logging.getLogger("blme")


# A pool of topically DISTINCT in-distribution English sentences for the
# dataset=None structural fallback. They must be genuinely diverse (not
# near-duplicates) so a train/held-out split produces distinct hidden-state
# rows — the old fallback cycled 5 sentences, so the "held-out" ID score
# rows were exact duplicates of the fit rows (artificial ~0 ID Mahalanobis
# and a spurious AUROC = 1.0).
_DEFAULT_ID_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models learn patterns from data.",
    "Natural language processing is a subfield of artificial intelligence.",
    "The weather forecast predicts rain for tomorrow.",
    "Scientists discovered a new species in the deep ocean.",
    "The ancient library held thousands of fragile manuscripts.",
    "She planted tomatoes and basil in the garden this spring.",
    "The orchestra tuned their instruments before the concert began.",
    "Electric cars are becoming more common on city streets.",
    "The mountain trail offered a stunning view of the valley.",
    "He repaired the old bicycle in the garage over the weekend.",
    "Coffee prices rose sharply after the unexpected frost.",
    "The children built an elaborate sandcastle near the tide.",
    "A gentle breeze carried the scent of pine through the forest.",
    "The startup released its new mobile app to enthusiastic reviews.",
    "Historians debated the true cause of the empire's collapse.",
    "The chef simmered the sauce for hours to deepen its flavor.",
    "Astronomers detected a faint signal from a distant galaxy.",
    "The marathon runners trained through the cold winter months.",
    "Rain tapped softly against the windows all afternoon.",
    "The museum unveiled a rare collection of Renaissance paintings.",
    "Bees pollinate a large share of the world's food crops.",
    "The lawyer reviewed the contract line by line before signing.",
    "Fresh snow blanketed the quiet village overnight.",
    "The engineer tested the bridge design under simulated load.",
    "A curious cat watched the birds from the kitchen sill.",
    "The novelist wrote every morning before the sun came up.",
    "Volunteers cleaned the riverbank and sorted the recycling.",
    "The satellite adjusted its orbit to avoid the debris field.",
    "Warm bread and butter waited on the wooden table.",
]


def _default_id_dataset(num_samples: int):
    """Distinct in-distribution samples for the dataset=None fallback.

    Returns up to ``num_samples`` topically-distinct sentences WITHOUT cycling,
    so an index-based train/held-out split is genuinely text-disjoint.
    """
    n = max(1, min(num_samples, len(_DEFAULT_ID_TEXTS)))
    return [{"text": _DEFAULT_ID_TEXTS[i]} for i in range(n)]


def _as_id_list(x) -> list:
    """Flatten a tokenizer's input_ids to a plain list of python ints.

    Tokenizers return ids as a list, a nested [[...]] batch, an ndarray or a
    torch tensor depending on `return_tensors` and on whether the tokenizer is
    a real one or a stub, so normalise before doing arithmetic on them.
    """
    if hasattr(x, "tolist"):
        x = x.tolist()
    else:
        x = list(x)
    while len(x) == 1 and isinstance(x[0], (list, tuple)):
        x = list(x[0])
    return [int(v) for v in x]


def _compute_mahalanobis_distances(X_train, X_test, diag=None):
    """
    Computes Mahalanobis distances for X_test relative to the distribution of X_train.
    Uses Ledoit-Wolf shrinkage covariance estimator for numerical stability,
    especially when n_samples << n_features.

    Args:
        X_train: (N_train, D) numpy array of reference hidden states
        X_test: (N_test, D) numpy array of OOD hidden states

    Returns:
        List of distances for each sample in X_test.
    """
    try:
        # Centroid
        mu = np.mean(X_train, axis=0)

        # Covariance estimation
        if HAS_SKLEARN:
            # Ledoit-Wolf shrinkage covariance estimator
            lw = LedoitWolf()
            lw.fit(X_train)
            inv_cov = lw.precision_  # already the inverse covariance
            # Surface the REALISED shrinkage. Shrinking toward an isotropic
            # target trades affine invariance -- the property that makes a
            # Mahalanobis distance meaningful -- for conditioning, and the
            # amount differs by an order of magnitude across the zoo
            # (0.068 gpt2 vs 0.606 pythia-70m on the same protocol), so
            # models are NOT scored by the same function. Reporting it makes
            # that visible instead of silent.
            if diag is not None:
                diag["shrinkage"] = float(getattr(lw, "shrinkage_", float("nan")))
        else:
            # Fallback: regularized sample covariance
            cov = np.cov(X_train, rowvar=False)
            cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
            inv_cov = np.linalg.pinv(cov_reg)

        if diag is not None:
            diag["d"] = int(X_train.shape[1])
            diag["n_fit"] = int(X_train.shape[0])

        distances = []
        for x in X_test:
            dist = mahalanobis(x, mu, inv_cov)
            distances.append(dist)

        return distances
    except Exception as e:
        logger.info(f"Error computing Mahalanobis: {e}")
        return []


def _collect_mean_pooled_hidden_states(
    model, tokenizer, dataset, num_samples: int, max_length: int = 128,
) -> torch.Tensor:
    """Collect one final-layer mean-pooled representation per text sample."""
    samples = list(dataset)[:num_samples]
    if not samples:
        return torch.empty(0)

    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    reps = []
    with torch.no_grad():
        for sample in samples:
            # A sample may carry PRE-TOKENIZED ids. The token-shuffle OOD set
            # does, because decoding a shuffled id list and letting the model
            # re-tokenize the string does NOT preserve the token multiset --
            # merges recombine across the new adjacencies. Measured on 50
            # WikiText texts, the re-tokenized multiset differed from the
            # shuffled one in 38/50 (gpt2) and 40/50 (pythia-70m), so the
            # docstring's "preserves vocabulary" was false for most inputs
            # (adversarial review, 2026-09-03). Feeding ids straight to the
            # model makes it exactly true.
            if isinstance(sample, dict) and sample.get("input_ids") is not None:
                ids = _as_id_list(sample["input_ids"])[:max_length]
                inputs = {
                    "input_ids": torch.tensor([ids], dtype=torch.long),
                    "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                }
            else:
                text = sample["text"] if isinstance(sample, dict) and "text" in sample else str(sample)
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
            if hasattr(inputs, "to") and callable(inputs.to):
                inputs = inputs.to(device)
            else:
                inputs = {
                    k: (v.to(device) if hasattr(v, "to") else v)
                    for k, v in inputs.items()
                }

            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = getattr(outputs, "hidden_states", None)
            if not hidden_states:
                continue

            # post-final-norm: DELIBERATELY post-norm. Lee et al. 2018 fit
            # their Gaussians to THE FEATURES FEEDING THE CLASSIFIER, which for
            # an LM is the post-`ln_f` vector the head consumes -- the same
            # reading `geometry_neural_collapse` takes. Recorded for the
            # record: scoring the raw block output instead separates BETTER
            # (gpt2 AUROC 0.9296 -> 0.9904, pythia-70m 0.6448 -> 0.7616), but
            # "separates better" is not an argument about which estimand the
            # reference defines, and this task reports a single layer, so no
            # per-layer profile mixes conventions.
            h = hidden_states[-1][0].detach().float().cpu()
            attention_mask = inputs.get("attention_mask") if isinstance(inputs, dict) else None
            if attention_mask is not None and hasattr(attention_mask, "detach"):
                mask = attention_mask[0].detach().cpu().bool()
                if mask.numel() == h.shape[0] and bool(mask.any()):
                    h = h[mask]
            if h.numel() > 0:
                reps.append(h.mean(dim=0))

    return torch.stack(reps, dim=0) if reps else torch.empty(0)


@register_task("geometry_mahalanobis")
class MahalanobisOODTask(DiagnosticTask):
    """
    Computes the Mahalanobis distance between an In-Distribution (ID)
    reference dataset and an Out-Of-Distribution (OOD) dataset at the final layer.

    OOD data is generated by token-level shuffling of each text's OWN token
    ids, permuted IN PLACE and fed to the model as ids (preserves vocabulary
    exactly -- the earlier decode/re-tokenize round trip did not, differing in
    41/50 gpt2 and 43/50 pythia-70m WikiText texts,
    destroys syntax) rather than character scrambling.

    Reports the average ID distance vs OOD distance, the separation gap,
    and AUROC.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Latent Mahalanobis OOD Detection...")
        num_samples = self.config.get("num_samples", 50)
        max_length = int(self.config.get("max_length", 128))
        ood_strategy = self.config.get("ood_strategy", "shuffle")  # "shuffle" or "random"

        def _shuffle_tokens(text, tokenizer, rng):
            """Permute a text's OWN token ids in place; returns a sample dict.

            Tokenizes with the SAME call the ID path uses, then permutes only
            the non-special positions. The returned token multiset is therefore
            an exact permutation of the ID sample's -- vocabulary preserved by
            construction, syntax destroyed -- and no decode/re-tokenize round
            trip can perturb it.
            """
            # Prefer the same __call__ the ID path uses so the two token
            # sequences are produced identically; fall back to .encode for
            # tokenizers that only expose that.
            try:
                enc = tokenizer(text, truncation=True, max_length=max_length)
                ids = _as_id_list(enc["input_ids"])
            except TypeError:
                ids = _as_id_list(
                    tokenizer.encode(text, add_special_tokens=False))[:max_length]
            special = set(getattr(tokenizer, "all_special_ids", None) or [])
            movable = [i for i, t in enumerate(ids) if t not in special]
            vals = [ids[i] for i in movable]
            rng.shuffle(vals)
            out = list(ids)
            for pos, v in zip(movable, vals):
                out[pos] = v
            return {"input_ids": out, "text": tokenizer.decode(out)}

        rng = np.random.RandomState(42)

        # We need two pseudo-datasets for structural evaluation if none provided:
        # 1. ID: Normal English
        # 2. OOD: Depends on ood_strategy
        if dataset is None:
             # Distinct (non-cycled) ID sentences so the held-out split below
             # scores genuinely new texts, not duplicates of the fit set.
             dataset_id = _default_id_dataset(num_samples)
             if ood_strategy == "random":
                 # Generate random token sequences
                 import string
                 dataset_ood = [{"text": " ".join("".join(rng.choice(list(string.ascii_lowercase), size=rng.randint(3, 8))) for _ in range(10))} for _ in range(num_samples)]
             else:
                 # Token-level shuffling: preserves vocabulary, destroys syntax
                 dataset_ood = [_shuffle_tokens(d["text"], tokenizer, rng) for d in dataset_id]
        else:
            # For standard benchmark, we split the given dataset in half and perturb the second half
            samples = list(dataset)[:num_samples*2]
            mid = max(1, len(samples) // 2)
            dataset_id = samples[:mid]
            if ood_strategy == "random":
                import string
                dataset_ood = [{"text": " ".join("".join(rng.choice(list(string.ascii_lowercase), size=rng.randint(3, 8))) for _ in range(10))} for _ in samples[mid:]]
            else:
                # PAIRED token-level shuffling: shuffle the SAME texts that
                # form the ID set, not a disjoint half.
                #
                # Shuffling a DIFFERENT half makes the comparison confounded --
                # the OOD score then reflects both the shuffling and whatever
                # distinguishes those particular sentences. A paired re-run
                # moved pythia-70m's AUROC from **0.824 to 0.598**, so most of
                # the apparent detection was the unpaired split (adversarial
                # review, 2026-09-03).
                #
                # Paired also means the covariance is fit on texts that are NOT
                # scored, which keeps the held-out property: the fit uses the
                # second half, the scoring uses the first half and its own
                # shuffles.
                dataset_ood = []
                for s in dataset_id:
                    text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                    dataset_ood.append(_shuffle_tokens(text, tokenizer, rng))


        if len(dataset_id) < 5 or len(dataset_ood) < 5:
            return {"error": "Need at least 5 ID and 5 OOD samples to compute stable covariance."}

        # Collect one final-layer representation per text sample, mean-pooled
        # over sequence positions. Token-flattening would inflate the
        # covariance population by sequence length and violate the sample-level
        # OOD framing.
        X_id = _collect_mean_pooled_hidden_states(
            model, tokenizer, dataset_id, num_samples=len(dataset_id), max_length=max_length,
        )
        X_id = X_id.float().numpy()

        X_ood = _collect_mean_pooled_hidden_states(
            model, tokenizer, dataset_ood, num_samples=len(dataset_ood), max_length=max_length,
        )
        X_ood = X_ood.float().numpy()

        if X_id.shape[0] < 5 or X_ood.shape[0] < 5:
            return {"error": "Need at least 5 ID and 5 OOD sample representations."}

        # Lee et al. 2018 scores **held-out** samples against a Gaussian
        # fitted on a distinct training split. Re-using the same ID
        # samples for both fit and score produces in-sample Mahalanobis
        # distances that are systematically biased downward by leverage
        # (~ p/n in the moderate-D regime, worse under D >> n and even
        # with Ledoit-Wolf shrinkage). Split the ID set roughly 50/50
        # so the ID-distance baseline matches the OOD-distance statistic.
        #
        # NOT A PREFIX/SUFFIX CUT. The corpus is
        # `load_default_corpus`, which returns CONSECUTIVE WikiText passages,
        # so a contiguous split puts different ARTICLES on the two sides: at
        # the shipped 100 samples, rows 0-24 were 24 paragraphs of one lobster
        # article plus one of a biography, while rows 25-49 were that biography
        # (14), a highway article (6) and a manga character (5). The
        # "in-distribution" baseline was therefore measuring a TOPIC SHIFT, and
        # every derived column inherited it. Re-assigning the SAME 50 passages
        # (red-team review, 2026-09-05; the figures below are for the stride
        # split that was the first attempt at this, superseded by the
        # permutation described further down):
        #
        #                                    gpt2               pythia-70m
        #   mahalanobis_sq_per_dim_id    1.4552 -> 0.5444    1.6878 -> 1.6387
        #                                   (-63%)
        #   mean_mahalanobis_id          32.35  -> 19.89     28.68  -> 28.32
        #   ood_separation_gap_per_dim   1.7728 -> 1.4090    0.2430 -> 0.4818
        #                                   (-21%)              (+98%)
        #   auroc                        0.9296 -> 0.9792    0.6448 -> 0.7584
        #
        # The 2026-09-03 pairing note below claimed to remove "the
        # sentence-identity confound"; it fixed the OOD arm only, and the ID
        # arm kept a confound of its own.
        # A SEEDED RANDOM PERMUTATION, not a stride.
        #
        # The stride-2 split introduced above fixed the contiguous split's topic
        # shift and then over-corrected in the other direction: on consecutive
        # WikiText passages, stride-2 pairs every SCORED passage with its
        # immediate same-article NEIGHBOURS in the fit set. Five seeded random
        # 25/25 splits of the same 50 passages give
        # `mahalanobis_sq_per_dim_id` 0.5721 / 0.6093 / 0.6511 / 0.6606 /
        # 0.6764 (mean 0.634); the stride gives **0.5444, below all five**,
        # where the contiguous split gave 1.4552 (+130% vs random)
        # (independent Opus review, 2026-09-05).
        #
        # A permutation is ordering-independent BY CONSTRUCTION, which neither
        # of the other two is -- it does not matter how the corpus is sorted.
        # Seeded so the split is reproducible.
        n_id = X_id.shape[0]
        fit_n = max(2, min(n_id - 2, max(5, n_id // 2)))
        _perm_rng = np.random.default_rng(int(self.config.get("seed", 42)))
        _perm = _perm_rng.permutation(n_id)
        _fit_idx = np.sort(_perm[:fit_n])
        _score_idx = np.sort(_perm[fit_n:])
        X_id_fit = X_id[_fit_idx]
        X_id_score = X_id[_score_idx]
        # PAIRING: when the OOD set is a per-text shuffle of the ID set, row i
        # of X_ood is the shuffle of row i of X_id, so restricting to the same
        # INDEX SET compares each SCORED text against its OWN shuffle. That
        # removes the sentence-identity confound: an unpaired split moved
        # pythia-70m's AUROC from 0.598 (paired) to 0.824 (adversarial review,
        # 2026-09-03). The index set is now the interleaved complement rather
        # than a suffix, but the pairing property is unchanged.
        _paired_ood = (X_ood.shape[0] == X_id.shape[0])
        X_ood_full = X_ood
        if _paired_ood:
            X_ood = X_ood[_score_idx]
        insufficient_holdout = X_id_score.shape[0] < 2

        if insufficient_holdout:
            dists_ood = _compute_mahalanobis_distances(X_id_fit, X_ood)
            if not dists_ood:
                return {
                    "error": "Failed to compute Mahalanobis distance (likely covariance singularity).",
                    "insufficient_holdout": True,
                    "n_id_fit": int(X_id_fit.shape[0]),
                    "n_id_score": int(X_id_score.shape[0]),
                    "n_ood_score": int(X_ood.shape[0]),
                }
            return {
                "insufficient_holdout": True,
                "n_id_fit": int(X_id_fit.shape[0]),
                "n_id_score": int(X_id_score.shape[0]),
                "n_ood_score": int(X_ood.shape[0]),
                "mean_mahalanobis_ood": float(np.mean(dists_ood)),
                "ood_separation_gap": float("nan"),
                "ood_separation_ratio": float("nan"),
            }

        # AVERAGE OVER SPLITS, because one split is a coin toss at this N.
        #
        # With the permutation in place the estimator is unbiased, but the SPLIT
        # ITSELF carries real variance: five seeds at the shipped
        # `num_samples: 100` give `mahalanobis_sq_per_dim_id` 0.6545 / 0.5219 /
        # 0.4440 / 0.9019 / 0.8116 -- mean 0.6668, sd 0.1715, a range that is
        # 68% of the mean (measured here, 2026-09-05). Reporting whichever
        # number seed 42 happened to produce would put a +-26% arbitrary choice
        # into a cross-model comparison.
        #
        # The forward passes are already done, so the extra splits cost only a
        # covariance fit and a scoring pass each. `n_splits: 1` recovers the
        # single-split behaviour exactly.
        _n_splits = max(1, int(self.config.get("n_splits", 5)))
        # ONE generator, advanced across splits -- not re-seeded.
        #
        # `default_rng(seed)` was created here and its FIRST permutation
        # therefore reproduced `_perm` above exactly, so split 1 was a
        # bit-for-bit duplicate of split 0 and `n_splits: 5` averaged FOUR
        # distinct splits with one counted twice (verified at seeds 42/0/7;
        # 4 distinct fit signatures across 5 calls -- independent Opus review,
        # 2026-09-06). Advancing the same stream that produced `_perm` gives
        # `_n_splits` genuinely distinct permutations.
        _rng_split = _perm_rng
        _diag = {}
        _acc_id, _acc_ood = [], []
        _per_split_auroc = []
        for _k in range(_n_splits):
            if _k == 0:
                _fi, _si = _fit_idx, _score_idx
            else:
                _pk = _rng_split.permutation(n_id)
                _fi, _si = np.sort(_pk[:fit_n]), np.sort(_pk[fit_n:])
            _d_id = _compute_mahalanobis_distances(
                X_id[_fi], X_id[_si], diag=_diag if _k == 0 else None)
            _ood_k = X_ood_full[_si] if _paired_ood else X_ood
            _d_ood = _compute_mahalanobis_distances(X_id[_fi], _ood_k)
            if _d_id and _d_ood:
                _acc_id.append(np.asarray(_d_id, dtype=float))
                _acc_ood.append(np.asarray(_d_ood, dtype=float))
                # PER-SPLIT AUROC, averaged below. AUROC is rank-based and
                # each split has its own shrunk covariance, so distances from
                # different splits are on different scales; ranking them
                # against each other compares numbers that were never
                # commensurable. See the note at the pooled computation.
                try:
                    from sklearn.metrics import roc_auc_score as _ras
                    _per_split_auroc.append(float(_ras(
                        np.concatenate([np.zeros(len(_d_id)),
                                        np.ones(len(_d_ood))]),
                        np.concatenate([_acc_id[-1], _acc_ood[-1]]))))
                except Exception:
                    pass

        if not _acc_id or not _acc_ood:
            return {"error": "Failed to compute Mahalanobis distance (likely covariance singularity)."}
        # Pool the distances across splits: every scored point appears once per
        # split, so the pooled mean is the average over splits and the AUROC is
        # computed on the same pooled scores rather than averaged post hoc.
        dists_id = np.concatenate(_acc_id).tolist()
        dists_ood = np.concatenate(_acc_ood).tolist()

        mean_id = float(np.mean(dists_id))
        mean_ood = float(np.mean(dists_ood))

        # AUROC: can we distinguish ID from OOD using Mahalanobis distance?
        #
        # AVERAGED OVER SPLITS, not computed on the pooled scores.
        #
        # The previous form pooled every split's distances into one ranking and
        # the comment above defended it as better than "averaged post hoc".
        # That was backwards. AUROC is a RANK statistic, and each split fits its
        # own shrunk covariance, so a distance from split 0 and a distance from
        # split 3 are expressed in different units; interleaving them ranks
        # incommensurable numbers. The symptom is unmistakable -- averaging is
        # supposed to REDUCE variance, and over 20 seeds this key was the only
        # one that got noisier, sd 0.024750 -> 0.037685 (**+52.3%**), while
        # every other key improved 50-60% (Opus round 9, 2026-09-07).
        #
        # Per-split AUROC then mean is the standard cross-validated estimator:
        # each AUROC is computed within one covariance, so only comparable
        # distances are ever ranked against each other.
        if _per_split_auroc:
            auroc = float(np.mean(_per_split_auroc))
        else:
            auroc = None
        # Kept for continuity with archived runs, and so the two are
        # comparable on any single-split configuration (where they coincide).
        try:
            from sklearn.metrics import roc_auc_score
            labels = np.concatenate([np.zeros(len(dists_id)), np.ones(len(dists_ood))])
            scores = np.concatenate([dists_id, dists_ood])
            _auroc_pooled = float(roc_auc_score(labels, scores))
        except Exception:
            _auroc_pooled = None

        result = {
            "insufficient_holdout": False,
            "mean_mahalanobis_id": mean_id,
            "mean_mahalanobis_ood": mean_ood,
            "ood_separation_gap": mean_ood - mean_id,
            "ood_separation_ratio": mean_ood / max(mean_id, 1e-10),
            "n_id_fit": int(X_id_fit.shape[0]),
            "n_id_score": int(X_id_score.shape[0]),
            # How many fit/score splits the reported numbers average over. One
            # split at num_samples=100 has a seed-to-seed range 68% of its own
            # mean, so this is not a detail.
            "_meta_n_splits": int(_n_splits),
            "n_ood_score": int(X_ood.shape[0]),
        }

        # DIMENSION CALIBRATION. A raw Mahalanobis distance grows as sqrt(d)
        # under the null, so mean_mahalanobis_* are not comparable across
        # models of different width. dist^2 ~ chi2(d) for a correctly specified
        # Gaussian, so dist^2/d removes the leading d-dependence.
        #
        # It does NOT land at exactly 1. Measured on an iid null sweeping
        # d = 64..1024 at the shipped fit size, the raw distance grows 4.11x
        # while dist^2/d moves only 1.1018 -> 1.1575: the d-scaling is removed,
        # but the level sits at ~1.10-1.16 because the covariance is fitted on
        # n << d with Ledoit-Wolf shrinkage, not estimated exactly. Read it as
        # a width-invariant statistic, not as a calibrated chi-square whose
        # null is 1.0. It also does not make models commensurable on its own:
        # at IDENTICAL d = 1024, gpt2-medium reads 0.0237 against pythia-410m's
        # 2.1219, a 90x gap (adversarial review, 2026-09-03).
        _d = _diag.get("d")
        if _d:
            result["mahalanobis_sq_per_dim_id"] = float(np.mean(np.square(dists_id)) / _d)
            result["mahalanobis_sq_per_dim_ood"] = float(np.mean(np.square(dists_ood)) / _d)
            result["ood_separation_gap_per_dim"] = (
                result["mahalanobis_sq_per_dim_ood"] - result["mahalanobis_sq_per_dim_id"])
            # d/n_fit records how rank-deficient the covariance is: at the
            # shipped defaults the fit is n=25 in d=768, i.e. a rank-24
            # empirical covariance, which is why shrinkage is doing the work.
            _n = _diag.get("n_fit") or 0
            result["_meta_d_over_n_fit"] = float(_d / _n) if _n else float("nan")
            result["_meta_d_model"] = int(_d)
            result["_meta_n_cov_fit"] = int(_n)
        if "shrinkage" in _diag:
            result["_meta_ledoit_wolf_shrinkage"] = _diag["shrinkage"]

        if auroc is not None:
            result["auroc"] = auroc
            result["_meta_auroc_pooled"] = _auroc_pooled
            result["_meta_auroc_split_sd"] = (
                float(np.std(_per_split_auroc)) if len(_per_split_auroc) > 1
                else float("nan"))
            result["_meta_n_auroc_splits"] = int(len(_per_split_auroc))

        return result
