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
from .._tokens import (massive_token_share, middle_depth_layers, position0_share,
                       sink_keep_masks, sink_meta, strip_trailing_special)
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


# WikiText-103 splits in the order the extended corpus walks them.
_SPLITS = ("validation", "test", "train")


def _wikitext_passages_extended(n: int) -> list:
    """``n`` stripped WikiText-103 passages beyond the validation split.

    The default corpus (``load_default_corpus``) holds only the validation
    split (about 1,700 passages of >= 64 characters). This walks validation,
    then test, then train, each in its own ``nested_order`` with the same
    filter and stripping, so draws nest and the first passages ARE the default
    corpus. Raises ``ValueError`` if the three splits hold fewer than ``n``.
    """
    from ...cache import (MIN_PASSAGE_CHARS, WIKITEXT_DATASET, _load_split,  # noqa: F401
                          _stripped_texts, nested_order)
    path, name, _ = WIKITEXT_DATASET
    out = []
    for split in _SPLITS:
        if len(out) >= n:
            break
        passages = _stripped_texts(_load_split(path, name, split), "text")
        out.extend({"text": passages[i]} for i in nested_order(len(passages))[:n - len(out)])
    if len(out) < n:
        raise ValueError(f"{n} passages requested, WikiText-103 holds {len(out)}")
    return out


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


def _fit_gaussian(X_train, diag=None):
    """Centroid and precision of ``X_train``: Ledoit-Wolf shrinkage (the
    empirical covariance is rank-deficient at n_fit << d), else a ridge-
    regularised sample covariance. ``diag`` receives the realised shrinkage
    (it differs by an order of magnitude across models), d and n_fit."""
    mu = np.mean(X_train, axis=0)
    if HAS_SKLEARN:
        lw = LedoitWolf()
        lw.fit(X_train)
        inv_cov = lw.precision_
        if diag is not None:
            diag["shrinkage"] = float(getattr(lw, "shrinkage_", float("nan")))
    else:
        cov = np.cov(X_train, rowvar=False)
        cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
        inv_cov = np.linalg.pinv(cov_reg)
    if diag is not None:
        diag["d"] = int(X_train.shape[1])
        diag["n_fit"] = int(X_train.shape[0])
    return mu, inv_cov


def _mahalanobis_scores(X_test, mu, inv_cov):
    return [mahalanobis(x, mu, inv_cov) for x in X_test]


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
        mu, inv_cov = _fit_gaussian(X_train, diag)
        return _mahalanobis_scores(X_test, mu, inv_cov)
    except Exception as e:
        logger.info(f"Error computing Mahalanobis: {e}")
        return []


# Documents whose middle-layer token rows feed the sink-share diagnostics.
_SHARE_DOCS = 200


def _collect_mean_pooled_hidden_states(
    model, tokenizer, dataset, num_samples: int, max_length: int = 128, stats=None,
    policy: str = "exclude_sinks",
) -> torch.Tensor:
    """One final-layer vector per text: the float64 mean over its kept rows.

    Sink policy (non-LLM-origin point cloud): position 0 and every massive
    row -- norm > 10x the median at any middle depth, masks from
    ``sink_keep_masks`` over this collection -- are left out of the mean
    (``mean_excl_sinks``); a text with no kept row falls back to all its rows.
    Tokenizer-appended EOS/SEP are stripped before the forward. ``stats``
    receives the sink-mask info and the share diagnostics.
    """
    samples = list(dataset)[:num_samples]
    if not samples:
        return torch.empty(0)

    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    finals, norms, mid_rows = [], {}, []
    depth = None
    with torch.no_grad():
        for sample in samples:
            # Pre-tokenized ids (the shuffled OOD set) go to the model as ids:
            # decode + re-tokenize does not preserve the token multiset.
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
            inputs, _ = strip_trailing_special(inputs, tokenizer)
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

            # post-final-norm: DELIBERATELY post-norm -- the features the LM
            # head consumes, as Lee et al. fit the classifier's input. The
            # block outputs hidden_states[1:] only set the sink masks.
            h = hidden_states[-1][0].detach().cpu()
            if h.shape[0] == 0:
                continue
            blocks = hidden_states[1:]
            if depth is None:
                depth = middle_depth_layers({k: None for k in range(len(blocks))})
            for k in depth:
                norms.setdefault(k, []).append(torch.linalg.vector_norm(
                    blocks[k][0].detach().to("cpu", torch.float64), dim=1).reshape(-1, 1))
            if len(mid_rows) < _SHARE_DOCS:
                mid_rows.append(blocks[len(blocks) // 2][0].detach().cpu().float())
            finals.append(h)

    if not finals:
        return torch.empty(0)
    keep, info = sink_keep_masks([norms[k] for k in depth])
    if policy != "exclude_sinks":           # every row ("all")
        keep = [torch.ones_like(m) for m in keep]
    reps, n_fallback = [], 0
    for h, m in zip(finals, keep):
        rows = h[m] if bool(m.any()) else h
        n_fallback += int(not bool(m.any()))
        # float64 mean, kept in float64.
        reps.append(rows.double().mean(dim=0))
    if stats is not None:
        stats.update(sink_meta(info))
        stats["_meta_sink_n_docs_all_rows_fallback"] = int(n_fallback)
        stats["_meta_position0_share"] = position0_share(mid_rows)
        stats["_meta_massive_token_share"] = massive_token_share(mid_rows)
    return torch.stack(reps, dim=0)


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
    # Mean-pool over the rows kept by the sink masks (position 0 + massive).
    TOKEN_POLICY = "exclude_sinks"
    SOURCES = {
        "paper": "arXiv:1807.03888",
        "code": "pokaxpoka/deep_Mahalanobis_detector@90c2105e78c6f76a2801fc4c1cb1b84f4ff9af63",
        "llm_precedent": ("Ren et al. 2022 arXiv:2209.15558 (one Gaussian on mean-pooled "
                          "final-layer states of a language model)"),
    }
    ADAPTATION_NOTE = (
        "BLME scores each text's mean-pooled final-layer state by its Mahalanobis distance to a "
        "single Gaussian fitted on other texts, the single-Gaussian, mean-pooled form Ren et "
        "al. (2022) use for language models; Lee et al.'s (2018) class-conditional form needs "
        "labels. The out-of-distribution set is BLME's own: each held-out text with its tokens "
        "shuffled in place, so the task measures how far the model moves a text's "
        "representation when word order is destroyed, at matched vocabulary. The covariance is "
        "Ledoit-Wolf, fitted on 1000 texts (2000 drawn from WikiText-103 in a nested order, "
        "half held out), averaged over 5 random splits. Position 0 and massive rows are left "
        "out of every mean (sink policy). Raw distances grow with sqrt(d) when n_fit < d, so "
        "AUROC, the distance ratio and the dist^2/d forms are the cross-model comparable "
        "outputs."
    )
    HEADLINE_KEYS = ("auroc", "ood_separation_ratio", "mean_mahalanobis_id",
                     "mean_mahalanobis_ood", "ood_separation_gap", "mahalanobis_sq_per_dim_id",
                     "mahalanobis_sq_per_dim_ood", "ood_separation_gap_per_dim")
    # Width: approximate (< 5%) only while n_fit << d; as n_fit nears d the
    # isotropic Ledoit-Wolf target moves with the width (pinned by
    # test_per_dim_width_invariance_needs_n_fit_below_d). Rescale is exact. The
    # raw distances and gap are not claimed (they grow as sqrt(d)).
    SIZE_INVARIANT = dict.fromkeys(
        ("auroc", "ood_separation_ratio", "mahalanobis_sq_per_dim_id",
         "mahalanobis_sq_per_dim_ood", "ood_separation_gap_per_dim"),
        (("zero_pad", "embed_isometric", "rescale"),
         "tests/tasks/invariance/test_geometry_mahalanobis.py::test_per_dim_forms_invariant"))

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Latent Mahalanobis OOD Detection...")
        # Q3: the fit should not be 25 points in 768-5120 dimensions; the
        # default 2000 ID texts give a Ledoit-Wolf fit on 1000 and 1000 held out.
        num_samples = int(self.config.get("num_samples", 2000))
        id_corpus = self.config.get("id_corpus", "wikitext")
        if id_corpus == "wikitext":
            # The default corpus stops at the validation split, below what
            # this fit needs: an extended nested corpus (validation, test,
            # train) is loaded here (test_mahalanobis_protocol.py).
            try:
                dataset = _wikitext_passages_extended(num_samples)
            except (ValueError, RuntimeError) as e:
                return {"error": f"geometry_mahalanobis: WikiText passages unavailable: {e}"}
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

        rng = np.random.RandomState(42 + int(self.config.get("seed_offset", 0)))

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
            # Shuffle mode scores the ID texts against their own shuffles, so it
            # needs num_samples texts; the random strategy also uses the next
            # num_samples as its comparison set.
            if ood_strategy == "random":
                samples = list(dataset)[:num_samples*2]
                mid = max(1, len(samples) // 2)
            else:
                samples = list(dataset)[:num_samples]
                mid = len(samples)
            dataset_id = samples[:mid]
            if ood_strategy == "random":
                import string
                dataset_ood = [{"text": " ".join("".join(rng.choice(list(string.ascii_lowercase), size=rng.randint(3, 8))) for _ in range(10))} for _ in samples[mid:]]
            else:
                # PAIRED: each ID text is scored against a shuffle of itself, so
                # the OOD score reflects the shuffle, not which texts were drawn.
                dataset_ood = []
                for s in dataset_id:
                    text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                    dataset_ood.append(_shuffle_tokens(text, tokenizer, rng))


        if len(dataset_id) < 5 or len(dataset_ood) < 5:
            return {"error": "Need at least 5 ID and 5 OOD samples to compute stable covariance."}

        # One mean-pooled final-layer vector per text (not per token).
        _sink_stats = {}
        X_id = _collect_mean_pooled_hidden_states(
            model, tokenizer, dataset_id, num_samples=len(dataset_id), max_length=max_length,
            stats=_sink_stats, policy=self.TOKEN_POLICY,
        )
        # float64 before the fit: in float32 the scores reorder at a +1e6
        # offset, and every reported number depends on their order. Not
        # bit-pinned to a release, so promotion loses no parity.
        X_id = X_id.double().numpy()

        X_ood = _collect_mean_pooled_hidden_states(
            model, tokenizer, dataset_ood, num_samples=len(dataset_ood), max_length=max_length,
            policy=self.TOKEN_POLICY,
        )
        X_ood = X_ood.double().numpy()   # see the note above

        if X_id.shape[0] < 5 or X_ood.shape[0] < 5:
            return {"error": "Need at least 5 ID and 5 OOD sample representations."}

        # Held-out scoring (Lee et al. fit and score on distinct splits), with a
        # seeded random permutation: a contiguous split is a topic shift on
        # consecutive passages, a stride pairs each text with its neighbours.
        n_id = X_id.shape[0]
        fit_n = max(2, min(n_id - 2, max(5, n_id // 2)))
        _perm_rng = np.random.default_rng(int(self.config.get("seed", 42)))
        _perm = _perm_rng.permutation(n_id)
        _fit_idx = np.sort(_perm[:fit_n])
        _score_idx = np.sort(_perm[fit_n:])
        X_id_fit = X_id[_fit_idx]
        X_id_score = X_id[_score_idx]
        # Pairing: row i of X_ood is the shuffle of row i of X_id.
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
                    "_meta_insufficient_holdout": True,
                    "_meta_n_id_fit": int(X_id_fit.shape[0]),
                    "_meta_n_id_score": int(X_id_score.shape[0]),
                    "_meta_n_ood_score": int(X_ood.shape[0]),
                }
            return {
                "_meta_insufficient_holdout": True,
                "_meta_n_id_fit": int(X_id_fit.shape[0]),
                "_meta_n_id_score": int(X_id_score.shape[0]),
                "_meta_n_ood_score": int(X_ood.shape[0]),
                "mean_mahalanobis_ood": float(np.mean(dists_ood)),
                "ood_separation_gap": float("nan"),
                "ood_separation_ratio": float("nan"),
            }

        # Average over n_splits splits (one split has a seed-to-seed range of
        # ~68% of its mean at num_samples=100); n_splits: 1 is a single split.
        _n_splits = max(1, int(self.config.get("n_splits", 5)))
        # One generator, advanced across splits (re-seeding duplicated split 0).
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
            # ONE Ledoit-Wolf fit per split scores both arms (the fit is
            # deterministic, so this is what two separate fits returned).
            _ood_k = X_ood_full[_si] if _paired_ood else X_ood
            try:
                _mu, _P = _fit_gaussian(X_id[_fi], diag=_diag if _k == 0 else None)
                _d_id = _mahalanobis_scores(X_id[_si], _mu, _P)
                _d_ood = _mahalanobis_scores(_ood_k, _mu, _P)
            except Exception as e:
                logger.info(f"Error computing Mahalanobis: {e}")
                _d_id, _d_ood = [], []
            if _d_id and _d_ood:
                _acc_id.append(np.asarray(_d_id, dtype=float))
                _acc_ood.append(np.asarray(_d_ood, dtype=float))
                # Per-split AUROC: each split has its own covariance, so only
                # distances from one fit are ranked against each other.
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
        # Distances pooled across splits for the mean distances.
        dists_id = np.concatenate(_acc_id).tolist()
        dists_ood = np.concatenate(_acc_ood).tolist()

        mean_id = float(np.mean(dists_id))
        mean_ood = float(np.mean(dists_ood))

        # AUROC = mean of the per-split AUROCs (cross-validated estimator).
        if _per_split_auroc:
            auroc = float(np.mean(_per_split_auroc))
        else:
            auroc = None
        # The pooled-ranking AUROC, _meta_ for continuity.
        try:
            from sklearn.metrics import roc_auc_score
            labels = np.concatenate([np.zeros(len(dists_id)), np.ones(len(dists_ood))])
            scores = np.concatenate([dists_id, dists_ood])
            _auroc_pooled = float(roc_auc_score(labels, scores))
        except Exception:
            _auroc_pooled = None

        result = {
            "_meta_insufficient_holdout": False,
            "mean_mahalanobis_id": mean_id,
            "mean_mahalanobis_ood": mean_ood,
            "ood_separation_gap": mean_ood - mean_id,
            "ood_separation_ratio": mean_ood / max(mean_id, 1e-10),
            "_meta_n_id_fit": int(X_id_fit.shape[0]),
            "_meta_n_id_score": int(X_id_score.shape[0]),
            "_meta_n_splits": int(_n_splits),
            "_meta_n_ood_score": int(X_ood.shape[0]),
        }

        # dist^2/d: dist^2 ~ chi2(d) under a correctly specified Gaussian, so
        # the per-dimension form removes the sqrt(d) growth of the raw distance
        # (approximately, under Ledoit-Wolf at n_fit << d).
        _d = _diag.get("d")
        if _d:
            result["mahalanobis_sq_per_dim_id"] = float(np.mean(np.square(dists_id)) / _d)
            result["mahalanobis_sq_per_dim_ood"] = float(np.mean(np.square(dists_ood)) / _d)
            result["ood_separation_gap_per_dim"] = (
                result["mahalanobis_sq_per_dim_ood"] - result["mahalanobis_sq_per_dim_id"])
            # d/n_fit records how rank-deficient the covariance is.
            _n = _diag.get("n_fit") or 0
            result["_meta_d_over_n_fit"] = float(_d / _n) if _n else float("nan")
            result["_meta_d_model"] = int(_d)
            result["_meta_n_cov_fit"] = int(_n)
            result["_meta_id_corpus"] = str(id_corpus)
        if "shrinkage" in _diag:
            result["_meta_ledoit_wolf_shrinkage"] = _diag["shrinkage"]
        # Sink policy of the ID collection (position 0 + massive rows out of
        # every mean; test_mahalanobis_protocol.py).
        result.update(_sink_stats)

        if auroc is not None:
            result["auroc"] = auroc
            result["_meta_auroc_pooled"] = _auroc_pooled
            result["_meta_auroc_split_sd"] = (
                float(np.std(_per_split_auroc)) if len(_per_split_auroc) > 1
                else float("nan"))
            result["_meta_n_auroc_splits"] = int(len(_per_split_auroc))

        return result
