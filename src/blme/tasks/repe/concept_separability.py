"""``repe_concept_separability``: linear separability of a concept per layer."""
import re

import numpy as np
import torch

from ..base import DiagnosticTask
from ...registry import register_task
from ..common import get_layers
from . import _common
from ._common import logger

# Minimal true/false pairs: each pair shares its frame and differs in one
# entity, and every entity appears in both classes across the set, so neither
# surface form nor unigram statistics carry the label.
_FACT_PAIRS = [
    ("The capital of France is Paris.", "The capital of France is Rome."),
    ("The capital of Japan is Tokyo.", "The capital of Japan is Beijing."),
    ("The capital of Italy is Rome.", "The capital of Italy is Paris."),
    ("The capital of China is Beijing.", "The capital of China is Tokyo."),
    ("Water freezes at zero degrees Celsius.",
     "Water freezes at fifty degrees Celsius."),
    ("Water boils at one hundred degrees Celsius.",
     "Water boils at zero degrees Celsius."),
    ("The Earth orbits the Sun.", "The Sun orbits the Earth."),
    ("The Moon orbits the Earth.", "The Earth orbits the Moon."),
    ("A triangle has three sides.", "A triangle has four sides."),
    ("A square has four sides.", "A square has three sides."),
    ("Ice is frozen water.", "Ice is frozen sand."),
    ("Sand is not frozen water.", "Sand is frozen water."),
    ("The capital of Spain is Madrid.", "The capital of Spain is Lisbon."),
    ("The capital of Portugal is Lisbon.", "The capital of Portugal is Madrid."),
    ("The capital of Germany is Berlin.", "The capital of Germany is Vienna."),
    ("The capital of Austria is Vienna.", "The capital of Austria is Berlin."),
    ("The capital of Canada is Ottawa.", "The capital of Canada is Toronto."),
    ("The capital of Egypt is Cairo.", "The capital of Egypt is Athens."),
    ("The capital of Greece is Athens.", "The capital of Greece is Cairo."),
    ("The capital of Russia is Moscow.", "The capital of Russia is Kyiv."),
    ("A week has seven days.", "A week has ten days."),
    ("A year has twelve months.", "A year has seven months."),
    ("An hour has sixty minutes.", "An hour has twelve minutes."),
    ("A minute has sixty seconds.", "A minute has ten seconds."),
    ("A pentagon has five sides.", "A pentagon has six sides."),
    ("A hexagon has six sides.", "A hexagon has five sides."),
    ("Birds have feathers.", "Birds have scales."),
    ("Fish have scales.", "Fish have feathers."),
    ("Cows eat grass.", "Cows eat metal."),
    ("Fire is hot.", "Fire is cold."),
    ("Snow is cold.", "Snow is hot."),
    ("The Pacific is an ocean.", "The Pacific is a desert."),
    ("The Sahara is a desert.", "The Sahara is an ocean."),
    ("Oxygen is a gas.", "Oxygen is a metal."),
    ("Iron is a metal.", "Iron is a gas."),
    ("Two plus two equals four.", "Two plus two equals five."),
    ("Three plus three equals six.", "Three plus three equals four."),
    ("Ten is greater than five.", "Ten is less than five."),
    ("Five is less than ten.", "Five is greater than ten."),
]

# Words that carry no entity identity; without them "the"/"is" would union
# every pair into one group.
_STOP = {"the", "a", "an", "is", "are", "was", "were", "of", "at",
         "in", "on", "to", "and", "not", "has", "have", "than",
         "equals", "plus"}


def _vary(a, b):
    """Content words that differ between a pair; for a pure word permutation
    ("Earth orbits Sun" / "Sun orbits Earth") the shared content words."""
    ta = set(re.findall(r"[a-z]+", a.lower()))
    tb = set(re.findall(r"[a-z]+", b.lower()))
    sym = (ta ^ tb) - _STOP
    if sym:
        return sym
    return (ta | tb) - _STOP


def _fact_pair_groups(pairs):
    """Union pairs that share a varying word, so entity-swap partners share a
    CV fold (otherwise an entity-keyed probe anti-generalises: held-out AUC
    below chance). 39 pairs -> 19 groups."""
    var = [_vary(a, b) for a, b in pairs]
    parent = list(range(len(pairs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            if var[i] & var[j]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    return [find(i) for i in range(len(pairs))]


def _fallback_dataset(num_samples):
    """The bundled fact pairs as labelled rows: all positives, then all negatives."""
    gid = _fact_pair_groups(_FACT_PAIRS)
    pos = [t for t, _f in _FACT_PAIRS]
    neg = [f for _t, f in _FACT_PAIRS]
    reps = max(1, (num_samples + len(pos) - 1) // len(pos))
    pos_x = (list(zip(pos, gid)) * reps)[:num_samples]
    neg_x = (list(zip(neg, gid)) * reps)[:num_samples]
    return ([{"text": t, "label": 1, "group": g} for t, g in pos_x]
            + [{"text": t, "label": 0, "group": g} for t, g in neg_x])


def _is_labelled(items):
    """True only if EVERY item is a labelled dict; a mixed list falls back."""
    if not items:
        return False
    return all(isinstance(it, dict) and "label" in it and "text" in it
               for it in items)


def _auc_summary(layer_aucs, layer_accs):
    """Summarise per-layer separability, NaN-aware.

    Non-finite low-precision layers give NaN AUCs; the nan-aware reductions keep
    them from winning the argmax or poisoning the mean. All-NaN is an error.
    """
    if not layer_aucs or all(np.isnan(a) for a in layer_aucs):
        return {"error": "No layer produced finite separability values"}
    return {
        "layer_separability_auc": layer_aucs,
        "layer_separability_acc": layer_accs,
        "max_auc_layer": int(np.nanargmax(layer_aucs)),
        "max_auc": float(np.nanmax(layer_aucs)),
        "mean_auc": float(np.nanmean(layer_aucs)),
    }


@register_task("repe_concept_separability")
class ConceptSeparabilityTask(DiagnosticTask):
    """Held-out linear separability (AUC/accuracy) of a concept at each layer.

    Following Zou et al. (2023), tests whether a concept is linearly readable
    from the representations. Estimator: stratified/grouped-CV logistic
    regression on the mean-pooled sentence state of every block (hs[1..L] of
    the raw tuple, as the released reader indexes it).
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Concept Separability Analysis (RepE)...")

        if not _common.HAS_SKLEARN:
            return {"error": "scikit-learn is required for concept separability. Install via: pip install scikit-learn"}

        num_samples = self.config.get("num_samples", 20)

        # Only a dataset whose every row carries a real label is used; the
        # caller's corpus is never mutated.
        materialised = None if dataset is None else list(dataset)
        used_fallback = False
        if materialised is None or not _is_labelled(materialised):
            used_fallback = True
            dataset = _fallback_dataset(num_samples)
        if not used_fallback and materialised is not None:
            dataset = materialised
        samples = list(dataset)[:num_samples * 2]
        texts = [s["text"] for s in samples]

        labels, err = self._binary_labels(samples)
        if err is not None:
            return err
        if len(set(labels)) < 2: return {"error": "Need at least two classes."}
        if len(texts) < 4: return {"error": "Need at least 4 samples for CV."}
        # Reject labels that cannot support stratified CV before any forward pass.
        _counts = np.bincount(np.array(labels))
        if _counts.min() < 2:
            return {"error": "concept labels too imbalanced for stratified CV "
                             f"(class counts: {_counts.tolist()})"}

        device = next(model.parameters()).device
        layers = get_layers(model)
        if layers is None:
            return {"error": "Could not detect model layers"}
        num_layers = len(layers)

        layer_reps = self._mean_pooled_states(model, tokenizer, texts, num_layers, device)
        y = np.array(labels)
        n_splits = min(3, np.min(np.bincount(y)))
        if n_splits < 2: n_splits = 2
        split = self._splitter(samples, y, used_fallback, n_splits)

        layer_aucs, layer_accs = [], []
        for l_idx in range(num_layers):
            auc, acc = self._layer_cv(np.array(layer_reps[l_idx]), y, n_splits, split)
            layer_aucs.append(auc)
            layer_accs.append(acc)
        return _auc_summary(layer_aucs, layer_accs)

    @staticmethod
    def _binary_labels(samples):
        """Exact-integer 0/1 labels, or this task's error dict."""
        try:
            raw = [s["label"] for s in samples]
            labels = [int(v) for v in raw]
        except (TypeError, ValueError):
            return None, {"error": "Concept labels must be binary integers (0/1); "
                                   "got non-numeric labels."}
        # int() truncates (0.9 -> 0), so require exact integrality.
        if any(float(v) != float(i) for v, i in zip(raw, labels)):
            return None, {"error": "Concept labels must be binary INTEGERS (0/1); "
                                   f"got non-integral values {sorted(set(raw))}."}
        if set(labels) - {0, 1}:
            return None, {"error": "Concept labels must be binary (0/1); "
                                   f"got {sorted(set(labels))}."}
        return labels, None

    @staticmethod
    def _mean_pooled_states(model, tokenizer, texts, num_layers, device):
        """Per-layer float64 token-mean of each text at hs[1..L].

        post-final-norm=False, DELIBERATELY: the released reader reads the raw
        tuple (rep_reading_pipeline.py:23-24), whose deepest entry is post-ln_f
        (tests/tasks/test_comprehensive_parity.py encodes the same convention).
        """
        layer_reps = {l: [] for l in range(num_layers)}
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                out = model(**inputs, output_hidden_states=True)
                hs = list(out.hidden_states)
                for l_idx in range(num_layers):
                    hidden = hs[l_idx + 1][0]
                    # Mean in float64: a model-dtype accumulator overflows near
                    # the top of the range (tests/test_defect_class_guards.py).
                    rep = hidden.double().mean(dim=0).cpu().numpy()
                    layer_reps[l_idx].append(rep)
        return layer_reps

    @staticmethod
    def _splitter(samples, y, used_fallback, n_splits):
        """``split(X, y, row_mask)`` -> folds; grouped when groups are known.

        Explicit per-row groups win; the bundled set's pair layout (all
        positives then all negatives) is inferred only for the fallback set,
        never for caller data.
        """
        n_half = len(y) // 2
        groups = None
        explicit = [s.get("group") if isinstance(s, dict) else None for s in samples]
        if len(explicit) == len(y) and all(g is not None for g in explicit):
            uniq = {g: i for i, g in enumerate(dict.fromkeys(explicit))}
            groups = np.array([uniq[g] for g in explicit])
        elif (used_fallback and n_half > 1 and int(y[:n_half].sum()) == n_half
                and int(y[n_half:].sum()) == 0 and len(y) == 2 * n_half):
            groups = np.concatenate([np.arange(n_half), np.arange(n_half)])
        if groups is not None and len(np.unique(groups)) >= n_splits:
            from sklearn.model_selection import GroupKFold
            cv = GroupKFold(n_splits=n_splits)

            def split(Xa, ya, row_mask=None):
                # Groups follow the same row mask that dropped non-finite rows.
                g = groups if row_mask is None else groups[row_mask]
                return cv.split(Xa, ya, g)
        else:
            cv = _common.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            def split(Xa, ya, row_mask=None):
                return cv.split(Xa, ya)
        return split

    @staticmethod
    def _layer_cv(X, y, n_splits, split):
        """Mean held-out (AUC, accuracy) of a logistic probe; NaN if the layer
        cannot support the CV (non-finite rows, too few rows per class, or a
        materialised fold missing a class)."""
        nan = (float("nan"), float("nan"))
        mask = np.isfinite(X).all(axis=1)
        if mask.sum() < n_splits * 2:
            return nan
        X = X[mask]
        y_l = y[mask]
        cls_counts = np.bincount(y_l.astype(np.int64), minlength=2)
        if len(np.unique(y_l)) < 2 or cls_counts.min() < n_splits:
            return nan
        try:
            folds = list(split(X, y_l, mask))
        except Exception:
            folds = []
        if not folds or any(len(np.unique(y_l[tr])) < 2
                            or len(np.unique(y_l[te])) < 2
                            for tr, te in folds):
            return nan

        fold_aucs, fold_accs = [], []
        for train_idx, test_idx in folds:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_l[train_idx], y_l[test_idx]
            clf = _common.LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            probas = clf.predict_proba(X_test)[:, 1] if len(set(y_train)) > 1 else preds
            fold_accs.append(_common.accuracy_score(y_test, preds))
            try: fold_aucs.append(_common.roc_auc_score(y_test, probas))
            except ValueError: fold_aucs.append(_common.accuracy_score(y_test, preds))
        return (float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
                float(np.mean(fold_accs)) if fold_accs else float("nan"))
