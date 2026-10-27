"""``repe_concept_separability``: RepE LAT concept reading (Zou et al. 2023).

The headline follows the released concept-reading protocol
(andyzoujm/representation-engineering @5455d8a): LAT task template,
``rep_token=-1`` on the raw hidden-state tuple, ``hidden_layers = -1..-(L-1)``,
PCA(1) of the paired differences as the reading vector with its sign from the
training labels, pairwise choice accuracy, and the layer chosen on a
validation split with the test accuracy reported there. The earlier
logistic-regression AUC on mean-pooled sentences is kept as ``_meta_``.
"""
import random
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


# LAT task template for the truthfulness concept (Zou et al. 2023 Sec 3.1.1:
# "Consider the amount of <concept> in the following: ... The amount of
# <concept> is"); the stimulus is the statement.
_LAT_TEMPLATE = ("Consider the amount of truthfulness in the following statement:\n"
                 "{s}\nThe amount of truthfulness in the statement is")


def _rep_token(policy):
    """The token index a ``paper_protocol:... rep_token=K`` policy reads (-1 default)."""
    m = re.search(r"rep_token=(-?\d+)", policy or "")
    return int(m.group(1)) if m else -1


def _pairs(samples, labels):
    """(positive, negative) sample pairs in order, with one group id per pair.

    The bundled set is laid out as all positives then all negatives with
    matching order; caller rows are paired the same way. A pair takes its
    positive's ``group`` when every row has one, else its own index.
    """
    pos = [s for s, y in zip(samples, labels) if y == 1]
    neg = [s for s, y in zip(samples, labels) if y == 0]
    n = min(len(pos), len(neg))
    pairs = list(zip(pos[:n], neg[:n]))
    if pairs and all(isinstance(a, dict) and a.get("group") is not None for a, _ in pairs):
        gid = [a["group"] for a, _ in pairs]
    else:
        gid = list(range(n))
    return pairs, gid


def _reading_states(model, tokenizer, texts, num_layers, device, rep_token):
    """``{h: (n, d) float32}`` at ``rep_token`` for h in -1..-(L-1).

    post-final-norm=False, DELIBERATELY: ``rep_reading_pipeline.py:23-27``
    reads ``outputs['hidden_states'][layer][:, rep_token]`` from the raw tuple
    (hs[-1] is post-ln_f) and casts bf16 to float32.
    """
    hidden_layers = list(range(-1, -num_layers, -1))
    states = {h: [] for h in hidden_layers}
    last_ids = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            last_ids.append(int(enc["input_ids"][0, rep_token]))
            hs = model(**enc, output_hidden_states=True).hidden_states
            for h in hidden_layers:
                states[h].append(hs[h][0, rep_token].detach().float().cpu().numpy())
    return {h: np.stack(v).astype(np.float32) for h, v in states.items()}, hidden_layers, last_ids


def _project(H, mean, direction):
    """``rep_readers.recenter`` + ``project_onto_direction`` (float32 torch)."""
    x = torch.Tensor(H) - torch.Tensor(mean)
    d = torch.Tensor(direction)
    return x.matmul(d) / torch.norm(d)


def _pca_reader(H_train, train_labels):
    """``PCARepReader`` on one layer: PCA(1) of the recentred differences of
    consecutive (shuffled) pairs, then the sign from the training labels
    (``rep_readers.py:137-184``, ``rep_reading_pipeline.py:141-156``)."""
    from sklearn.decomposition import PCA
    rel = H_train[::2] - H_train[1::2]
    mean = rel.mean(axis=0, keepdims=True)
    centred = (torch.Tensor(rel) - torch.Tensor(mean)).numpy()
    pca = PCA(n_components=1, whiten=False).fit(np.vstack(centred))
    direction = pca.components_[0].astype(np.float32)
    proj = _project(H_train, mean, direction).numpy()
    outs = [proj[i:i + 2] for i in range(0, len(proj), 2)]
    p_min = np.mean([o[lab.index(1)] == min(o) for o, lab in zip(outs, train_labels)])
    p_max = np.mean([o[lab.index(1)] == max(o) for o, lab in zip(outs, train_labels)])
    sign = np.sign(np.mean(p_max) - np.mean(p_min))
    return direction, mean, (1.0 if sign == 0 else float(sign)), float(pca.explained_variance_ratio_[0])


def _choice_accuracy(H_true, H_false, mean, direction, sign):
    """``rep_reading_eval.py:67-75``: a pair is correct when the true item has
    the larger (sign +1) / smaller (sign -1) projection."""
    pt = _project(H_true, mean, direction).numpy()
    pf = _project(H_false, mean, direction).numpy()
    pick = np.argmax if sign != -1 else np.argmin
    return float(np.mean([pick([a, b]) == 0 for a, b in zip(pt, pf)]))


def _lat_reading(states, hidden_layers, n_pairs, gid, seed):
    """Val-selected-layer test accuracy, averaged over the 3 rotations of three
    group folds (each fold is the test split once; train and val are the other
    two), so every bundled pair is tested once and entity-swap partners never
    straddle a split."""
    from sklearn.model_selection import GroupKFold
    folds = [te for _, te in GroupKFold(n_splits=3).split(
        np.zeros(n_pairs), np.zeros(n_pairs), np.asarray(gid))]
    per_rot = []
    for r in range(3):
        te, tr, va = folds[r], folds[(r + 1) % 3], folds[(r + 2) % 3]
        rng = random.Random(seed)
        order = []
        for _ in tr:
            o = [0, 1]           # 0 = the true statement
            rng.shuffle(o)
            order.append(o)
        train_labels = [[int(j == 0) for j in o] for o in order]
        val, test, evr = [], [], []
        for h in hidden_layers:
            T, Fz = states[h]
            H_train = np.concatenate([(T if j == 0 else Fz)[[i]]
                                      for i, o in zip(tr, order) for j in o], axis=0)
            direction, mean, sign, ev = _pca_reader(H_train, train_labels)
            val.append(_choice_accuracy(T[va], Fz[va], mean, direction, sign))
            test.append(_choice_accuracy(T[te], Fz[te], mean, direction, sign))
            evr.append(ev)
        best = val.index(max(val))      # first maximum in -1, -2, ... order
        per_rot.append({"layer": hidden_layers[best], "val": val[best], "test": test[best],
                        "val_curve": val, "test_curve": test, "evr": evr})
    return per_rot


@register_task("repe_concept_separability")
class ConceptSeparabilityTask(DiagnosticTask):
    """RepE LAT concept reading of truthfulness (Zou et al. 2023, arXiv:2310.01405).

    Each true/false statement is wrapped in the LAT template and read at the
    last token at hidden_states -1..-(L-1). Per layer, the reading vector is
    PCA(1) of the training pairs' differences; a held-out pair is correct when
    the true statement projects further along it. The layer is chosen on a
    validation split and ``lat_test_acc`` is the test accuracy there.
    """

    TOKEN_POLICY = "paper_protocol:Zou2023 rep_token=-1"

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Concept Separability Analysis (RepE)...")

        if not _common.HAS_SKLEARN:
            return {"error": "scikit-learn is required for concept separability. Install via: pip install scikit-learn"}

        num_samples = self.config.get("num_samples", len(_FACT_PAIRS))

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
        pairs, gid = _pairs(samples, labels)
        if len(set(gid)) < 3:
            return {"error": "the LAT protocol needs at least 3 independent pair "
                             f"groups for train/val/test (got {len(set(gid))})"}

        device = next(model.parameters()).device
        layers = get_layers(model)
        if layers is None:
            return {"error": "Could not detect model layers"}
        num_layers = len(layers)

        out = self._lat_headline(model, tokenizer, pairs, gid, num_layers, device)
        out.update(self._legacy_lr_auc(model, tokenizer, samples, texts, labels,
                                       used_fallback, num_layers, device))
        return out

    def _lat_headline(self, model, tokenizer, pairs, gid, num_layers, device):
        rep_token = _rep_token(self.TOKEN_POLICY)
        use_template = bool(self.config.get("lat_template", True))
        fmt = (lambda t: _LAT_TEMPLATE.format(s=t)) if use_template else (lambda t: t)
        texts = [fmt(a["text"]) for a, _ in pairs] + [fmt(b["text"]) for _, b in pairs]
        raw, hidden_layers, last_ids = _reading_states(
            model, tokenizer, texts, num_layers, device, rep_token)
        n = len(pairs)
        states = {h: (raw[h][:n], raw[h][n:]) for h in hidden_layers}
        rot = _lat_reading(states, hidden_layers, n, gid, int(self.config.get("seed", 0)))

        # hidden_states index of -k is L+1-k: depth fraction in [2/L, 1].
        frac = [(num_layers + 1 + r["layer"]) / num_layers for r in rot]
        asc = sorted(range(len(hidden_layers)), key=lambda i: hidden_layers[i])
        special = set(getattr(tokenizer, "all_special_ids", None) or [])
        return {
            "lat_test_acc": float(np.mean([r["test"] for r in rot])),
            "lat_layer_frac": float(np.mean(frac)),
            "_meta_lat_val_acc": float(np.mean([r["val"] for r in rot])),
            "_meta_lat_selected_layers": [int(num_layers + 1 + r["layer"]) for r in rot],
            "_meta_layer_lat_test_acc": [float(np.mean([r["test_curve"][i] for r in rot])) for i in asc],
            "_meta_layer_lat_val_acc": [float(np.mean([r["val_curve"][i] for r in rot])) for i in asc],
            "_meta_layer_pca_explained_ratio": [float(np.mean([r["evr"][i] for r in rot])) for i in asc],
            "_meta_lat_hidden_state_indices": [int(num_layers + 1 + hidden_layers[i]) for i in asc],
            "_meta_lat_n_pairs": int(n),
            "_meta_lat_n_groups": int(len(set(gid))),
            "_meta_lat_rotations": len(rot),
            "_meta_rep_token": int(rep_token),
            "_meta_lat_template_used": use_template,
            # The share of readouts that land on a special token (e.g. an
            # appended EOS), which a last-token protocol would read.
            "_meta_rep_token_is_special_frac": float(np.mean([i in special for i in last_ids])),
        }

    def _legacy_lr_auc(self, model, tokenizer, samples, texts, labels, used_fallback,
                       num_layers, device):
        """The earlier estimator, ``_meta_`` only: CV logistic-regression AUC on
        mean-pooled raw sentences. An unstandardised L2 penalty makes it move
        under a function-preserving residual rescale."""
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
        summ = _auc_summary(layer_aucs, layer_accs)
        if "error" in summ:
            return {"_meta_lr_error": summ["error"]}
        return {f"_meta_{k}": v for k, v in summ.items()}

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
