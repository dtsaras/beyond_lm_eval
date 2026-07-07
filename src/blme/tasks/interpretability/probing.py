from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..geometry.utils import collect_hidden_states
import torch
import numpy as np
from tqdm import tqdm
import logging
logger = logging.getLogger("blme")


def _probe_accuracy(X_train, y_train, X_test, y_test) -> float:
    """Linear-probe test accuracy for one layer.

    A degenerate single-class training split returns the majority-class rate
    (a constant probe's linear decodability) rather than a fabricated 0.0, and
    a genuine fit failure returns NaN with a warning rather than a silent 0.0
    that masks the failure.
    """
    from collections import Counter

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    if y_test.size == 0:
        return float("nan")
    if np.unique(y_train).size < 2:
        maj = Counter(y_test.tolist()).most_common(1)[0][1]
        return float(maj / y_test.size)
    try:
        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(loss="log_loss", max_iter=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        return float(clf.score(X_test, y_test))
    except Exception as e:  # narrow, logged — never a silent fabricated 0.0
        logger.warning(f"linear probe failed: {type(e).__name__}: {e}")
        return float("nan")


@register_task("interpretability_probing")
class LinearProbingTask(DiagnosticTask):
    """
    Trains a linear probe on frozen hidden states to measure what
    information is linearly decodable at each layer.
    Default probe: predict the token identity from its hidden state.
    Ref: Alain & Bengio, "Understanding Intermediate Layers Using Linear
         Classifier Probes", ICLR 2017 Workshop. arXiv:1610.01644

    Caveat: High probe accuracy doesn't confirm that the model mechanistically
    uses the probed feature (Hewitt & Liang, 2019). A control task or
    selectivity metric is needed to distinguish true encoding from
    high-dimensional noise.
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Linear Probing Analysis...")

        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(50)

        num_samples = self.config.get("num_samples", 50)
        max_tokens = self.config.get("max_tokens", 128)

        # Collect representations and labels from all layers
        all_labels = []
        layer_features = {}

        with torch.no_grad():
            for i, sample in enumerate(
                tqdm(dataset[:num_samples], desc="Collecting Probing Data")
            ):
                text = sample.get("text", "") if isinstance(sample, dict) else sample
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_tokens,
                    truncation=True,
                ).to(model.device)

                outputs = model(
                    **inputs, output_hidden_states=True
                )
                hidden_states = outputs.hidden_states  # tuple of (B, T, D)

                # Labels: next token prediction (shift by 1)
                input_ids = inputs.input_ids[0]  # (T,)
                if len(input_ids) < 2:
                    continue

                # Use tokens [0..T-2] as inputs, [1..T-1] as labels
                labels = input_ids[1:].cpu().numpy()
                all_labels.append(labels)

                for layer_idx, hs in enumerate(hidden_states):
                    # .float() first: numpy doesn't accept bf16.
                    feats = hs[0, :-1, :].float().cpu().numpy()  # (T-1, D)
                    if layer_idx not in layer_features:
                        layer_features[layer_idx] = []
                    layer_features[layer_idx].append(feats)

        if not all_labels:
            return {"error": "No data collected"}

        # Concatenate
        all_labels = np.concatenate(all_labels, axis=0)

        # Subsample if too large (probing is expensive)
        max_probe_samples = self.config.get("max_probe_samples", 5000)
        if len(all_labels) > max_probe_samples:
            # Seeded subsample so the default-config output is reproducible.
            _rng = np.random.default_rng(int(self.config.get("seed", 42)))
            idx = _rng.choice(len(all_labels), max_probe_samples, replace=False)
        else:
            idx = np.arange(len(all_labels))

        labels = all_labels[idx]

        # Train/test split (80/20)
        n = len(labels)
        n_train = int(0.8 * n)
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        train_labels = all_labels[train_idx]
        test_labels = all_labels[test_idx]

        # Evaluate each layer
        try:
            from sklearn.linear_model import SGDClassifier
        except ImportError:
            return {"error": "scikit-learn is required for probing. Install via: pip install scikit-learn"}

        layers = sorted(layer_features.keys())
        accuracies = []

        for layer_idx in tqdm(layers, desc="Probing Layers"):
            X = np.concatenate(layer_features[layer_idx], axis=0)
            X_train = X[train_idx]
            X_test = X[test_idx]
            accuracies.append(_probe_accuracy(X_train, train_labels, X_test, test_labels))

        acc_arr = np.asarray(accuracies, dtype=float)
        has_finite = bool(np.isfinite(acc_arr).any())
        return {
            "probing_accuracy_per_layer": accuracies,
            "max_probing_accuracy": float(np.nanmax(acc_arr)) if has_finite else 0.0,
            "best_layer": int(np.nanargmax(acc_arr)) if has_finite else -1,
            "num_classes": int(len(np.unique(all_labels))),
        }
