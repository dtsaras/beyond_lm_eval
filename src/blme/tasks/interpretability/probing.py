from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..geometry.utils import collect_hidden_states
import torch
import numpy as np
from tqdm import tqdm
import logging
logger = logging.getLogger("blme")


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
            dataset = [
                {"text": "The quick brown fox jumps over the lazy dog."}
                for _ in range(50)
            ]

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
                    feats = hs[0, :-1, :].cpu().numpy()  # (T-1, D)
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
            idx = np.random.choice(len(all_labels), max_probe_samples, replace=False)
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

            # Use SGD classifier for speed (equivalent to logistic regression)
            try:
                clf = SGDClassifier(
                    loss="log_loss",
                    max_iter=100,
                    random_state=42,
                    n_jobs=-1,
                )
                clf.fit(X_train, train_labels)
                acc = float(clf.score(X_test, test_labels))
            except Exception:
                acc = 0.0

            accuracies.append(acc)

        return {
            "probing_accuracy_per_layer": accuracies,
            "max_probing_accuracy": float(np.max(accuracies)) if accuracies else 0.0,
            "best_layer": int(np.argmax(accuracies)) if accuracies else -1,
            "num_classes": int(len(np.unique(all_labels))),
        }
