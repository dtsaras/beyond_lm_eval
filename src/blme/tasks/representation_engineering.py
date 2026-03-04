import torch
import torch.nn.functional as F
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .base import DiagnosticTask
from ..registry import register_task
from .common import get_layers
import logging
logger = logging.getLogger("blme")

@register_task("repe_task_vectors")
class TaskVectorGeometryTask(DiagnosticTask):
    """
    Implements a Representation Engineering (RepE) task vector extraction.
    Takes paired contrastive datasets (e.g. true vs false statements) and 
    extracts the 'Reading Vector' / 'Task Vector' by taking the mean difference
    of the activations at the last token. Measures the geometry (norm, distinctness)
    of the resulting vector.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Task Vector Geometry (RepE)...")
        num_samples = self.config.get("num_samples", 5)
        
        device = next(model.parameters()).device
        layers = get_layers(model)
        num_layers = len(layers)
        
        # We need paired contrasting positive/negative examples
        if dataset is None:
            dataset = [
                {"text_pos": "The earth revolves around the sun.", "text_neg": "The sun revolves around the earth."},
                {"text_pos": "Water boils at 100 degrees Celsius.", "text_neg": "Water boils at 0 degrees Celsius."},
                {"text_pos": "A triangle has three sides.", "text_neg": "A triangle has four sides."},
            ] * num_samples
        
        samples = list(dataset)[:num_samples]
        if len(samples) < 1:
             return {"error": "Need at least 1 sample with 'text_pos' and 'text_neg' keys"}
             
        if not all("text_pos" in s and "text_neg" in s for s in samples):
             # If dataset doesn't have pairs, we can't do contrastive task vectors
             return {"error": "Dataset must contain 'text_pos' and 'text_neg' paired keys for contrastive Task Vector extraction"}

        
        # Dictionaries to hold the activations for each sample pair across layers
        pos_activations = {l: [] for l in range(num_layers)}
        neg_activations = {l: [] for l in range(num_layers)}

        with torch.no_grad():
            for s in samples:
                # 1. Forward pass on positive example
                ids_pos = tokenizer.encode(s["text_pos"], return_tensors="pt", truncation=True, max_length=128).to(device)
                out_pos = model(ids_pos, output_hidden_states=True)
                
                # 2. Forward pass on negative example
                ids_neg = tokenizer.encode(s["text_neg"], return_tensors="pt", truncation=True, max_length=128).to(device)
                out_neg = model(ids_neg, output_hidden_states=True)
                
                # Collect representations at the last token for all layers
                for l in range(num_layers):
                    # Hidden states include the embedding layer as index 0, so add 1
                    h_pos = out_pos.hidden_states[l + 1][0, -1].cpu().float()
                    h_neg = out_neg.hidden_states[l + 1][0, -1].cpu().float()
                    
                    pos_activations[l].append(h_pos)
                    neg_activations[l].append(h_neg)
                    
        results = {}
        # Now define and analyze the Task Vectors
        # Task Vector = Mean(Pos) - Mean(Neg)
        
        task_vector_norms = []
        task_vector_cosine_similarities = []
        
        for l in range(num_layers):
            A_pos = torch.stack(pos_activations[l]) # Shape: (samples, hidden_dim)
            A_neg = torch.stack(neg_activations[l]) # Shape: (samples, hidden_dim)
            
            mean_pos = A_pos.mean(dim=0)
            mean_neg = A_neg.mean(dim=0)
            
            # The Task Vector / Reading Vector v
            v = mean_pos - mean_neg
            
            # 1. Magnitude of the task vector
            v_norm = torch.norm(v, p=2).item()
            task_vector_norms.append(v_norm)
            
            # 2. Cosine similarity between pos and neg means (Is the distinction clear or murky?)
            cos_sim = F.cosine_similarity(mean_pos.unsqueeze(0), mean_neg.unsqueeze(0)).item()
            task_vector_cosine_similarities.append(cos_sim)
            
        results["layer_task_vector_norms"] = task_vector_norms
        results["layer_task_vector_cosine_sim"] = task_vector_cosine_similarities
        
        if task_vector_norms:
            results["max_norm_layer"] = int(np.argmax(task_vector_norms))
            results["mean_vector_norm"] = float(np.mean(task_vector_norms))
            
        return results


@register_task("repe_concept_separability")
class ConceptSeparabilityTask(DiagnosticTask):
    """
    Computes Linear Separability (AUC/Accuracy) of a target concept at each layer.
    
    Following Zou et al. (2023), tests if activating concepts can be linearly
    separated (A prerequisite for Representation Engineering).
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Concept Separability Analysis (RepE)...")

        if not HAS_SKLEARN:
            return {"error": "scikit-learn is required for concept separability. Install via: pip install scikit-learn"}

        num_samples = self.config.get("num_samples", 20)

        if dataset is None:
            dataset = [{"text": f"This is clearly a wonderful and true statement number {i}.", "label": 1} for i in range(num_samples)] + \
                      [{"text": f"This is an absolutely terrible and false lie number {i}.", "label": 0} for i in range(num_samples)]
        else:
            if len(dataset) > 0 and "label" not in dataset[0]:
                for i, d in enumerate(dataset):
                    if isinstance(d, str): dataset[i] = {"text": d, "label": i % 2}
                    else: d["label"] = i % 2
        
        samples = list(dataset)[:num_samples*2]
        texts = [s["text"] for s in samples]
        labels = [s["label"] for s in samples]
        
        if len(set(labels)) < 2: return {"error": "Need at least two classes."}
        if len(texts) < 4: return {"error": "Need at least 4 samples for CV."}
        
        device = next(model.parameters()).device
        layers = get_layers(model)
        num_layers = len(layers)
        
        layer_reps = {l: [] for l in range(num_layers)}
        
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                out = model(**inputs, output_hidden_states=True)
                for l_idx in range(num_layers):
                    hidden = out.hidden_states[l_idx + 1][0]
                    rep = hidden.mean(dim=0).cpu().numpy()
                    layer_reps[l_idx].append(rep)
        
        y = np.array(labels)
        layer_aucs, layer_accs = [], []
        
        n_splits = min(3, np.min(np.bincount(y)))
        if n_splits < 2: n_splits = 2
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for l_idx in range(num_layers):
            X = np.array(layer_reps[l_idx])
            fold_aucs, fold_accs = [], []
            
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                clf = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)
                clf.fit(X_train, y_train)
                
                preds = clf.predict(X_test)
                probas = clf.predict_proba(X_test)[:, 1] if len(set(y_train)) > 1 else preds
                
                fold_accs.append(accuracy_score(y_test, preds))
                try: fold_aucs.append(roc_auc_score(y_test, probas))
                except ValueError: fold_aucs.append(accuracy_score(y_test, preds))
                    
            layer_aucs.append(float(np.mean(fold_aucs)))
            layer_accs.append(float(np.mean(fold_accs)))
            
        return {
            "layer_separability_auc": layer_aucs,
            "layer_separability_acc": layer_accs,
            "max_auc_layer": int(np.argmax(layer_aucs)),
            "max_auc": float(np.max(layer_aucs)),
            "mean_auc": float(np.mean(layer_aucs))
        }


@register_task("repe_steering_effectiveness")
class SteeringEffectivenessTask(DiagnosticTask):
    """
    Measures the effectiveness of representation steering by extracting
    task vectors (reusing the contrastive approach) and injecting them
    during forward passes on neutral prompts, measuring output shift
    via KL divergence.

    Returns layer_steering_kl_divergence, best_steering_layer,
    and steering_success_rate.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Steering Vector Effectiveness...")
        num_samples = self.config.get("num_samples", 3)
        steering_alpha = self.config.get("steering_alpha", 1.0)

        device = next(model.parameters()).device
        layers = get_layers(model)
        if layers is None:
            return {"error": "Could not detect model layers."}
        num_layers = len(layers)

        # Contrastive dataset for task vector extraction
        if dataset is None:
            dataset = [
                {"text_pos": "This is absolutely true and correct.",
                 "text_neg": "This is completely false and wrong.",
                 "neutral": "The weather today is"},
            ] * num_samples

        samples = list(dataset)[:num_samples]
        if not samples:
            return {"error": "Need at least 1 sample."}

        required = {"text_pos", "text_neg", "neutral"}
        # If dataset lacks neutral, provide a default
        for s in samples:
            if "neutral" not in s:
                s["neutral"] = "The weather today is"
            if "text_pos" not in s or "text_neg" not in s:
                return {"error": "Dataset must contain 'text_pos' and 'text_neg' keys."}

        # Step 1: Extract task vectors at each layer
        task_vectors = {}
        with torch.no_grad():
            pos_acts = {l: [] for l in range(num_layers)}
            neg_acts = {l: [] for l in range(num_layers)}

            for s in samples:
                ids_pos = tokenizer.encode(s["text_pos"], return_tensors="pt",
                                           truncation=True, max_length=128).to(device)
                out_pos = model(ids_pos, output_hidden_states=True)

                ids_neg = tokenizer.encode(s["text_neg"], return_tensors="pt",
                                           truncation=True, max_length=128).to(device)
                out_neg = model(ids_neg, output_hidden_states=True)

                for l in range(num_layers):
                    pos_acts[l].append(out_pos.hidden_states[l + 1][0, -1].cpu().float())
                    neg_acts[l].append(out_neg.hidden_states[l + 1][0, -1].cpu().float())

            for l in range(num_layers):
                mean_pos = torch.stack(pos_acts[l]).mean(dim=0)
                mean_neg = torch.stack(neg_acts[l]).mean(dim=0)
                task_vectors[l] = mean_pos - mean_neg

        # Step 2: For each layer, inject task vector and measure KL divergence
        layer_kl_divs = []

        # Sample layers to test (avoid testing all for speed)
        if num_layers > 10:
            test_layers = [0, num_layers // 4, num_layers // 2,
                           3 * num_layers // 4, num_layers - 1]
        else:
            test_layers = list(range(num_layers))

        with torch.no_grad():
            for s in samples:
                neutral_ids = tokenizer.encode(s["neutral"], return_tensors="pt",
                                               truncation=True, max_length=128).to(device)

                # Baseline output distribution
                base_out = model(neutral_ids)
                base_probs = F.softmax(base_out.logits[0, -1], dim=-1)
                base_log_probs = F.log_softmax(base_out.logits[0, -1], dim=-1)

                for l_idx in test_layers:
                    tv = task_vectors[l_idx].to(device)

                    def get_steering_hook(vec, alpha):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                out_t = output[0].clone()
                                out_t[:, -1, :] += alpha * vec
                                return (out_t,) + output[1:]
                            else:
                                out_t = output.clone()
                                out_t[:, -1, :] += alpha * vec
                                return out_t
                        return hook

                    handle = layers[l_idx].register_forward_hook(
                        get_steering_hook(tv, steering_alpha)
                    )
                    try:
                        steered_out = model(neutral_ids)
                        steered_log_probs = F.log_softmax(
                            steered_out.logits[0, -1], dim=-1
                        )
                        # KL(base || steered)
                        kl = F.kl_div(steered_log_probs, base_probs,
                                      reduction='sum', log_target=False).item()
                        layer_kl_divs.append((l_idx, max(0.0, kl)))
                    finally:
                        handle.remove()

        if not layer_kl_divs:
            return {"error": "No steering results computed."}

        # Aggregate per-layer
        from collections import defaultdict
        kl_by_layer = defaultdict(list)
        for l_idx, kl in layer_kl_divs:
            kl_by_layer[l_idx].append(kl)

        layer_mean_kl = {l: float(np.mean(kls)) for l, kls in kl_by_layer.items()}
        kl_values = list(layer_mean_kl.values())
        kl_layers = list(layer_mean_kl.keys())

        best_idx = int(np.argmax(kl_values))
        best_layer = kl_layers[best_idx]

        # Success rate: fraction of layers where KL > threshold
        threshold = self.config.get("steering_threshold", 0.01)
        success_rate = sum(1 for v in kl_values if v > threshold) / len(kl_values)

        return {
            "layer_steering_kl_divergence": layer_mean_kl,
            "best_steering_layer": int(best_layer),
            "best_steering_kl": float(kl_values[best_idx]),
            "steering_success_rate": float(success_rate),
        }

