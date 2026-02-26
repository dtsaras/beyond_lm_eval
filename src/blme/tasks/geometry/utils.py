"""Geometry utility functions — universal HuggingFace compatibility.

Uses `output_hidden_states=True` instead of architecture-specific hooks
so that all functions work with any AutoModelForCausalLM.
"""

import torch
import numpy as np
from tqdm import tqdm
from ..common import get_embeddings


def collect_hidden_states(model, tokenizer, dataset, num_samples=100, layer_idx=-1):
    """Collect hidden states from a specific layer or all layers.

    Uses `output_hidden_states=True` for universal compatibility.

    Args:
        model: HuggingFace causal LM
        tokenizer: corresponding tokenizer
        dataset: list of dicts with 'text' key or strings
        num_samples: max samples to process
        layer_idx: int for specific layer, "all" for all layers.
                   -1 means the final layer.

    Returns:
        If layer_idx != "all": single tensor (TotalTokens, D)
        If layer_idx == "all": dict {layer_idx: (TotalTokens, D)}
    """
    all_hidden = {}  # layer_index -> list of (N_tokens, D) tensors

    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc="Collecting states")):
            if i >= num_samples:
                break

            if isinstance(sample, str):
                inputs = tokenizer(sample, return_tensors="pt").to(model.device)
            else:
                text = sample.get("text", "")
                inputs = tokenizer(text, return_tensors="pt").to(model.device)

            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # tuple of (1, T, D)

            # hidden_states[0] = embedding output
            # hidden_states[i+1] = output of layer i
            n_layers = len(hidden_states) - 1  # subtract embedding layer

            if layer_idx == "all":
                for li in range(n_layers):
                    h = hidden_states[li + 1]  # skip embedding layer
                    h_flat = h.reshape(-1, h.shape[-1]).float().detach().cpu()
                    if li not in all_hidden:
                        all_hidden[li] = []
                    all_hidden[li].append(h_flat)
            else:
                # Resolve negative index
                actual_idx = layer_idx if layer_idx >= 0 else n_layers + layer_idx
                # Clamp
                actual_idx = max(0, min(actual_idx, n_layers - 1))

                h = hidden_states[actual_idx + 1]  # +1 to skip embedding

                # Subsample tokens for single-layer mode
                T = h.shape[1]
                if T > 10:
                    indices = torch.randperm(T)[:10]
                    h = h[:, indices, :]

                h_flat = h.reshape(-1, h.shape[-1]).float().detach().cpu()
                if layer_idx not in all_hidden:
                    all_hidden[layer_idx] = []
                all_hidden[layer_idx].append(h_flat)

    # Concatenate
    results = {}
    for idx, tensors in all_hidden.items():
        results[idx] = torch.cat(tensors, dim=0)

    if layer_idx != "all":
        return list(results.values())[0] if results else torch.empty(0)
    else:
        return results


def collect_prediction_stats(model, tokenizer, dataset, num_samples=100):
    """Collect logits, labels, and hidden states for consistency analysis.

    Uses `output_hidden_states=True` for universal compatibility.

    Returns:
        (stats, embeddings) where:
        - stats["logits"]: list of (N_tokens, V) tensors
        - stats["labels"]: list of (N_tokens,) tensors
        - stats["hidden"]: list of (N_tokens, D) tensors
        - stats["token_counts"]: numpy array of shape (vocab_size,)
        - embeddings: (V, D) tensor or None
    """
    stats = {
        "logits": [],
        "labels": [],
        "hidden": [],
        "token_counts": np.zeros(tokenizer.vocab_size),
    }

    embeddings = get_embeddings(model)

    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc="Collecting prediction stats")):
            if i >= num_samples:
                break

            if isinstance(sample, str):
                inputs = tokenizer(sample, return_tensors="pt").to(model.device)
            else:
                text = sample.get("text", "")
                inputs = tokenizer(text, return_tensors="pt").to(model.device)

            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits.float().detach().cpu()

            input_ids = inputs["input_ids"].cpu()
            labels = input_ids[:, 1:]
            logits = logits[:, :-1, :]

            # Last hidden state (before LM head)
            h = outputs.hidden_states[-1].float().detach().cpu()
            h_pred = h[:, :-1, :]

            stats["logits"].append(logits.reshape(-1, logits.shape[-1]))
            stats["labels"].append(labels.reshape(-1))
            stats["hidden"].append(h_pred.reshape(-1, h_pred.shape[-1]))

            ids_flat = input_ids.view(-1).numpy()
            np.add.at(stats["token_counts"], ids_flat, 1)

    return stats, embeddings
