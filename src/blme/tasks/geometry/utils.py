"""Geometry utility functions — universal HuggingFace compatibility.

Uses `output_hidden_states=True` instead of architecture-specific hooks
so that all functions work with any AutoModelForCausalLM.
"""

import torch
import numpy as np
from tqdm import tqdm
from ..common import get_embeddings
import os
import tempfile
import atexit

_OFFLOAD_FILES = []

def _cleanup_offload_files():
    for f in _OFFLOAD_FILES:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            pass

atexit.register(_cleanup_offload_files)

def _get_temp_dat_path():
    fd, path = tempfile.mkstemp(prefix="blme_offload_", suffix=".dat")
    os.close(fd)
    _OFFLOAD_FILES.append(path)
    return path


def collect_hidden_states(model, tokenizer, dataset, num_samples=100, layer_idx=-1, use_disk_offload=False):
    """Collect hidden states from a specific layer or all layers.

    Uses `output_hidden_states=True` for universal compatibility.

    Args:
        model: HuggingFace causal LM
        tokenizer: corresponding tokenizer
        dataset: list of dicts with 'text' key or strings
        num_samples: max samples to process
        layer_idx: int for specific layer, "all" for all layers.
                   -1 means the final layer.
        use_disk_offload: bool, if True streams tensors to memory-mapped disk files.

    Returns:
        If layer_idx != "all": single tensor (TotalTokens, D)
        If layer_idx == "all": dict {layer_idx: (TotalTokens, D)}
    """
    use_offload = use_disk_offload or (os.environ.get("BLME_DISK_OFFLOAD", "0") == "1")
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
                    if use_offload:
                        if li not in all_hidden:
                            path = _get_temp_dat_path()
                            f = open(path, "wb")
                            all_hidden[li] = {"path": path, "handle": f, "count": 0, "D": h_flat.shape[-1]}
                        f = all_hidden[li]["handle"]
                        f.write(h_flat.numpy().tobytes())
                        all_hidden[li]["count"] += h_flat.shape[0]
                    else:
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
                if use_offload:
                    if layer_idx not in all_hidden:
                        path = _get_temp_dat_path()
                        f = open(path, "wb")
                        all_hidden[layer_idx] = {"path": path, "handle": f, "count": 0, "D": h_flat.shape[-1]}
                    f = all_hidden[layer_idx]["handle"]
                    f.write(h_flat.numpy().tobytes())
                    all_hidden[layer_idx]["count"] += h_flat.shape[0]
                else:
                    if layer_idx not in all_hidden:
                        all_hidden[layer_idx] = []
                    all_hidden[layer_idx].append(h_flat)

    # Concatenate or MemMap
    results = {}
    if use_offload:
        for idx, meta in all_hidden.items():
            meta["handle"].close()
            if meta["count"] > 0:
                mmap_array = np.memmap(meta["path"], dtype=np.float32, mode='r', shape=(meta["count"], meta["D"]))
                results[idx] = torch.from_numpy(mmap_array)
            else:
                results[idx] = torch.empty(0)
    else:
        for idx, tensors in all_hidden.items():
            results[idx] = torch.cat(tensors, dim=0) if tensors else torch.empty(0)

    if layer_idx != "all":
        return list(results.values())[0] if results else torch.empty(0)
    else:
        return results


def collect_prediction_stats(model, tokenizer, dataset, num_samples=100, use_disk_offload=False):
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
    use_offload = use_disk_offload or (os.environ.get("BLME_DISK_OFFLOAD", "0") == "1")
    stats = {
        "token_counts": np.zeros(tokenizer.vocab_size),
    }

    if use_offload:
        p_logits, p_labels, p_hidden = _get_temp_dat_path(), _get_temp_dat_path(), _get_temp_dat_path()
        f_logits, f_labels, f_hidden = open(p_logits, "wb"), open(p_labels, "wb"), open(p_hidden, "wb")
        c_logits = c_labels = c_hidden = 0
        D_logits = D_hidden = 0
    else:
        stats["logits"] = []
        stats["labels"] = []
        stats["hidden"] = []

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

            l_flat = logits.reshape(-1, logits.shape[-1]).numpy()
            y_flat = labels.reshape(-1).numpy()
            h_flat = h_pred.reshape(-1, h_pred.shape[-1]).numpy()

            if use_offload:
                f_logits.write(l_flat.tobytes())
                f_labels.write(y_flat.tobytes())
                f_hidden.write(h_flat.tobytes())
                c_logits += l_flat.shape[0]
                c_labels += y_flat.shape[0]
                c_hidden += h_flat.shape[0]
                D_logits, D_hidden = l_flat.shape[-1], h_flat.shape[-1]
            else:
                stats["logits"].append(torch.from_numpy(l_flat))
                stats["labels"].append(torch.from_numpy(y_flat))
                stats["hidden"].append(torch.from_numpy(h_flat))

            ids_flat = input_ids.view(-1).numpy()
            np.add.at(stats["token_counts"], ids_flat, 1)

    if use_offload:
        f_logits.close()
        f_labels.close()
        f_hidden.close()
        if c_logits > 0:
            m_logits = np.memmap(p_logits, dtype=np.float32, mode='r', shape=(c_logits, D_logits))
            m_labels = np.memmap(p_labels, dtype=np.int64, mode='r', shape=(c_labels,))
            m_hidden = np.memmap(p_hidden, dtype=np.float32, mode='r', shape=(c_hidden, D_hidden))
            stats["logits"] = [torch.from_numpy(m_logits)]
            stats["labels"] = [torch.from_numpy(m_labels)]
            stats["hidden"] = [torch.from_numpy(m_hidden)]
        else:
            stats["logits"], stats["labels"], stats["hidden"] = [], [], []

    return stats, embeddings
