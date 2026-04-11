import torch
import torch.nn.functional as F
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_layers, get_embeddings
import logging
logger = logging.getLogger("blme")

# Hardcoded fallback triples (prompt, subject, target_true) used when the
# `datasets` library is unavailable. These are simple factual statements
# where the subject is a clearly delimited substring of the prompt.
_FALLBACK_FACTS = [
    ("The Space Needle is located in the city of", "The Space Needle", " Seattle"),
    ("The Eiffel Tower is located in the city of", "The Eiffel Tower", " Paris"),
    ("The capital of France is", "France", " Paris"),
    ("The Great Wall is located in the country of", "The Great Wall", " China"),
    ("The Statue of Liberty stands in the city of", "The Statue of Liberty", " New"),
]


def _find_subject_token_range(tokenizer, prompt, subject):
    """Locate the token indices in `prompt` that correspond to `subject`.

    Returns (start, end) where end is exclusive, or None if the subject
    cannot be unambiguously located in the tokenized prompt. Uses
    return_offsets_mapping when available; otherwise falls back to a
    decoded-substring scan.
    """
    char_start = prompt.find(subject)
    if char_start < 0:
        return None
    char_end = char_start + len(subject)

    # Preferred: offset mapping (works for most fast tokenizers).
    try:
        enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
        tok_start, tok_end = None, None
        for i, (s, e) in enumerate(offsets):
            if s == e:
                continue  # special token
            if tok_start is None and e > char_start:
                tok_start = i
            if s < char_end:
                tok_end = i + 1
        if tok_start is not None and tok_end is not None and tok_end > tok_start:
            return tok_start, tok_end
    except Exception:
        pass

    # Fallback: tokenize prefix and full string, diff the lengths. Less
    # reliable but works for slow tokenizers (e.g. SentencePiece without
    # offsets).
    try:
        ids_prefix = tokenizer(prompt[:char_start], add_special_tokens=False)["input_ids"]
        ids_full = tokenizer(prompt[:char_end], add_special_tokens=False)["input_ids"]
        if len(ids_full) > len(ids_prefix):
            return len(ids_prefix), len(ids_full)
    except Exception:
        pass
    return None


@register_task("causality_tracing")
class CausalTracingTask(DiagnosticTask):
    """
    Implements ROME-style causal tracing (Meng et al. 2022, arXiv:2202.05262).
    For each (prompt, subject, target) triple:
      1. Run the clean prompt and record `P(target | prompt)`.
      2. Add Gaussian noise to the *subject token embeddings* and run again
         to get `P(target | corrupted prompt)`.
      3. For each traced layer, restore the clean hidden state at the
         subject token positions only, and measure how much of the lost
         probability is recovered. This is the AIE per layer.

    Returns the per-layer AIE, the layer of maximum AIE, and the entropy of
    the (normalized) AIE distribution as a measure of localization.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Causal Tracing...")
        num_samples = self.config.get("num_samples", 3)
        noise_std = self.config.get("noise_std", 0.1)

        device = next(model.parameters()).device
        layers = get_layers(model)
        num_layers = len(layers)

        # Determine the number of layers to sample for tracing (to speed up)
        # We trace early, middle, and late layers.
        if num_layers > 10:
            trace_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
        else:
            trace_layers = list(range(num_layers))

        # Build the (prompt, subject, target) triples.
        # Build the (prompt, subject, target) triples.
        triples = []
        if dataset is None:
            try:
                from datasets import load_dataset
                dset = load_dataset("NeelNanda/counterfact-tracing", split="train")
                for i in range(min(num_samples, len(dset))):
                    item = dset[i]
                    prompt = item["prompt"]
                    subject = item["subject"]
                    target = item.get("target_true", "")
                    if not target.startswith(" "):
                        target = " " + target
                    triples.append((prompt, subject, target))
            except (ImportError, Exception) as e:
                logger.info(f"Warning: counterfact-tracing not available ({type(e).__name__}). Using fallback facts.")
                while len(triples) < num_samples:
                    for t in _FALLBACK_FACTS:
                        triples.append(t)
                        if len(triples) >= num_samples:
                            break
        else:
            for s in list(dataset)[:num_samples]:
                if isinstance(s, dict) and "prompt" in s and "subject" in s:
                    target = s.get("target_true", s.get("target", ""))
                    if target and not target.startswith(" "):
                        target = " " + target
                    triples.append((s["prompt"], s["subject"], target))
                elif isinstance(s, dict) and "text" in s:
                    # Dataset provided but lacks subject annotation — the
                    # default BLME cache corpus hits this branch. Fall back
                    # to the bundled facts so the task still produces output.
                    continue

            # If nothing survived (e.g. default corpus with only 'text'),
            # fall back to the bundled facts.
            if not triples:
                logger.info("  dataset lacks (prompt, subject) triples — using bundled fallback facts")
                while len(triples) < num_samples:
                    for t in _FALLBACK_FACTS:
                        triples.append(t)
                        if len(triples) >= num_samples:
                            break

        triples = triples[:num_samples]
        if not triples:
            return {"error": "No (prompt, subject, target) triples to trace"}

        # Noise injection is performed via a forward hook on the input
        # embedding module rather than by passing ``inputs_embeds=`` to the
        # model. This is portable across architectures: multimodal wrappers
        # like Gemma 4 reject ``model(inputs_embeds=...)`` without
        # ``input_ids`` because they need the token ids for placeholder
        # routing, but they happily call the text embedding module whose
        # output we can mutate from a hook.
        embed_module = model.get_input_embeddings()
        noise_state = {"enabled": False, "start": 0, "end": 0, "noise": None}

        def embed_noise_hook(module, inputs, output):
            if not noise_state["enabled"]:
                return output
            s = noise_state["start"]
            e = noise_state["end"]
            noise = noise_state["noise"]
            # Cast noise to the embedding output dtype (e.g. bfloat16) to
            # avoid dtype mismatches on models that store embeddings in
            # reduced precision.
            corrupted = output.clone()
            corrupted[:, s:e, :] = corrupted[:, s:e, :] + noise.to(corrupted.dtype)
            return corrupted

        embed_hook_handle = embed_module.register_forward_hook(embed_noise_hook)

        results_by_layer = {layer_idx: [] for layer_idx in trace_layers}

        try:
            with torch.no_grad():
                for prompt, subject, target in triples:
                    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                    seq_len = input_ids.shape[1]
                    if seq_len < 2:
                        continue

                    # Locate subject tokens in the prompt.
                    rng = _find_subject_token_range(tokenizer, prompt, subject)
                    if rng is None:
                        logger.info(f"Skipping — could not locate subject '{subject}' in '{prompt}'")
                        continue
                    corrupt_idx_start, corrupt_idx_end = rng

                    # Account for any special token at position 0 added by encode().
                    # tokenizer(prompt, add_special_tokens=False) was used in
                    # _find_subject_token_range, but tokenizer.encode adds bos for
                    # some models. Detect by length difference.
                    base_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                    offset = seq_len - len(base_ids)
                    if offset > 0:
                        corrupt_idx_start += offset
                        corrupt_idx_end += offset
                    corrupt_idx_end = min(corrupt_idx_end, seq_len)
                    if corrupt_idx_end <= corrupt_idx_start:
                        continue

                    # The "predict at" position is the last token of the prompt
                    # (next-token prediction).
                    target_idx = seq_len - 1

                    # Resolve the target token id (first token of `target`).
                    target_token_ids = tokenizer.encode(target, add_special_tokens=False)
                    if not target_token_ids:
                        continue
                    target_token_id = target_token_ids[0]

                    # 1. Clean Run — noise hook disabled.
                    noise_state["enabled"] = False
                    clean_out = model(input_ids, output_hidden_states=True)
                    clean_logits = clean_out.logits[0, target_idx]
                    clean_probs = F.softmax(clean_logits, dim=-1)
                    clean_prob_target = clean_probs[target_token_id].item()

                    # Cache the clean hidden states
                    clean_states = [h.detach() for h in clean_out.hidden_states]

                    # Generate per-sample noise once — subsequent restoration
                    # runs must reuse the same noise so corrupted baselines
                    # are consistent. Shape matches the subject slice of
                    # the embedding output (clean_states[0] is the embedding
                    # output before any transformer block).
                    clean_embed_out = clean_states[0]
                    noise = torch.randn_like(
                        clean_embed_out[:, corrupt_idx_start:corrupt_idx_end, :],
                        dtype=torch.float32,
                    ) * noise_std
                    noise_state["start"] = corrupt_idx_start
                    noise_state["end"] = corrupt_idx_end
                    noise_state["noise"] = noise

                    # 2. Corrupted Run — activate noise hook; forward with
                    # input_ids so multimodal wrappers route correctly.
                    noise_state["enabled"] = True
                    corrupted_out = model(input_ids)
                    corrupted_logits = corrupted_out.logits[0, target_idx]
                    corrupted_probs = F.softmax(corrupted_logits, dim=-1)
                    corrupted_prob_target = corrupted_probs[target_token_id].item()

                    # Calculate the maximum possible restoration
                    max_restoration = clean_prob_target - corrupted_prob_target
                    if max_restoration <= 0:
                        noise_state["enabled"] = False
                        continue  # Skip if noise didn't hurt the prediction

                    # 3. Restored Runs (Layer by Layer Patching).
                    # The embedding noise hook stays active so every
                    # restoration run sees the same corrupted embeddings;
                    # the layer patch hook restores the clean hidden state
                    # at the subject slice for one specific layer.
                    for layer_idx in trace_layers:

                        def get_patch_hook(clean_state_to_patch):
                            def patch_hook(module, input, output):
                                if isinstance(output, tuple):
                                    out_tensor = output[0].clone()
                                    out_tensor[:, corrupt_idx_start:corrupt_idx_end, :] = clean_state_to_patch[:, corrupt_idx_start:corrupt_idx_end, :]
                                    return (out_tensor,) + output[1:]
                                else:
                                    out_tensor = output.clone()
                                    out_tensor[:, corrupt_idx_start:corrupt_idx_end, :] = clean_state_to_patch[:, corrupt_idx_start:corrupt_idx_end, :]
                                    return out_tensor
                            return patch_hook

                        # hidden_states stores [embedding_out, layer_0_out, ...]
                        # So clean_states[layer_idx + 1] corresponds to the output of layers[layer_idx]
                        hook = layers[layer_idx].register_forward_hook(get_patch_hook(clean_states[layer_idx + 1]))

                        try:
                            restored_out = model(input_ids)
                            restored_logits = restored_out.logits[0, target_idx]
                            restored_probs = F.softmax(restored_logits, dim=-1)
                            restored_prob_target = restored_probs[target_token_id].item()

                            # Calculate Average Indirect Effect (AIE)
                            # How much of the lost probability did we get back?
                            aie = (restored_prob_target - corrupted_prob_target) / max_restoration
                            results_by_layer[layer_idx].append(aie)

                        finally:
                            hook.remove()

                    # Disable the noise hook until the next sample installs
                    # fresh noise — prevents accidental corruption leaking
                    # between samples.
                    noise_state["enabled"] = False
        finally:
            embed_hook_handle.remove()
                        
        results = {}
        aie_list = []
        for l_idx, aies in results_by_layer.items():
            if aies:
                mean_aie = float(np.mean(aies))
                results[f"layer_{l_idx}_aie"] = mean_aie
                # We clamp negative AIE to a small positive epsilon for entropy calculation
                aie_list.append(max(mean_aie, 1e-10))
                
        # Find the center of causal effect and evaluate the entropy (localization vs distribution)
        if aie_list:
            max_layer = max((k for k in results if "_aie" in k), key=results.get)
            results["max_causal_layer"] = max_layer
            results["max_aie"] = results[max_layer]
            
            # Causal Entropy
            aie_arr = np.array(aie_list)
            # Normalize to form a probability distribution
            if np.sum(aie_arr) > 0:
                p_aie = aie_arr / np.sum(aie_arr)
                causal_entropy = -np.sum(p_aie * np.log(p_aie + 1e-12))
            else:
                causal_entropy = 0.0
            
            results["causal_entropy"] = float(causal_entropy)
            
        return results
