import hashlib

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


def _stable_prompt_seed(prompt: str, base_seed: int = 1, modulo: int = 10_000) -> int:
    """Derive a cross-process-stable seed offset from prompt text."""
    digest = hashlib.blake2b(str(prompt).encode("utf-8"), digest_size=8).digest()
    prompt_offset = int.from_bytes(digest, byteorder="big", signed=False) % modulo
    return int(base_seed) + prompt_offset


def _resolve_noise_std(
    model,
    user_value=None,
    mul: float = 3.0,
    subject_strings=None,
    tokenizer=None,
) -> float:
    """Return the Gaussian noise σ to use for subject-embedding corruption.

    ROME (Meng et al. 2022, §3.2) uses ``3 × σ`` where σ is the sample
    standard deviation of **the embedding outputs for subject strings**
    from the Knowns dataset (``kmeng01/rome::collect_embedding_std``).
    When a list of representative subject strings is provided alongside
    a tokenizer we match the paper exactly; otherwise we fall back to
    the full embedding-matrix std (a close approximation). Either way,
    a fixed constant like 0.1 under-corrupts models with wide
    embeddings and over-corrupts ones with narrow ones.
    """
    if user_value is not None:
        return float(user_value)

    if subject_strings and tokenizer is not None:
        try:
            device = next(model.parameters()).device
            embed_module = model.get_input_embeddings()
            all_emb = []
            with torch.no_grad():
                for s in subject_strings:
                    ids = tokenizer(s, return_tensors="pt", add_special_tokens=False)
                    if "input_ids" in ids and ids["input_ids"].numel() > 0:
                        out = embed_module(ids["input_ids"].to(device))
                        all_emb.append(out.reshape(-1, out.shape[-1]).detach().float().cpu())
            if all_emb:
                cat = torch.cat(all_emb, dim=0)
                sigma = float(cat.std().item())
                if np.isfinite(sigma) and sigma > 0:
                    return mul * sigma
        except Exception:
            pass

    try:
        emb = model.get_input_embeddings().weight.detach().float()
        sigma = float(emb.std().item())
        if np.isfinite(sigma) and sigma > 0:
            return mul * sigma
    except Exception:
        pass
    return 0.1  # Safe fallback matching the legacy default.


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
        n_noise_samples = int(self.config.get("n_noise_samples", 10))

        device = next(model.parameters()).device
        layers = get_layers(model)
        num_layers = len(layers)

        # ROME Figure 2 reports AIE at every layer; we match that by
        # default. A user can still override via config to sub-sample
        # if compute is tight on deep models.
        trace_layers_cfg = self.config.get("trace_layers")
        if trace_layers_cfg is None:
            trace_layers = list(range(num_layers))
        else:
            trace_layers = [int(l) for l in trace_layers_cfg]

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

        # Resolve noise σ from subject embeddings (ROME's
        # ``collect_embedding_std``), falling back to full-E std when
        # subjects cannot be embedded.
        noise_std = _resolve_noise_std(
            model,
            user_value=self.config.get("noise_std"),
            subject_strings=[s for _, s, _ in triples],
            tokenizer=tokenizer,
        )

        # Noise is injected via a forward hook on the text embedding
        # module. We batch ``n_noise_samples + 1`` copies of every
        # prompt in a single forward pass: row 0 stays *clean* and
        # rows ``1..N`` each get their own independent Gaussian draw at
        # the subject slice. A single forward pass per layer then
        # gives us the mean-across-noise-draws prob via
        # ``logits[1:, -1, :].softmax(-1).mean(0)`` — 10× faster than
        # the sequential-noise implementation and matches the ROME
        # reference (kmeng01/rome ``causal_trace.py``).
        embed_module = model.get_input_embeddings()
        noise_state = {"enabled": False, "start": 0, "end": 0, "noise": None}

        def embed_noise_hook(module, inputs, output):
            if not noise_state["enabled"]:
                return output
            s = noise_state["start"]
            e = noise_state["end"]
            noise = noise_state["noise"]  # shape (N, e-s, D) or (1, e-s, D)
            if noise is None:
                return output
            if output.shape[0] <= noise.shape[0]:
                # Non-batched call (e.g. pure clean run) — leave as is.
                return output
            corrupted = output.clone()
            # Rows 1..N get noise; row 0 stays untouched (the clean ref).
            corrupted[1:noise.shape[0] + 1, s:e, :] = (
                corrupted[1:noise.shape[0] + 1, s:e, :]
                + noise.to(corrupted.dtype).to(corrupted.device)
            )
            return corrupted

        embed_hook_handle = embed_module.register_forward_hook(embed_noise_hook)

        results_by_layer = {layer_idx: [] for layer_idx in trace_layers}
        per_prompt_metadata = []

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

                    # Account for any special token at position 0 added by
                    # encode() (e.g. BOS for Llama). We compare against a
                    # no-special-tokens tokenisation.
                    base_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                    offset = seq_len - len(base_ids)
                    if offset > 0:
                        corrupt_idx_start += offset
                        corrupt_idx_end += offset
                    corrupt_idx_end = min(corrupt_idx_end, seq_len)
                    if corrupt_idx_end <= corrupt_idx_start:
                        continue

                    target_idx = seq_len - 1  # last prompt position

                    # Resolve the target token id (first token of `target`).
                    target_token_ids = tokenizer.encode(target, add_special_tokens=False)
                    if not target_token_ids:
                        continue
                    target_token_id = target_token_ids[0]

                    N = max(1, n_noise_samples)
                    # Build the batched input: 1 clean + N noisy copies.
                    batched_ids = input_ids.repeat(N + 1, 1)

                    # Draw all N noise samples at once (one tensor of
                    # shape (N, span, D)). Deterministic per-prompt seed
                    # so reruns produce the same AIE.
                    rng_gen = torch.Generator(device="cpu").manual_seed(
                        _stable_prompt_seed(prompt, base_seed=int(self.config.get("seed", 1)))
                    )
                    span = corrupt_idx_end - corrupt_idx_start
                    try:
                        embed_dim = int(embed_module.weight.shape[-1])
                    except Exception:
                        embed_dim = int(model.get_input_embeddings()(input_ids[:, :1]).shape[-1])

                    noise = (
                        torch.randn(
                            (N, span, embed_dim),
                            generator=rng_gen,
                            dtype=torch.float32,
                        ) * noise_std
                    )
                    noise_state["start"] = corrupt_idx_start
                    noise_state["end"] = corrupt_idx_end
                    noise_state["noise"] = noise

                    # --- Clean + corrupted forward pass (batched) ---
                    noise_state["enabled"] = True
                    out0 = model(batched_ids, output_hidden_states=True)
                    # Row 0 is the clean run; rows 1..N are corrupted.
                    clean_logits = out0.logits[0, target_idx]
                    clean_probs = F.softmax(clean_logits, dim=-1)
                    clean_prob_target = clean_probs[target_token_id].item()

                    corrupted_probs = F.softmax(
                        out0.logits[1:, target_idx], dim=-1,
                    ).mean(dim=0)
                    corrupted_prob_target = corrupted_probs[target_token_id].item()

                    if clean_prob_target - corrupted_prob_target <= 0:
                        # Noise didn't hurt — nothing to restore.
                        continue

                    # Cache row-0 (clean) hidden states for patching.
                    clean_states = [h.detach() for h in out0.hidden_states]

                    # --- Restoration sweep: one forward per layer ---
                    # The patch hook replaces rows 1..N's hidden state
                    # at the subject slice with row 0's (clean).
                    def get_patch_hook(clean_state_to_patch):
                        def patch_hook(module, inputs, output):
                            if isinstance(output, tuple):
                                out_tensor = output[0].clone()
                                out_tensor[1:, corrupt_idx_start:corrupt_idx_end, :] = (
                                    clean_state_to_patch[0:1, corrupt_idx_start:corrupt_idx_end, :]
                                )
                                return (out_tensor,) + output[1:]
                            else:
                                out_tensor = output.clone()
                                out_tensor[1:, corrupt_idx_start:corrupt_idx_end, :] = (
                                    clean_state_to_patch[0:1, corrupt_idx_start:corrupt_idx_end, :]
                                )
                                return out_tensor
                        return patch_hook

                    for layer_idx in trace_layers:
                        hook = layers[layer_idx].register_forward_hook(
                            get_patch_hook(clean_states[layer_idx + 1])
                        )
                        try:
                            restored_out = model(batched_ids)
                            restored_probs = F.softmax(
                                restored_out.logits[1:, target_idx], dim=-1,
                            ).mean(dim=0)
                            restored_prob_target = restored_probs[target_token_id].item()
                            aie = restored_prob_target - corrupted_prob_target
                            results_by_layer[layer_idx].append(aie)
                        finally:
                            hook.remove()

                    per_prompt_metadata.append({
                        "clean_prob": clean_prob_target,
                        "corrupted_prob": corrupted_prob_target,
                    })
                    noise_state["enabled"] = False
        finally:
            embed_hook_handle.remove()
                        
        results = {
            "traced_layers": list(trace_layers),
            "n_noise_samples": int(max(1, n_noise_samples)),
        }
        per_layer_aie = {}  # int layer idx -> mean AIE (may be negative)
        for l_idx, aies in results_by_layer.items():
            if aies:
                mean_aie = float(np.mean(aies))
                results[f"layer_{l_idx}_aie"] = mean_aie
                per_layer_aie[int(l_idx)] = mean_aie

        # Find the centre of causal effect and quantify localization via
        # the entropy of the (clipped-to-non-negative, renormalised) AIE
        # distribution. Expose the peak layer as both a string key (for
        # backward compatibility with older result readers) and as an
        # integer — the aggregation pipeline needs a numeric column.
        if per_layer_aie:
            peak_layer = max(per_layer_aie, key=per_layer_aie.get)
            results["max_causal_layer"] = f"layer_{peak_layer}"
            results["max_causal_layer_idx"] = int(peak_layer)
            results["max_aie"] = float(per_layer_aie[peak_layer])
            results["noise_std_applied"] = float(noise_std)

            # Entropy: only positive restorations carry "localization"
            # signal; negatives mean the patch hurt the restored prob and
            # should not get mass in the distribution.
            aie_arr = np.array(
                [max(0.0, per_layer_aie[k]) for k in sorted(per_layer_aie)],
                dtype=np.float64,
            )
            if aie_arr.sum() > 0:
                p_aie = aie_arr / aie_arr.sum()
                # 0 log 0 = 0: mask zeros to avoid NaN.
                nonzero = p_aie[p_aie > 0]
                causal_entropy = float(-np.sum(nonzero * np.log(nonzero)))
            else:
                causal_entropy = 0.0

            results["causal_entropy"] = causal_entropy

        return results
