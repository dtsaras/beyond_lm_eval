"""Logit Lens — decode intermediate hidden states via the final LM head.

Originally proposed by nostalgebraist (2020) — *interpreting GPT: the
logit lens* — the technique projects every layer's residual stream
through the model's own ``lm_head`` to get per-layer token
predictions, then reports per-layer accuracy and entropy relative to
the final prediction.

References:
  * nostalgebraist 2020 — https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/
    "interpreting GPT: the logit lens".
  * Belrose, Furman, Smith, Halawi, Ostrovsky, McKinney, Biderman,
    Steinhardt 2023 — "Eliciting Latent Predictions from Transformers
    with the Tuned Lens", arXiv:2303.08112 (the learned-probe variant
    — deliberately not used here, since we want the model-quality
    signal that tuned probes wash out).
"""

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings, get_layers, apply_lm_head, get_final_norm
import torch
import torch.nn.functional as F
import numpy as np
import logging
logger = logging.getLogger("blme")

@register_task("interpretability_logit_lens")
class LogitLensTask(DiagnosticTask):
    """
    Decodes hidden states at each layer using the final LM head (Logit Lens).
    Computes accuracy of intermediate layers relative to the final prediction.

    Implementation note: HuggingFace causal LMs return hidden_states as a tuple
    (embedding_output, layer_0_output, layer_1_output, ..., layer_{N-1}_output),
    so the per-layer outputs are at indices 1..N. We always slice from index 1.

    Important: models like LLaMA / Qwen / Mistral / Gemma apply a final RMSNorm
    *before* the LM head. Skipping that norm makes early-layer logits
    architecturally biased and breaks cross-architecture comparison. We apply
    `get_final_norm(model)` (when present) to every intermediate hidden state
    before projecting through the unembedding.

    Caveat: even with the final norm applied, the residual stream is only
    approximately interpretable at intermediate layers — features can still be
    in superposition. Tuned-lens style learned probes would be more accurate
    but trade away the model-quality signal we want to measure here.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Logit Lens Analysis...")
        num_samples = self.config.get("num_samples", 100)

        device = next(model.parameters()).device

        if dataset is None:
             from ...cache import load_default_corpus
             dataset = load_default_corpus(num_samples)
        # Detect layers (universal)
        layers = get_layers(model)
        if layers is None:
            # Fallback: use config
            from ..common import get_num_layers
            n_layers = get_num_layers(model)
            if n_layers == 0:
                return {"error": "Could not detect layers"}
        else:
            n_layers = len(layers)

        # Final norm (applied before LM head in modern transformers)
        final_norm = get_final_norm(model)

        layer_accs = {i: [] for i in range(n_layers)}
        layer_entropies = {i: [] for i in range(n_layers)}

        count = 0
        with torch.no_grad():
            for sample in dataset:
                if count >= num_samples: break

                if isinstance(sample, str):
                    inputs = tokenizer(sample, return_tensors="pt").to(device)
                elif isinstance(sample, dict) and 'text' in sample:
                    inputs = tokenizer(sample['text'], return_tensors="pt", truncation=True, max_length=128).to(device)
                elif 'input_ids' in sample:
                     inputs = {'input_ids': torch.tensor(sample['input_ids']).long().unsqueeze(0).to(device)}
                else: continue
                count += 1

                outputs = model(**inputs, output_hidden_states=True)
                final_preds = outputs.logits[0].argmax(dim=-1)

                # HF convention: hidden_states[0] = embedding output,
                # hidden_states[1..N-1] = output of each transformer block,
                # hidden_states[N] (the last entry) is the output of the
                # final transformer block AFTER the model's final LayerNorm
                # — this was verified empirically on GPT-2 and Llama in
                # transformers 5.x: ``lm_head(hidden_states[-1])`` exactly
                # reproduces ``outputs.logits``. So the last entry must
                # NOT get another pass of ``final_norm``, or we'd
                # double-norm and systematically corrupt the final-layer
                # logit-lens accuracy/entropy (reviewer-visible:
                # pre-fix accuracy dipped on the last layer for every
                # GPT-2 / Pythia / OLMo / Phi-2).
                per_layer_states = outputs.hidden_states[1:]
                last_state_is_normed = len(outputs.hidden_states) == n_layers + 1
                if len(per_layer_states) < n_layers:
                    n_used = len(per_layer_states)
                else:
                    n_used = n_layers

                for i in range(n_used):
                    h = per_layer_states[i][0]  # (T, D)
                    # The last entry in ``hidden_states`` is already
                    # post-final-norm in modern ``transformers``; only
                    # apply ``final_norm`` to the earlier layers.
                    is_last = last_state_is_normed and i == len(per_layer_states) - 1

                    # Apply final norm before unembedding so intermediate
                    # layers are projected on the same footing as the final
                    # layer (matters for RMSNorm-based architectures).
                    if final_norm is not None and not is_last:
                        try:
                            norm_param = next(final_norm.parameters())
                            norm_dtype = norm_param.dtype
                            norm_device = norm_param.device
                        except StopIteration:
                            norm_dtype = h.dtype
                            norm_device = h.device
                        # ``device_map=auto`` on very large models
                        # (qwen-27b, pythia-12b) places different
                        # blocks on different GPUs — move both dtype
                        # *and* device so the norm doesn't crash on a
                        # cross-device tensor.
                        h_normed = final_norm(h.to(norm_device).to(norm_dtype))
                    else:
                        # Final layer is already post-norm; leave as-is.
                        h_normed = h

                    try:
                        logits = apply_lm_head(model, h_normed)
                    except RuntimeError:
                        continue

                    # On device_map=auto models, ``preds`` lands on the
                    # lm-head's device while ``final_preds`` may be on
                    # another GPU (from the original forward). Move to
                    # the same device before comparing.
                    preds = logits.argmax(dim=-1)
                    fp = final_preds.to(preds.device) if preds.device != final_preds.device else final_preds
                    acc = (preds == fp).float().mean().item()
                    layer_accs[i].append(acc)

                    probs = F.softmax(logits, dim=-1)
                    log_probs = torch.log(probs.clamp(min=1e-12))
                    entropy = -(probs * log_probs).sum(dim=-1).mean().item()
                    layer_entropies[i].append(entropy)
                    
        results = {}
        for i in range(n_layers):
            if layer_accs[i]:
                results[f"layer{i}_acc"] = float(np.mean(layer_accs[i]))
                results[f"layer{i}_entropy"] = float(np.mean(layer_entropies[i]))
                
        return results
