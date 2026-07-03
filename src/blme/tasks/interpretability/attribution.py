from ...tasks.base import DiagnosticTask
from ...registry import register_task
import torch.nn.functional as F
import torch
import numpy as np
import logging
logger = logging.getLogger("blme")


def _gini_nonnegative(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    values = np.maximum(values, 0.0)
    total = values.sum()
    if total <= 0:
        return 0.0
    values = np.sort(values)
    n = values.size
    idx = np.arange(1, n + 1, dtype=float)
    gini = (2.0 * np.sum(idx * values) / (n * total)) - ((n + 1.0) / n)
    return float(np.clip(gini, 0.0, 1.0))


def _input_x_gradient_per_token(activation, grad):
    """Per-token input x gradient attribution (Simonyan et al. 2014; equal to
    captum ``InputXGradient`` reduced per token): ``|grad * activation|`` summed
    over the hidden dimension, dropping the last token (which carries no
    next-token cross-entropy term).

    ``activation``, ``grad``: ``(B, T, D)`` tensors. Returns a ``(B, T-1)``
    tensor. Extracted so the reference-parity test exercises BLME's real kernel.
    """
    token_attr = (grad * activation).abs().sum(dim=-1)
    return token_attr[:, :-1]


@register_task("interpretability_attribution")
class ComponentAttributionTask(DiagnosticTask):
    """
    Gradient-based input attribution for language-model predictions.

    For each sample, computes the next-token cross-entropy loss and attributes
    it to input tokens with |gradient × input-embedding activation|. This keeps
    the public attribution task name meaningful while avoiding residual-delta
    coherence proxies.

    Reference: Simonyan, Vedaldi & Zisserman 2014 saliency maps; BLME adapts
    input × gradient to language-model embedding activations.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Component Attribution Analysis...")
        num_samples = self.config.get("num_samples", 50)
        
        device = next(model.parameters()).device
        try:
            embedding_module = model.get_input_embeddings()
        except Exception:
            embedding_module = None
        if embedding_module is None:
            return {"error": "Input embeddings not found"}
        
        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)
        
        attribution_scores = []
        count = 0

        captured = {}

        def _capture_embedding_output(_module, _input_args, output):
            activation = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(activation) and activation.requires_grad:
                activation.retain_grad()
            captured["activation"] = activation

        handle = embedding_module.register_forward_hook(_capture_embedding_output)
        try:
            for sample in dataset:
                if count >= num_samples:
                    break

                if isinstance(sample, str):
                    inputs = tokenizer(sample, return_tensors="pt").to(device)
                elif isinstance(sample, dict) and 'text' in sample:
                    inputs = tokenizer(sample['text'], return_tensors="pt", truncation=True, max_length=128).to(device)
                elif isinstance(sample, dict) and 'input_ids' in sample:
                    inputs = {'input_ids': torch.tensor(sample['input_ids']).long().unsqueeze(0).to(device)}
                else:
                    continue

                input_ids = inputs.get("input_ids")
                if input_ids is None or input_ids.shape[-1] < 2:
                    continue

                captured.clear()
                model.zero_grad(set_to_none=True)

                outputs = model(**inputs)
                logits = getattr(outputs, "logits", outputs[0] if isinstance(outputs, tuple) else None)
                if logits is None or logits.shape[1] < 2:
                    continue

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous().to(shift_logits.device)
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                )
                loss.backward()

                activation = captured.get("activation")
                if not torch.is_tensor(activation) or activation.grad is None:
                    continue

                token_attr = _input_x_gradient_per_token(activation, activation.grad)
                token_attr = token_attr.detach().float().cpu().reshape(-1)
                attribution_scores.extend(token_attr.tolist())
                count += 1
        finally:
            handle.remove()
            model.zero_grad(set_to_none=True)

        if not attribution_scores:
            return {"error": "Failed to compute gradient attribution"}

        return {
            "mean_gradient_x_activation": float(np.mean(attribution_scores)),
            "std_gradient_x_activation": float(np.std(attribution_scores)),
            "max_gradient_x_activation": float(np.max(attribution_scores)),
            "attribution_gini": _gini_nonnegative(attribution_scores),
            "tokens_evaluated": int(len(attribution_scores)),
            "samples_evaluated": int(count),
        }
