"""
Representation Sensitivity Task
──────────────────────────────────────────────────────────────────────
Evaluates how sensitive the model's output distribution is to perturbations
in the final-layer hidden states, by computing ||grad_h log P(y|h)||^2
averaged over tokens.

This measures the squared norm of the gradient of the log-likelihood with
respect to the representation — a form of local sensitivity or Jacobian
norm.  Higher values indicate the representation space is "sharp" (small
changes in h cause large changes in the prediction), while lower values
indicate a flatter, more robust manifold.

Since the LM head is linear (logits = W h + b), the gradient has a
closed-form:  grad_h log P(y|h) = W_y - sum_k p(k) W_k,  which avoids
per-token backward passes entirely.

References:
- "Information Geometry of Neural Networks" (Amari, 1998)
"""

import torch
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import apply_lm_head, get_lm_head, get_embeddings
import logging
logger = logging.getLogger("blme")


@register_task("geometry_representation_sensitivity")
class RepresentationSensitivityTask(DiagnosticTask):
    """
    Computes ||grad_h log P(y|h)||^2 with respect to final-layer hidden
    states using a closed-form derivation for the linear LM head.
    Uses the true next token as the target rather than argmax.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Representation Sensitivity Analysis...")
        num_samples = self.config.get("num_samples", 20)

        if dataset is None:
             dataset = [
                 {"text": "Information geometry studies probability distributions as a Riemannian manifold."}
             ] * num_samples

        samples = list(dataset)[:num_samples]
        if not samples:
             return {"error": "Need at least 1 sample."}

        device = next(model.parameters()).device

        # Get the unembedding matrix W (vocab_size, hidden_dim)
        head = get_lm_head(model)
        if head is not None:
            W = head.weight.detach().float()  # (V, D)
        else:
            W = get_embeddings(model)
            if W is None:
                return {"error": "Cannot access unembedding matrix."}
            W = W.detach().float()

        W = W.to(device)

        sensitivity_values = []

        model.eval()
        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                input_ids = inputs["input_ids"]

                if input_ids.shape[1] < 2:
                    continue

                out = model(**inputs, output_hidden_states=True)

                # Last hidden state: (1, seq_len, D)
                final_hidden = out.hidden_states[-1][0].float()  # (T, D)

                # True next tokens (shifted input_ids)
                targets = input_ids[0, 1:]  # (T-1,)
                h = final_hidden[:-1]       # (T-1, D) — hidden states predicting next token

                # Compute logits and probabilities
                logits = h @ W.T  # (T-1, V)
                # Add bias if present
                if head is not None and head.bias is not None:
                    logits = logits + head.bias.detach().float()
                probs = torch.softmax(logits, dim=-1)  # (T-1, V)

                # Closed-form gradient: grad_h log P(y|h) = W_y - sum_k p(k) W_k
                # W_y for each token: (T-1, D)
                W_target = W[targets]  # (T-1, D)
                # Expected W under model distribution: (T-1, D)
                W_expected = probs @ W  # (T-1, V) @ (V, D) -> (T-1, D)

                grad = W_target - W_expected  # (T-1, D)

                # Squared L2 norm per token
                grad_sq_norms = (grad ** 2).sum(dim=-1)  # (T-1,)
                sensitivity_values.append(grad_sq_norms.mean().item())

        if not sensitivity_values:
             return {"error": "Could not compute representation sensitivity."}

        return {
            "representation_sensitivity": float(np.mean(sensitivity_values)),
            "sensitivity_std": float(np.std(sensitivity_values)),
            "num_samples_analyzed": len(sensitivity_values)
        }
