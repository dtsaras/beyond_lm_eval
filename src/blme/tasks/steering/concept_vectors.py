from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import apply_lm_head
import torch
import torch.nn.functional as F
import numpy as np


@register_task("steering_concept")
class ConceptSteeringTask(DiagnosticTask):
    """
    Computes a concept direction from positive/negative prompts and measures
    distribution shift after adding the direction to target hidden states.
    """

    def _last_hidden_and_probs(self, model, tokenizer, prompt, device):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        h = outputs.hidden_states[-1][0, -1].float()
        probs = F.softmax(outputs.logits[0, -1].float(), dim=-1)
        return h, probs

    def evaluate(self, model, tokenizer, dataset):
        print("Running Concept Steering Analysis...")
        method = self.config.get("method", "hidden_original")
        scales = self.config.get("scales", [0.5, 1.0, 2.0])
        target_prompts = self.config.get("target_prompts", ["The answer is"])
        positive_prompts = self.config.get("positive_prompts", ["good"])
        negative_prompts = self.config.get("negative_prompts", ["bad"])

        if not positive_prompts or not negative_prompts or not target_prompts:
            return {"error": "positive_prompts, negative_prompts, and target_prompts must be non-empty"}

        device = next(model.parameters()).device

        with torch.no_grad():
            pos_states = [self._last_hidden_and_probs(model, tokenizer, p, device)[0] for p in positive_prompts]
            neg_states = [self._last_hidden_and_probs(model, tokenizer, p, device)[0] for p in negative_prompts]

            steering = torch.stack(pos_states).mean(dim=0) - torch.stack(neg_states).mean(dim=0)
            if method == "hidden_normalized":
                steering = steering / (steering.norm() + 1e-10)

            shifts = {scale: [] for scale in scales}
            for prompt in target_prompts:
                h, base_probs = self._last_hidden_and_probs(model, tokenizer, prompt, device)
                for scale in scales:
                    h_edit = h + (float(scale) * steering)
                    logits_edit = apply_lm_head(model, h_edit.unsqueeze(0)).squeeze(0)
                    edit_probs = F.softmax(logits_edit, dim=-1)
                    kl = F.kl_div((edit_probs + 1e-10).log(), base_probs, reduction="sum").item()
                    shifts[scale].append(abs(kl))

        return {
            f"shift_scale_{scale}": float(np.mean(vals)) if vals else 0.0
            for scale, vals in shifts.items()
        }
