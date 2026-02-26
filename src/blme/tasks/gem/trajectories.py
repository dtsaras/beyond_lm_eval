from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings, apply_lm_head
import torch
import torch.nn.functional as F
import numpy as np


@register_task("gem_trajectories")
class MixtureTrajectoriesTask(DiagnosticTask):
    """
    Tracks how hidden states move as context disambiguates.
    Measures containment within the convex hull of top-k tokens and trajectory smoothness.
    GEM-specific: interprets hidden states as mixtures of token embeddings.
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running Mixture Trajectories Analysis...")
        prompts = self.config.get("prompts", None)

        if prompts is None:
            prompts = [
                ("The bank", [" was", " river", " loan", " account"]),
                ("I saw a bat", [" flying", " wooden", " cave", " baseball"]),
                ("The spring", [" water", " season", " coil", " bounce"]),
                ("She felt", [" happy", " sad", " the", " cold"]),
                ("The model", [" walked", " predicted", " car", " fashion"]),
            ]

        device = next(model.parameters()).device
        E = get_embeddings(model)
        if E is None: return {"error": "Embeddings not found"}
        E = E.to(device)

        trajectory_containment = []
        trajectory_smoothness = []

        with torch.no_grad():
            for base_int, continuations in prompts:
                base_prompt = base_int

                base_inputs = tokenizer(base_prompt, return_tensors='pt').to(device)
                base_outputs = model(**base_inputs, output_hidden_states=True)

                h_base = base_outputs.hidden_states[-1][0, -1].float()
                logits_base = base_outputs.logits[0, -1].float()
                probs_base = F.softmax(logits_base, dim=-1)

                k = 16
                topk_probs, topk_idx = torch.topk(probs_base, k=k)
                topk_embs = E[topk_idx].float()
                centroid = (topk_probs.unsqueeze(-1) * topk_embs).sum(dim=0)

                for cont in continuations:
                    full_prompt = base_prompt + cont
                    full_inputs = tokenizer(full_prompt, return_tensors='pt').to(device)
                    full_outputs = model(**full_inputs, output_hidden_states=True)
                    h_new = full_outputs.hidden_states[-1][0, -1].float()

                    delta_h = h_new - h_base

                    topk_norm = topk_embs / (topk_embs.norm(dim=1, keepdim=True) + 1e-10)
                    delta_norm = delta_h / (delta_h.norm() + 1e-10)
                    alignments = (topk_norm @ delta_norm).abs()
                    trajectory_containment.append(alignments.max().item())

                    dist_to_centroid = (h_new - centroid).norm().item()
                    base_dist = (h_base - centroid).norm().item()
                    smoothness = dist_to_centroid / (base_dist + 1e-10)
                    trajectory_smoothness.append(smoothness)

        return {
            "trajectory_containment_mean": float(np.mean(trajectory_containment)),
            "trajectory_smoothness_mean": float(np.mean(trajectory_smoothness))
        }
