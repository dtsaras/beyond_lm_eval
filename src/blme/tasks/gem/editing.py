from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings, apply_lm_head
import torch
import torch.nn.functional as F
import numpy as np


@register_task("gem_editing")
class MixtureEditingTask(DiagnosticTask):
    """
    Tests mixture editing: modifies the top-k weights of the hidden state mixture
    directly and observes the change in prediction.
    Operations: Drop top-1, Swap top-1/2, Boost top-1.
    GEM-specific: assumes hidden states have a mixture-of-embeddings interpretation.
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running Mixture Editing Analysis...")
        prompts = self.config.get("prompts", [
            "The number is",
            "The capital of France is",
            "2 + 2 =",
            "The color red is",
        ])

        device = next(model.parameters()).device
        E = get_embeddings(model)
        if E is None: return {"error": "Embeddings not found"}
        E = E.to(device)

        drop_kl = []
        swap_kl = []
        boost_kl = []

        k = 8

        with torch.no_grad():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors='pt').to(device)
                outputs = model(**inputs, output_hidden_states=True)

                h = outputs.hidden_states[-1][0, -1].float()
                logits = outputs.logits[0, -1]
                probs = F.softmax(logits, dim=-1)

                topk_probs, topk_idx = torch.topk(probs, k=k)

                topk_w = topk_probs / topk_probs.sum()
                topk_e = E[topk_idx]
                mu_orig = (topk_w.unsqueeze(-1) * topk_e).sum(dim=0)

                def predict(mu):
                    try:
                        l = apply_lm_head(model, mu.unsqueeze(0)).squeeze(0)
                    except RuntimeError:
                        l = mu.float() @ E.float().T
                    return F.softmax(l, dim=-1)

                p_orig = predict(mu_orig)

                # Drop top-1
                w_drop = topk_w.clone()
                w_drop[0] = 0
                w_drop = w_drop / (w_drop.sum() + 1e-10)
                mu_drop = (w_drop.unsqueeze(-1) * topk_e).sum(dim=0)
                p_drop = predict(mu_drop)
                kl = F.kl_div(p_drop.log(), p_orig, reduction='sum').item()
                drop_kl.append(abs(kl))

                # Swap top-1/2
                w_swap = topk_w.clone()
                w_swap[0], w_swap[1] = w_swap[1].item(), w_swap[0].item()
                mu_swap = (w_swap.unsqueeze(-1) * topk_e).sum(dim=0)
                p_swap = predict(mu_swap)
                kl = F.kl_div(p_swap.log(), p_orig, reduction='sum').item()
                swap_kl.append(abs(kl))

                # Boost top-1
                w_boost = topk_w.clone()
                w_boost[0] *= 2.0
                w_boost = w_boost / w_boost.sum()
                mu_boost = (w_boost.unsqueeze(-1) * topk_e).sum(dim=0)
                p_boost = predict(mu_boost)
                kl = F.kl_div(p_boost.log(), p_orig, reduction='sum').item()
                boost_kl.append(abs(kl))

        return {
            "edit_drop_kl": float(np.mean(drop_kl)),
            "edit_swap_kl": float(np.mean(swap_kl)),
            "edit_boost_kl": float(np.mean(boost_kl)),
        }
