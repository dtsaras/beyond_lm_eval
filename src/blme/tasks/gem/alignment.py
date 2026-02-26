from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings, apply_lm_head
import torch
import torch.nn.functional as F
import numpy as np


@register_task("gem_alignment")
class AlignmentResidualTask(DiagnosticTask):
    """
    Measures how well hidden states align with the convex hull of top-k token embeddings.
    For each position, computes the distance between h and its nearest mixture
    representation mu = sum(w_i * e_i) using top-k probability weights.
    GEM-specific: validates the hypothesis that h ≈ mixture of embeddings.
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running GEM Alignment Residual Analysis...")
        num_samples = self.config.get("num_samples", 10)
        k_values = self.config.get("k_values", [8, 16, 32])
        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(max(1, num_samples))]

        device = next(model.parameters()).device
        E = get_embeddings(model)
        if E is None: return {"error": "Embeddings not found"}
        E = E.to(device).float()

        results_by_k = {}

        for k in k_values:
            cosine_sims = []
            l2_dists = []

            with torch.no_grad():
                for sample in dataset[:num_samples]:
                    text = sample["text"]
                    inputs = tokenizer(text, return_tensors='pt').to(device)
                    outputs = model(**inputs, output_hidden_states=True)

                    h = outputs.hidden_states[-1][0].float()  # [seq_len, dim]
                    logits = outputs.logits[0].float()  # [seq_len, vocab]
                    probs = F.softmax(logits, dim=-1)

                    for pos in range(h.size(0)):
                        topk_probs, topk_idx = torch.topk(probs[pos], k=k)
                        topk_w = topk_probs / topk_probs.sum()
                        topk_embs = E[topk_idx]

                        mu = (topk_w.unsqueeze(-1) * topk_embs).sum(dim=0)

                        cos = F.cosine_similarity(h[pos].unsqueeze(0), mu.unsqueeze(0)).item()
                        cosine_sims.append(cos)

                        l2 = (h[pos] - mu).norm().item()
                        l2_dists.append(l2)

            results_by_k[k] = {
                "cosine_sim_mean": float(np.mean(cosine_sims)),
                "l2_dist_mean": float(np.mean(l2_dists)),
            }

        flat = {}
        for k, v in results_by_k.items():
            for metric, val in v.items():
                flat[f"alignment_k{k}_{metric}"] = val

        return flat


@register_task("gem_substitution")
class SubstitutionConsistencyTask(DiagnosticTask):
    """
    Tests if replacing hidden state h with its mixture approximation mu
    preserves the model's prediction distribution.
    Measures KL-divergence between p(h) and p(mu).
    GEM-specific: validates that the mixture representation is functionally equivalent.
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running GEM Substitution Consistency Analysis...")
        num_samples = self.config.get("num_samples", 10)
        k = self.config.get("k", 16)
        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(max(1, num_samples))]

        device = next(model.parameters()).device
        E = get_embeddings(model)
        if E is None: return {"error": "Embeddings not found"}
        E = E.to(device).float()

        kl_divs = []
        top1_matches = []

        with torch.no_grad():
            for sample in dataset[:num_samples]:
                text = sample["text"]
                inputs = tokenizer(text, return_tensors='pt').to(device)
                outputs = model(**inputs, output_hidden_states=True)

                h = outputs.hidden_states[-1][0].float()
                logits = outputs.logits[0].float()
                probs = F.softmax(logits, dim=-1)

                for pos in range(h.size(0)):
                    p_h = probs[pos]

                    topk_probs, topk_idx = torch.topk(p_h, k=k)
                    topk_w = topk_probs / topk_probs.sum()
                    topk_embs = E[topk_idx]
                    mu = (topk_w.unsqueeze(-1) * topk_embs).sum(dim=0)

                    try:
                        logits_mu = apply_lm_head(model, mu.unsqueeze(0)).squeeze(0)
                    except RuntimeError:
                        logits_mu = mu @ E.T

                    p_mu = F.softmax(logits_mu, dim=-1)

                    kl = F.kl_div(
                        (p_mu + 1e-10).log(), p_h,
                        reduction='sum'
                    ).item()
                    kl_divs.append(abs(kl))

                    top1_match = (p_h.argmax() == p_mu.argmax()).item()
                    top1_matches.append(top1_match)

        return {
            "substitution_kl_mean": float(np.mean(kl_divs)),
            "substitution_top1_match": float(np.mean(top1_matches)),
            "substitution_top1_agreement": float(np.mean(top1_matches)),
        }
