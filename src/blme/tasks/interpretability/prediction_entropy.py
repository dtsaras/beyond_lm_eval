from ...tasks.base import DiagnosticTask
from ...registry import register_task
import torch
import numpy as np
from tqdm import tqdm


@register_task("interpretability_prediction_entropy")
class PredictionEntropyTask(DiagnosticTask):
    """
    Computes the entropy of the output probability distribution at each
    token position, profiling the model's inherent uncertainty.
    Ref: Holtzman et al., "The Curious Case of Neural Text Degeneration",
         ICLR 2020. arXiv:1904.09751
    """

    def evaluate(self, model, tokenizer, dataset):
        print("Running Prediction Entropy Analysis...")

        if dataset is None:
            dataset = [
                {"text": "The quick brown fox jumps over the lazy dog."}
                for _ in range(50)
            ]

        num_samples = self.config.get("num_samples", 100)
        all_entropies = []
        all_top1_probs = []
        all_top5_probs = []

        with torch.no_grad():
            for i, sample in enumerate(tqdm(dataset, desc="Computing Entropy")):
                if i >= num_samples:
                    break

                text = sample.get("text", "") if isinstance(sample, dict) else sample
                inputs = tokenizer(text, return_tensors="pt").to(model.device)

                outputs = model(**inputs)
                logits = outputs.logits  # (B, T, V)

                # Compute probabilities
                probs = torch.softmax(logits, dim=-1)  # (B, T, V)

                # Entropy per token: H = -sum(p * log(p))
                log_probs = torch.log(probs + 1e-12)
                entropy = -torch.sum(probs * log_probs, dim=-1)  # (B, T)
                all_entropies.extend(entropy[0].cpu().tolist())

                # Top-1 probability (confidence)
                top1 = probs.max(dim=-1).values  # (B, T)
                all_top1_probs.extend(top1[0].cpu().tolist())

                # Top-5 cumulative probability
                top5 = torch.topk(probs, k=min(5, probs.shape[-1]), dim=-1).values
                top5_sum = top5.sum(dim=-1)  # (B, T)
                all_top5_probs.extend(top5_sum[0].cpu().tolist())

        ent_arr = np.array(all_entropies)
        top1_arr = np.array(all_top1_probs)
        top5_arr = np.array(all_top5_probs)

        return {
            "mean_entropy": float(np.mean(ent_arr)),
            "std_entropy": float(np.std(ent_arr)),
            "median_entropy": float(np.median(ent_arr)),
            "p90_entropy": float(np.percentile(ent_arr, 90)),
            "mean_top1_prob": float(np.mean(top1_arr)),
            "mean_top5_prob": float(np.mean(top5_arr)),
        }
