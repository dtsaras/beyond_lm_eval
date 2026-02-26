import torch
import torch.nn.functional as F
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task

@register_task("consistency_contrastive")
class ContrastiveConsistencyTask(DiagnosticTask):
    """
    Measures Contrastive Consistency (Negative Rejection).
    Evaluates whether the model strongly rejects mutually exclusive 
    alternatives (B) when it assigns high probability to a factual baseline (A).
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running Contrastive Consistency Analysis...")
        num_samples = self.config.get("num_samples", 3)
        
        device = next(model.parameters()).device
        
        if dataset is None:
            try:
                from datasets import load_dataset
                # Load the NeelNanda/counterfact-tracing dataset and take the top num_samples
                dset = load_dataset("NeelNanda/counterfact-tracing", split="train")
                dataset = []
                for i in range(min(num_samples, len(dset))):
                    item = dset[i]
                    prompt = item["prompt"]
                    factual = prompt + item["target_true"]
                    exclusive = prompt + item["target_false"]
                    dataset.append({"factual": factual, "exclusive": exclusive})
            except ImportError:
                print("Warning: `datasets` library not found. Falling back to default examples.")
                dataset = [
                    {"factual": "The capital of France is Paris.", "exclusive": "The capital of France is London."},
                    {"factual": "Water boils at 100 degrees Celsius.", "exclusive": "Water boils at 0 degrees Celsius."},
                    {"factual": "A triangle has three sides.", "exclusive": "A triangle has four sides."},
                ][:num_samples]
                
        samples = list(dataset)[:num_samples]
        if len(samples) < 1:
             return {"error": "Need at least 1 sample with 'factual' and 'exclusive' keys"}
             
        if not all(k in samples[0] for k in ["factual", "exclusive"]):
             return {"error": "Dataset must contain 'factual' and 'exclusive' keys"}
             
        def get_sequence_prob(text):
            ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            if ids.shape[1] < 2:
                return 1e-10
            outputs = model(ids)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            shift_probs = probs[0, :-1, :]
            shift_labels = ids[0, 1:]
            token_probs = torch.gather(shift_probs, 1, shift_labels.unsqueeze(1)).squeeze(1)
            mean_log_prob = torch.log(token_probs + 1e-10).mean()
            return torch.exp(mean_log_prob).item()
            
        factual_probs = []
        exclusive_probs = []
        contrast_ratios = []
        
        with torch.no_grad():
            for s in samples:
                p_factual = get_sequence_prob(s["factual"])
                p_exclusive = get_sequence_prob(s["exclusive"])
                factual_probs.append(p_factual)
                exclusive_probs.append(p_exclusive)
                if p_factual > 0:
                    contrast_ratios.append(p_exclusive / p_factual)
                else:
                    contrast_ratios.append(1.0)
                    
        return {
            "mean_factual_prob": float(np.mean(factual_probs)),
            "mean_exclusive_prob": float(np.mean(exclusive_probs)),
            "mean_rejection_ratio": float(np.mean(contrast_ratios)),
        }
