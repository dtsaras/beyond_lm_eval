import torch
import torch.nn.functional as F
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task
import logging
logger = logging.getLogger("blme")

@register_task("consistency_logical")
class LogicalConsistencyTask(DiagnosticTask):
    """
    Measures Logical Consistency (A implies B).
    Evaluates whether the model assigns consistent probabilities to logically 
    implicated statements. If a model assigns high probability to a premise (A), 
    it should assign high probability to its necessary conclusion (B).
    
    References:
    - "Measuring and Improving Consistency in Pretrained Language Models" (Elazar et al., 2021)
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Logical Consistency Analysis...")
        num_samples = self.config.get("num_samples", 3)
        
        device = next(model.parameters()).device
        
        if dataset is None:
            try:
                from datasets import load_dataset
                dset = load_dataset("EleutherAI/lambada_openai", "en", split="test")
                dataset = []
                for i in range(min(num_samples, len(dset))):
                    # For logical consistency, we need a 'premise' and 'conclusion'.
                    # LAMBADA is just text. We'll simply use the first half as premise and second half as conclusion to keep the format.
                    # Note: this is a dummy setup just to feed data into the evaluation pipeline
                    text = dset[i]["text"]
                    mid = len(text) // 2
                    dataset.append({"premise": text[:mid], "conclusion": text[mid:]})
            except ImportError:
                logger.info("Warning: `datasets` library not found. Falling back to default examples.")
                dataset = [
                    {"premise": "John is a bachelor.", "conclusion": "John is unmarried."},
                    {"premise": "The car is completely destroyed.", "conclusion": "The car cannot be driven."},
                    {"premise": "Paris is the capital of France.", "conclusion": "Paris is in France."}
                ][:num_samples]
        samples = list(dataset)[:num_samples]
        if len(samples) < 1:
             return {"error": "Need at least 1 sample with 'premise' and 'conclusion' keys"}
             
        if not all(k in samples[0] for k in ["premise", "conclusion"]):
             return {"error": "Dataset must contain 'premise' and 'conclusion' keys"}
             
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
            
        premise_probs = []
        conclusion_probs = []
        violations = 0
        
        with torch.no_grad():
            for s in samples:
                p_A = get_sequence_prob(s["premise"])
                p_B = get_sequence_prob(s["conclusion"])
                premise_probs.append(p_A)
                conclusion_probs.append(p_B)
                margin_threshold = self.config.get("margin", 0.1)
                if p_A > (p_B + margin_threshold):
                    violations += 1
                    
        return {
            "mean_premise_prob": float(np.mean(premise_probs)),
            "mean_conclusion_prob": float(np.mean(conclusion_probs)),
            "logical_violation_rate": float(violations / len(samples)),
        }
