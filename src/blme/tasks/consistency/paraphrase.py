import torch
import torch.nn.functional as F
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task
import logging
logger = logging.getLogger("blme")

@register_task("consistency_paraphrase")
class ParaphraseInvarianceTask(DiagnosticTask):
    """
    Measures Paraphrase Invariance (Semantic Isometry).
    Evaluates how much the representation distance changes between sentences
    that mean the exact same thing but are syntactically different, compared
    to completely unrelated sentences.

    Caveat: This metric can be gamed via superficial pattern matching
    (e.g., lexical overlap). Results are most meaningful with diverse
    paraphrases that share semantics but differ substantially in surface form.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Paraphrase Invariance...")
        num_samples = self.config.get("num_samples", 5)
        
        device = next(model.parameters()).device
        
        # We need paired paraphrases (pos) and unrelated (neg) examples
        if dataset is None:
            try:
                from datasets import load_dataset
                dset = load_dataset("coastalcph/mpararel", "en", split="train")
                dataset = []
                
                # ParaRel provides multiple valid templates (pattern) for the same relation and subject-object pair.
                # We group them to create valid text1 to text2 (same meaning) vs text3 (different relation) pairs.
                # We'll just grab nearby samples for unrelated.
                from collections import defaultdict
                import random
                
                grouped = defaultdict(list)
                for item in dset:
                    grouped[item["relation_id"]].append(item["text"])
                    
                relations = list(grouped.keys())
                num_rel = len(relations)
                
                for i in range(min(num_samples, 20)):
                    rel = relations[i % num_rel]
                    other_rel = relations[(i + 1) % num_rel]
                    
                    if len(grouped[rel]) >= 2 and len(grouped[other_rel]) >= 1:
                        text1 = grouped[rel][0]
                        text2 = grouped[rel][1]
                        unrelated = grouped[other_rel][0]
                        dataset.append({"text1": text1, "text2": text2, "unrelated": unrelated})
            except ImportError:
                logger.info("Warning: `datasets` library not found. Falling back to default examples.")
                dataset = [
                    {
                        "text1": "The quick brown fox jumps over the lazy dog.",
                        "text2": "A fast, dark-colored fox leaps above a sleepy hound.",
                        "unrelated": "Machine learning is transforming data processing."
                    },
                    {
                        "text1": "Water boils at 100 degrees Celsius.",
                        "text2": "The boiling point of H2O is one hundred degrees Celsius.",
                        "unrelated": "The Eiffel Tower is located in Paris."
                    },
                ][:num_samples]
            
        samples = list(dataset)[:num_samples]
        if len(samples) < 1:
             return {"error": "Need at least 1 sample with 'text1', 'text2', and 'unrelated' keys"}
             
        if not all(k in samples[0] for k in ["text1", "text2", "unrelated"]):
             return {"error": "Dataset must contain 'text1', 'text2', and 'unrelated' keys"}

        paraphrase_distances = []
        unrelated_distances = []
        paraphrase_cos_sims = []
        unrelated_cos_sims = []

        with torch.no_grad():
            for s in samples:
                # Tokenize all three
                inputs1 = tokenizer(s["text1"], return_tensors="pt", truncation=True, max_length=128).to(device)
                inputs2 = tokenizer(s["text2"], return_tensors="pt", truncation=True, max_length=128).to(device)
                inputs3 = tokenizer(s["unrelated"], return_tensors="pt", truncation=True, max_length=128).to(device)
                
                # Get reps (using mean pooling over the sequence for sentence representation)
                # We use the final layer hidden states
                out1 = model(**inputs1, output_hidden_states=True)
                out2 = model(**inputs2, output_hidden_states=True)
                out3 = model(**inputs3, output_hidden_states=True)
                
                rep1 = out1.hidden_states[-1][0].mean(dim=0)
                rep2 = out2.hidden_states[-1][0].mean(dim=0)
                rep3 = out3.hidden_states[-1][0].mean(dim=0)
                
                # Distances (L2)
                paraphrase_distances.append(torch.norm(rep1 - rep2, p=2).item())
                unrelated_distances.append(torch.norm(rep1 - rep3, p=2).item())
                
                # Cosine Similarities
                paraphrase_cos_sims.append(F.cosine_similarity(rep1.unsqueeze(0), rep2.unsqueeze(0)).item())
                unrelated_cos_sims.append(F.cosine_similarity(rep1.unsqueeze(0), rep3.unsqueeze(0)).item())
                
        results = {
            "mean_paraphrase_l2_dist": float(np.mean(paraphrase_distances)),
            "mean_unrelated_l2_dist": float(np.mean(unrelated_distances)),
            "mean_paraphrase_cos_sim": float(np.mean(paraphrase_cos_sims)),
            "mean_unrelated_cos_sim": float(np.mean(unrelated_cos_sims)),
        }
        
        # Ratio of distances: Lower is better (paraphrases are closer relative to unrelated)
        if results["mean_unrelated_l2_dist"] > 0:
            results["isometry_ratio_l2"] = results["mean_paraphrase_l2_dist"] / results["mean_unrelated_l2_dist"]
            
        return results
