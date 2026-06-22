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
    Measures a last-token representation-distance proxy for paraphrase pairs.
    Evaluates how much the representation distance changes between supplied
    paraphrases compared to supplied unrelated sentences.

    Caveat: This metric can be gamed via superficial pattern matching
    (e.g., lexical overlap). Results are most meaningful with diverse
    paraphrases that share semantics but differ substantially in surface form.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Paraphrase Invariance...")
        num_samples = self.config.get("num_samples", 5)
        
        device = next(model.parameters()).device
        
        _BUNDLED = [
            {"text1": "The quick brown fox jumps over the lazy dog.",
             "text2": "A fast, dark-coloured fox leaps above a sleepy hound.",
             "unrelated": "Machine learning is transforming data processing."},
            {"text1": "Water boils at 100 degrees Celsius.",
             "text2": "The boiling point of H2O is one hundred degrees Celsius.",
             "unrelated": "The Eiffel Tower is located in Paris."},
            {"text1": "She quickly finished her homework before dinner.",
             "text2": "Before eating dinner she had already completed her schoolwork.",
             "unrelated": "The Pacific Ocean is the largest ocean on Earth."},
            {"text1": "The stock market dropped sharply on Friday.",
             "text2": "On Friday the equity markets took a steep plunge.",
             "unrelated": "Trees provide oxygen through photosynthesis."},
            {"text1": "He couldn't find his keys anywhere.",
             "text2": "His keys were nowhere to be found.",
             "unrelated": "Mount Everest stands in the Himalayan range."},
        ]

        # Only accept dataset entries that actually carry the required
        # (text1, text2, unrelated) triple. Generic BLME pipeline
        # corpora are {"text": ...} and would otherwise drop straight
        # through to the error path.
        usable = []
        if dataset is not None and isinstance(dataset, list):
            for item in dataset[:num_samples]:
                if isinstance(item, dict) and {"text1", "text2", "unrelated"} <= set(item):
                    usable.append(item)

        if len(usable) < 1:
            try:
                from datasets import load_dataset
                dset = load_dataset("coastalcph/mpararel", "en", split="train")
                from collections import defaultdict
                grouped = defaultdict(list)
                for item in dset:
                    grouped[item["relation_id"]].append(item["text"])
                relations = list(grouped.keys())
                num_rel = len(relations)
                usable = []
                for i in range(min(num_samples, 20)):
                    rel = relations[i % num_rel]
                    other_rel = relations[(i + 1) % num_rel]
                    if len(grouped[rel]) >= 2 and len(grouped[other_rel]) >= 1:
                        usable.append({
                            "text1": grouped[rel][0],
                            "text2": grouped[rel][1],
                            "unrelated": grouped[other_rel][0],
                        })
            except Exception as e:
                logger.info(
                    f"Warning: mpararel unavailable ({type(e).__name__}); "
                    "using bundled triples."
                )
                usable = _BUNDLED[:num_samples]

        samples = list(usable)[:num_samples]
        if len(samples) < 1:
            return {"error": "Need at least 1 (text1, text2, unrelated) triple"}

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
                
                # Last-token hidden state (causal LM "sentence embedding").
                # Historic code did a mean-pool over every position
                # including BOS; for decoder-only models with a strong
                # BOS attractor (Llama / Gemma) the BOS state dominated
                # the mean and the paraphrase signal collapsed.
                out1 = model(**inputs1, output_hidden_states=True)
                out2 = model(**inputs2, output_hidden_states=True)
                out3 = model(**inputs3, output_hidden_states=True)

                rep1 = out1.hidden_states[-1][0, -1].float()
                rep2 = out2.hidden_states[-1][0, -1].float()
                rep3 = out3.hidden_states[-1][0, -1].float()
                
                # Distances (L2)
                paraphrase_distances.append(torch.norm(rep1 - rep2, p=2).item())
                unrelated_distances.append(torch.norm(rep1 - rep3, p=2).item())
                
                # Cosine Similarities
                paraphrase_cos_sims.append(F.cosine_similarity(rep1.unsqueeze(0), rep2.unsqueeze(0)).item())
                unrelated_cos_sims.append(F.cosine_similarity(rep1.unsqueeze(0), rep3.unsqueeze(0)).item())
                
        mean_para_l2 = float(np.mean(paraphrase_distances))
        mean_unrelated_l2 = float(np.mean(unrelated_distances))
        mean_para_cos = float(np.mean(paraphrase_cos_sims))
        mean_unrelated_cos = float(np.mean(unrelated_cos_sims))

        results = {
            "diagnostic_semantics": "last_token_representation_distance_proxy",
            "representation_paraphrase_l2_dist": mean_para_l2,
            "representation_unrelated_l2_dist": mean_unrelated_l2,
            "representation_paraphrase_cos_sim": mean_para_cos,
            "representation_unrelated_cos_sim": mean_unrelated_cos,
            # Legacy aliases retained for downstream compatibility.
            "mean_paraphrase_l2_dist": mean_para_l2,
            "mean_unrelated_l2_dist": mean_unrelated_l2,
            "mean_paraphrase_cos_sim": mean_para_cos,
            "mean_unrelated_cos_sim": mean_unrelated_cos,
        }
        
        # Ratio of distances: lower means supplied paraphrases are closer
        # than supplied unrelated examples in this representation space.
        if mean_unrelated_l2 > 0:
            ratio = mean_para_l2 / mean_unrelated_l2
            results["representation_distance_ratio_l2"] = ratio
            results["isometry_ratio_l2"] = ratio
            
        return results
