import torch
import numpy as np
import warnings

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_layers
import logging
logger = logging.getLogger("blme")

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False


@register_task("topology_homology")
class PersistentHomologyTask(DiagnosticTask):
    """
    Implements a Topological Data Analysis (TDA) task.
    Takes a set of hidden representations and computes their persistent homology
    using the Vietoris-Rips complex, specifically extracting Betti-0 and Betti-1
    persistent features (holes/clusters in the manifold).
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Persistent Homology (TDA)...")
        num_samples = self.config.get("num_samples", 20)
        
        if not HAS_RIPSER:
            msg = "Ripser library not installed. Skipping TDA module. Install with: pip install ripser"
            logger.info(msg)
            return {"error": msg}
            
        device = next(model.parameters()).device
        
        if dataset is None:
            try:
                from datasets import load_dataset
                dset = load_dataset("EleutherAI/lambada_openai", "en", split="test")
                dataset = []
                for i in range(min(num_samples, len(dset))):
                    dataset.append({"text": dset[i]["text"]})
            except ImportError:
                logger.info("Warning: `datasets` library not found. Falling back to default examples.")
                dataset = [{"text": f"Random sample {i} for topological analysis of language models."} for i in range(num_samples)]
        samples = list(dataset)[:num_samples]
        if len(samples) < 3:
            return {"error": "Need at least 3 samples for meaningful topological features"}

        layers = get_layers(model)
        num_layers = len(layers)
        
        # We'll analyze the space at specific layers (early, middle, late)
        target_layers = [0, num_layers // 2, num_layers - 1]
        layer_representations = {l: [] for l in target_layers}
        
        with torch.no_grad():
            for s in samples:
                if isinstance(s, dict) and "text" in s:
                    text = s["text"]
                else:
                    text = str(s)
                    
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                out = model(**inputs, output_hidden_states=True)
                
                # Take the mean pooling over the sequence to represent the sentence
                for l_idx in target_layers:
                    # hidden_states includes embedding as index 0, so layer l_idx is l_idx + 1
                    hidden = out.hidden_states[l_idx + 1][0] # shape (seq_len, hidden_dim)
                    sentence_rep = hidden.mean(dim=0).cpu().numpy()
                    layer_representations[l_idx].append(sentence_rep)
                    
        results = {}
        for l_idx, data_points in layer_representations.items():
            data_matrix = np.array(data_points) # Shape: (num_samples, hidden_dim)
            
            # Compute Persistent Homology
            # maxdim=1 computes up to 1-dimensional holes (loops)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dgms = ripser(data_matrix, maxdim=1)['dgms']
                
            # dgms[0] contains 0-dimensional features (connected components)
            # dgms[1] contains 1-dimensional features (loops)
            
            # Feature lifespans
            # The lifespan is birth - death. For H0, first feature has infinite death.
            lifespans_h0 = []
            for birth, death in dgms[0]:
                if death != np.inf:
                    lifespans_h0.append(death - birth)
                    
            lifespans_h1 = []
            for birth, death in dgms[1]:
                 if death != np.inf:
                    lifespans_h1.append(death - birth)
                    
            # Describe the topology topology
            # Mean lifespan gives an idea of how "persistent" the structural features are.
            results[f"layer_{l_idx}_mean_persistance_h0"] = float(np.mean(lifespans_h0)) if lifespans_h0 else 0.0
            results[f"layer_{l_idx}_max_persistance_h0"] = float(np.max(lifespans_h0)) if lifespans_h0 else 0.0
            results[f"layer_{l_idx}_mean_persistance_h1"] = float(np.mean(lifespans_h1)) if lifespans_h1 else 0.0
            
            # The number of non-trivial loops
            results[f"layer_{l_idx}_num_loops_h1"] = len(lifespans_h1)
            
        return results
