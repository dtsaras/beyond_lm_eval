"""
Persistence Entropy — Shannon entropy of the persistence diagram lifespans.

Quantifies the "complexity" or "disorder" of the topological features found
in the representation space. High entropy indicates many features with similar
lifetimes (complex, disordered topology), while low entropy indicates a few
dominant persistent features (simple, ordered topology).

References:
- "A New Topological Entropy-based Approach for Measuring Similarities Among
  Piecewise Linear Functions" (Rucco, Castiglione, Merelli et al., 2016)
- "On the Stability of Persistent Entropy and New Summary Functions for TDA"
  (Atienza et al., Entropy 2019)
"""

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


def _persistence_entropy(lifespans):
    """Compute Shannon entropy of normalized persistence lifespans.
    
    PE = -Σ p_i · log(p_i)   where p_i = lifespan_i / Σ lifespans
    
    Args:
        lifespans: list of positive floats (birth-death intervals)
        
    Returns:
        float: persistence entropy (0 = single dominant feature, high = uniform)
    """
    if not lifespans or len(lifespans) < 2:
        return 0.0
        
    arr = np.array(lifespans)
    total = np.sum(arr)
    if total < 1e-12:
        return 0.0
        
    p = arr / total
    p = p[p > 1e-12]  # filter near-zero for log stability
    return float(-np.sum(p * np.log(p)))


@register_task("topology_persistence_entropy")
class PersistenceEntropyTask(DiagnosticTask):
    """
    Computes Persistence Entropy at early, middle, and late layers.
    
    For each layer, constructs a persistence diagram via Vietoris-Rips
    and computes the Shannon entropy of the H0 and H1 lifespan distributions.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Persistence Entropy Analysis...")
        num_samples = self.config.get("num_samples", 20)
        
        if not HAS_RIPSER:
            msg = "Ripser library not installed. Install with: pip install ripser"
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
                dataset = [{"text": f"Topological sample {i} for persistence entropy calculation."} for i in range(num_samples)]
        samples = list(dataset)[:num_samples]
        if len(samples) < 3:
            return {"error": "Need at least 3 samples for persistence entropy"}
        
        layers = get_layers(model)
        num_layers = len(layers)
        target_layers = [0, num_layers // 2, num_layers - 1]
        layer_reps = {l: [] for l in target_layers}
        
        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                out = model(**inputs, output_hidden_states=True)
                
                for l_idx in target_layers:
                    hidden = out.hidden_states[l_idx + 1][0]
                    rep = hidden.mean(dim=0).cpu().numpy()
                    layer_reps[l_idx].append(rep)
        
        results = {}
        for l_idx, points in layer_reps.items():
            data = np.array(points)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dgms = ripser(data, maxdim=1)['dgms']
            
            # H0 lifespans (excluding infinite-lifespan component)
            lifespans_h0 = [d - b for b, d in dgms[0] if d != np.inf]
            # H1 lifespans
            lifespans_h1 = [d - b for b, d in dgms[1] if d != np.inf]

            pe_h0 = _persistence_entropy(lifespans_h0)
            pe_h1 = _persistence_entropy(lifespans_h1)

            # pe_h0_full: approximate infinite component with 2x max finite lifespan
            finite_h0 = [d - b for b, d in dgms[0] if d != np.inf]
            inf_births = [b for b, d in dgms[0] if d == np.inf]
            if finite_h0 and inf_births:
                max_finite = max(finite_h0)
                approx_inf = max_finite * 2.0
                lifespans_h0_full = finite_h0 + [approx_inf] * len(inf_births)
                pe_h0_full = _persistence_entropy(lifespans_h0_full)
            else:
                pe_h0_full = pe_h0

            results[f"layer_{l_idx}_pe_h0"] = pe_h0
            results[f"layer_{l_idx}_pe_h0_full"] = pe_h0_full
            results[f"layer_{l_idx}_pe_h1"] = pe_h1
            results[f"layer_{l_idx}_num_features_h0"] = len(lifespans_h0)
            results[f"layer_{l_idx}_num_features_h1"] = len(lifespans_h1)
        
        # Summary: how much does topological entropy change across layers?
        pe_values = [results[f"layer_{l}_pe_h0"] for l in target_layers]
        if pe_values[0] > 0:
            results["pe_simplification_ratio"] = float(pe_values[-1] / pe_values[0])
        else:
            results["pe_simplification_ratio"] = 1.0
            
        return results
