"""
Betti Curve / Topological Complexity Trajectory — tracks how Betti numbers
change across ALL layers of the network.

The key insight from Naitzat et al. (ICLR 2020) is that well-generalized
networks progressively simplify topology: Betti numbers (connected components, 
loops) decrease with depth. The rate of this decrease and the final complexity
are both informative structural metrics.

References:
- "Topology of Deep Neural Networks" (Naitzat, Zhitnikov & Lim, ICLR 2020)
- "Topological Data Analysis of Large Language Models' Hidden Representations"
  (General TDA on Transformers literature)
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


def _count_betti(data, maxdim=1, threshold=None):
    """Count Betti numbers from a point cloud.
    
    Uses ripser to compute persistence and counts features alive at a 
    specified threshold (or uses a heuristic: median death value).
    
    Returns:
        (betti_0, betti_1): counts of connected components and loops
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ripser(data, maxdim=maxdim)
    
    dgms = result['dgms']
    
    # For H0: count connected components
    # Use a heuristic threshold: median of the finite death values
    h0 = dgms[0]
    finite_deaths_h0 = [d for _, d in h0 if d != np.inf]
    if threshold is None and finite_deaths_h0:
        threshold = np.median(finite_deaths_h0)
    elif threshold is None:
        threshold = 0.0
    
    # Count features alive at the threshold
    betti_0 = sum(1 for b, d in h0 if b <= threshold and (d > threshold or d == np.inf))
    
    # For H1: simply count finite loops
    h1 = dgms[1]
    betti_1 = len([1 for b, d in h1 if d != np.inf and (d - b) > 1e-6])
    
    return betti_0, betti_1


@register_task("topology_betti_curve")
class BettiCurveTask(DiagnosticTask):
    """
    Traces the Betti number trajectory (β0, β1) across all layers of the model.
    
    Following Naitzat et al. (ICLR 2020), tracks how topological complexity
    changes with depth. Reports:
    - β0 at each layer (connected components)
    - β1 at each layer (loops/holes)
    - Rate of topological simplification
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Betti Curve (Topological Complexity Trajectory)...")
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
                dataset = [{"text": f"Topological sample {i} for Betti curve calculation."} for i in range(num_samples)]
        samples = list(dataset)[:num_samples]
        if len(samples) < 3:
            return {"error": "Need at least 3 samples for Betti curve"}
        
        layers = get_layers(model)
        num_layers = len(layers)
        
        # Collect mean-pooled representations at EVERY layer
        layer_reps = {l: [] for l in range(num_layers)}
        
        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                out = model(**inputs, output_hidden_states=True)
                
                for l_idx in range(num_layers):
                    hidden = out.hidden_states[l_idx + 1][0]
                    rep = hidden.mean(dim=0).cpu().numpy()
                    layer_reps[l_idx].append(rep)
        
        betti_0_curve = []
        betti_1_curve = []
        
        for l_idx in range(num_layers):
            data = np.array(layer_reps[l_idx])
            b0, b1 = _count_betti(data, maxdim=1)
            betti_0_curve.append(b0)
            betti_1_curve.append(b1)
        
        results = {
            "betti_0_curve": betti_0_curve,
            "betti_1_curve": betti_1_curve,
            "betti_0_first": betti_0_curve[0],
            "betti_0_last": betti_0_curve[-1],
        }
        
        # Topological simplification ratio
        if betti_0_curve[0] > 0:
            results["simplification_ratio"] = float(betti_0_curve[-1] / betti_0_curve[0])
        else:
            results["simplification_ratio"] = 1.0
        
        # Rate of decay: linear regression slope of β0 vs layer index
        x = np.arange(num_layers)
        if len(betti_0_curve) > 1:
            slope = np.polyfit(x, betti_0_curve, 1)[0]
            results["betti_0_decay_rate"] = float(slope)
        else:
            results["betti_0_decay_rate"] = 0.0
            
        # Max β1 layer (most topological loops)
        results["max_betti_1_layer"] = int(np.argmax(betti_1_curve))
        results["max_betti_1"] = int(np.max(betti_1_curve))
            
        return results
