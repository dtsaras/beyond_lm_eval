"""Persistent Homology on hidden-state point clouds.

Standard Topological Data Analysis pipeline: take a set of hidden
representations, build the Vietoris-Rips filtration, compute
persistence diagrams with ripser, and report the H₀ / H₁ lifespans.

References:
  * Zomorodian, Carlsson 2005 — "Computing Persistent Homology",
    *Discrete and Computational Geometry* 33(2).
  * Edelsbrunner, Harer 2008 — "Persistent Homology: a Survey".
  * Naitzat, Zhitnikov, Lim 2020 — "Topology of Deep Neural
    Networks", JMLR 21(184):1-40 (arXiv:2004.06093; motivation for
    applying TDA to learned representations).
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


def _scale_stratified_summary(dgms, scale: float, frac: float = 0.5) -> dict:
    """Scale-stratified persistence summaries (campaign-4 B8, after Fay et
    al., arXiv:2505.20435: adversarial/degenerate inputs COMPRESS latent
    topology — fewer small-scale features, amplified coarse ones).

    Bars are stratified by the scale at which they DIE relative to the
    point cloud's mean pairwise distance: fine = death < frac * scale,
    coarse = death >= frac * scale. Reported per homology dimension:
    fine/coarse counts and the fine fraction of total persistence mass.
    """
    out = {}
    if scale <= 0:
        return out
    thresh = frac * scale
    for k, name in ((0, "h0"), (1, "h1")):
        if len(dgms) <= k:
            continue
        bars = [(b, d) for b, d in dgms[k] if np.isfinite(d)]
        fine = [(b, d) for b, d in bars if d < thresh]
        coarse = [(b, d) for b, d in bars if d >= thresh]
        out[f"{name}_fine_count"] = len(fine)
        out[f"{name}_coarse_count"] = len(coarse)
        total_mass = sum(d - b for b, d in bars)
        fine_mass = sum(d - b for b, d in fine)
        out[f"{name}_fine_persistence_frac"] = (
            fine_mass / total_mass if total_mass > 0 else float("nan")
        )
    return out


def _lifespan_summary(dgms) -> dict:
    """Summarize ripser/GUDHI persistence diagrams into finite-bar lifespan
    statistics (lifespan = death - birth; the essential infinite-death H0
    bar is dropped). Matches the textbook persistence summary and is exact
    against an analytically-known diagram (e.g. unit square -> H0 mean/max=1,
    one H1 loop of persistence sqrt(2)-1).
    """
    h0 = [d - b for b, d in dgms[0] if d != np.inf]
    h1 = [d - b for b, d in dgms[1] if d != np.inf] if len(dgms) > 1 else []
    return {
        "mean_persistence_h0": float(np.mean(h0)) if h0 else 0.0,
        "max_persistence_h0": float(np.max(h0)) if h0 else 0.0,
        "mean_persistence_h1": float(np.mean(h1)) if h1 else 0.0,
        "num_loops_h1": len(h1),
    }


@register_task("topology_homology")
class PersistentHomologyTask(DiagnosticTask):
    """
    Implements a Topological Data Analysis (TDA) task.
    Takes a set of hidden representations and computes their persistent homology
    using the Vietoris-Rips complex, specifically extracting Betti-0 and Betti-1
    persistent features (holes/clusters in the manifold).

    See module docstring for paper references (Zomorodian-Carlsson
    2005, Edelsbrunner-Harer 2008, Naitzat-Zhitnikov-Lim 2020).
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
            except Exception as e:
                logger.info(
                    f"Warning: could not load lambada_openai ({type(e).__name__}: "
                    "missing `datasets`, offline, or download error); falling back "
                    "to default examples."
                )
                dataset = [{"text": f"Random sample {i} for topological analysis of language models."} for i in range(num_samples)]
        samples = list(dataset)[:num_samples]
        if len(samples) < 3:
            return {"error": "Need at least 3 samples for meaningful topological features"}

        layers = get_layers(model)
        num_layers = len(layers)
        
        # We'll analyze the space at specific layers (early, middle, late).
        # Dedupe so models with <=2 layers don't feed ripser the same layer
        # multiple times (which diluted the point cloud).
        target_layers = sorted({0, num_layers // 2, num_layers - 1})
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
                    # .float() first: numpy doesn't accept bf16 tensors, and
                    # ripser requires a standard numpy dtype.
                    hidden = out.hidden_states[l_idx + 1][0] # shape (seq_len, hidden_dim)
                    sentence_rep = hidden.mean(dim=0).float().cpu().numpy()
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

            # Summarize finite-bar lifespans (shared, testable helper).
            summary = _lifespan_summary(dgms)
            results[f"layer_{l_idx}_mean_persistence_h0"] = summary["mean_persistence_h0"]
            results[f"layer_{l_idx}_max_persistence_h0"] = summary["max_persistence_h0"]
            results[f"layer_{l_idx}_mean_persistence_h1"] = summary["mean_persistence_h1"]
            results[f"layer_{l_idx}_num_loops_h1"] = summary["num_loops_h1"]

            # Scale-stratified summaries (B8): fine vs coarse features
            # relative to the mean pairwise distance (computed below).
            # (Inserted before the normalization block so both share mpd.)
            # -- see _scale_stratified_summary
            # Scale-normalized persistence: raw lifespans scale with the
            # hidden-state norms, which differ across models. Dividing by the
            # mean pairwise distance of the layer's point cloud makes the
            # summaries comparable across models (the normalization stated in
            # the paper's cross-model framework).
            from scipy.spatial.distance import pdist
            pairwise = pdist(data_matrix)
            mpd = float(pairwise.mean()) if pairwise.size else 0.0
            if mpd > 0:
                for key in ("mean_persistence_h0", "max_persistence_h0",
                            "mean_persistence_h1"):
                    results[f"layer_{l_idx}_{key}_norm"] = summary[key] / mpd
                results[f"layer_{l_idx}_mean_pairwise_distance"] = mpd
                for k, v in _scale_stratified_summary(dgms, mpd).items():
                    results[f"layer_{l_idx}_{k}"] = v

        return results
