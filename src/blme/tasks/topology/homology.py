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
    """Scale-stratified persistence summaries (a BLME construct).

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

        # Early, middle and late layers, deduplicated for models with <= 2 layers.
        target_layers = sorted({0, num_layers // 2, num_layers - 1})
        layer_representations = {l: [] for l in target_layers}

        # post-final-norm: SUBSTITUTED -- the deepest cloud is the raw last block, as in the sibling
        # topology tasks (HF's last entry is ln_f(block L-1)).
        _sink = {}
        _hook = None
        _n_fwd = _n_sub = 0
        try:
            _blocks = layers
            if _blocks is not None and len(_blocks) > 0:
                _hook = _blocks[-1].register_forward_hook(
                    lambda mod, inp, outp: _sink.__setitem__(
                        "h", (outp[0] if isinstance(outp, tuple)
                              else outp).detach()))
        except Exception as e:  # pragma: no cover - defensive
            logger.info(f"homology: could not hook last block ({e})")

        # The hook is removed in `finally` (tests/tasks/test_forward_hook_leaks.py).
        try:
            with torch.no_grad():
                for s in samples:
                    if isinstance(s, dict) and "text" in s:
                        text = s["text"]
                    else:
                        text = str(s)

                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                    _sink.pop("h", None)
                    out = model(**inputs, output_hidden_states=True)
                    hs = list(out.hidden_states)
                    _n_fwd += 1
                    _h_raw = _sink.get("h")
                    if (_h_raw is not None and len(hs) >= 2
                            and _h_raw.shape == hs[-1].shape):
                        hs[-1] = _h_raw
                        _n_sub += 1

                    # One mean-pooled vector per document, pooled in float64.
                    for l_idx in target_layers:
                        hidden = hs[l_idx + 1][0]  # hidden_states[0] is the embedding
                        sentence_rep = hidden.double().mean(dim=0).cpu().numpy()
                        layer_representations[l_idx].append(sentence_rep)

        finally:
            if _hook is not None:
                _hook.remove()

        results = {}
        results["_meta_last_entry_is_block_output"] = bool(
            _n_fwd > 0 and _n_sub == _n_fwd)
        # Depth-role names (first/mid/last); absolute-index names alias
        # different depths across models and are _meta_.
        _role = {}
        _sorted_targets = sorted(layer_representations.keys())
        if len(_sorted_targets) == 3:
            _role = dict(zip(_sorted_targets, ("first", "mid", "last")))
        elif len(_sorted_targets) == 2:
            _role = dict(zip(_sorted_targets, ("first", "last")))
        elif len(_sorted_targets) == 1:
            _role = {_sorted_targets[0]: "first"}

        for l_idx, data_points in layer_representations.items():
            _r = _role.get(l_idx)
            data_matrix = np.array(data_points)  # Shape: (num_samples, hidden_dim)

            # Duplicate points are quotiented (a repeated point is a zero-length
            # H0 bar), then the cloud is centred and divided by its max entry so
            # ripser's float32 neither under- nor overflows; lifetimes are scaled
            # back afterwards. `data_matrix` itself is kept for the mpd below.
            _centred = np.asarray(data_matrix, dtype=np.float64)
            _centred = np.unique(_centred, axis=0)
            _centred = _centred - _centred.mean(axis=0, keepdims=True)
            _hg = float(np.max(np.abs(_centred))) if _centred.size else 0.0
            if np.isfinite(_hg) and _hg > 0:
                _rips_input = _centred / _hg
            else:
                _hg = 1.0
                _rips_input = _centred

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dgms = ripser(_rips_input, maxdim=1)['dgms']
            if _hg != 1.0:
                dgms = [d * _hg for d in dgms]

            summary = _lifespan_summary(dgms)
            for _k in ("mean_persistence_h0", "max_persistence_h0",
                       "mean_persistence_h1", "num_loops_h1"):
                results[f"_meta_layer_{l_idx}_{_k}"] = summary[_k]
                if _r:
                    results[f"{_r}_layer_{_k}"] = summary[_k]

            # Lifetimes divided by the cloud's mean pairwise distance (mpd):
            # VR lifetimes and the mpd are both homogeneous of degree 1.
            from scipy.spatial.distance import pdist
            pairwise = pdist(data_matrix)
            mpd = float(pairwise.mean()) if pairwise.size else 0.0
            if mpd > 0:
                for key in ("mean_persistence_h0", "max_persistence_h0",
                            "mean_persistence_h1"):
                    results[f"_meta_layer_{l_idx}_{key}_norm"] = summary[key] / mpd
                    if _r:
                        results[f"{_r}_layer_{key}_norm"] = summary[key] / mpd
                results[f"_meta_layer_{l_idx}_mean_pairwise_distance"] = mpd
                if _r:
                    results[f"_meta_{_r}_layer_mean_pairwise_distance"] = mpd
                for k, v in _scale_stratified_summary(dgms, mpd).items():
                    results[f"_meta_layer_{l_idx}_{k}"] = v
                    if _r:
                        # h1_fine_* was constant (0.0) on every model.
                        _pfx = "_meta_" if k.startswith("h1_fine") else ""
                        results[f"{_pfx}{_r}_layer_{k}"] = v

        # H1 summaries do not separate models below a few hundred points.
        results["_meta_h1_resolves_between_models"] = bool(
            len(layer_representations.get(_sorted_targets[0], [])) >= 200)
        return results
