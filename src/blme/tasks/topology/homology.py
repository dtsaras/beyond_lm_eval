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
        
        # POST-FINAL-NORM. `target_layers` always includes `num_layers - 1`,
        # and `hidden_states[num_layers]` is `ln_f(b_{L-1})`, so the deepest of
        # the three point clouds was a LayerNorm output. All THREE topology
        # siblings already hook (`betti_curve.py`, `persistence_entropy.py`,
        # `persistence_landscape.py`); this one did not, and no test covered
        # its hidden-state path. Persistence is a metric-space statistic, so
        # normalising every point onto a sphere of fixed radius rewrites it
        # wholesale (red-team review, 2026-09-05, deepest layer):
        #
        #                          gpt2               distilgpt2         pythia-70m
        #   mean_persistence_h0  9.6559->63.9184    7.9500->50.0326    19.4694->6.0086
        #                          (+562%)            (+529%)            (-69%)
        #   max_persistence_h0  13.5337->82.3278    9.9653->67.0848    24.5858->8.2516
        #   mean_persistence_h1  0.5394->1.8351     0.4194->0.9718     0.4497->0.1465
        #   mean_persistence_h0_norm 0.4170->0.7309 0.5491->0.7641     0.7821->0.6814
        #
        # The NORMALISED twins move too, so this is not a common rescale, and
        # the sign is architecture-family determined.
        _sink = {}
        _hook = None
        _n_fwd = _n_sub = 0
        # `get_layers` is imported at module level; a LOCAL import here
        # made it a local name for the WHOLE function, so the earlier
        # `layers = get_layers(model)` above raised UnboundLocalError.
        try:
            _blocks = layers
            if _blocks is not None and len(_blocks) > 0:
                _hook = _blocks[-1].register_forward_hook(
                    lambda mod, inp, outp: _sink.__setitem__(
                        "h", (outp[0] if isinstance(outp, tuple)
                              else outp).detach()))
        except Exception as e:  # pragma: no cover - defensive
            logger.info(f"homology: could not hook last block ({e})")

        # post-final-norm / hook hygiene: a task that fails must leave the model exactly as it found it. A leaked forward hook is paid by every LATER task in the run and pins the last activation tensor alive (discovery-based leak test, 2026-09-05).
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
                    # post-final-norm: SUBSTITUTED -- the deepest sampled layer must be a
                    # block output like every other entry in this profile.
                    hs = list(out.hidden_states)
                    _n_fwd += 1
                    _h_raw = _sink.get("h")
                    if (_h_raw is not None and len(hs) >= 2
                            and _h_raw.shape == hs[-1].shape):
                        hs[-1] = _h_raw
                        _n_sub += 1

                    # Take the mean pooling over the sequence to represent the sentence
                    for l_idx in target_layers:
                        # hidden_states includes embedding as index 0, so layer l_idx is l_idx + 1
                        # .float() first: numpy doesn't accept bf16 tensors, and
                        # ripser requires a standard numpy dtype.
                        hidden = hs[l_idx + 1][0] # shape (seq_len, hidden_dim)
                        # The trailing .float() DISCARDED the precision the .double() was added to
                        # gain: promote, average, then narrow back is self-defeating. Removing only
                        # this cast moved geometry_correlation_dimension from 7.486064047571674 to
                        # 15.295160200026809 on native Pythia-70m hidden states (+104%), matching an
                        # independently reconstructed Grassberger-Procaccia value, and restored a
                        # lossless translation from 0.0 to 2.3540184486398976 (codex round 15,
                        # 2026-09-09). Swept as a class: 7 sites carried this exact pattern.
                        sentence_rep = hidden.double().mean(dim=0).cpu().numpy()
                        layer_representations[l_idx].append(sentence_rep)
                
        finally:
            if _hook is not None:
                _hook.remove()

        results = {}
        results["_meta_last_entry_is_block_output"] = bool(
            _n_fwd > 0 and _n_sub == _n_fwd)
        # DEPTH-ALIGNED names, alongside the absolute-index ones.
        #
        # Every key here was `layer_<ABSOLUTE INDEX>_*`. Across the 90 models in
        # the local HF cache that produces **37 distinct column names**, of
        # which only `layer_0_*` covers 100% of models and the rest cover 1-17%
        # -- so a cross-model feature matrix is almost all missing values. Worse,
        # six of the names ALIAS TWO DIFFERENT ROLES: `layer_11_*` is the
        # DEEPEST layer for gpt2/pythia-160m/opt-125m/bert (L=12) and the MIDDLE
        # layer for TinyLlama-1.1B (L=22); likewise layer_15/17/21/23/31. A
        # regression on those columns is fitting two different quantities to one
        # coefficient. `topology_homology` was the only topology task emitting
        # no depth-aligned scalar at all -- `betti_curve` has `betti_0_first`/
        # `_last` and `simplification_ratio` (red-team review, 2026-09-05).
        #
        # `target_layers` is sorted({0, L//2, L-1}), so the roles are known.
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
            data_matrix = np.array(data_points) # Shape: (num_samples, hidden_dim)

            # RESCALE TO UNIT MAX BEFORE RIPSER, THEN SCALE THE BARCODE BACK.
            #
            # Vietoris-Rips lifetimes are homogeneous of degree 1 in the
            # coordinates: multiply every point by c > 0 and every birth and
            # death multiplies by c, so the NORMALISED summaries are unchanged.
            # Ripser consumes float32 and squares distances internally, so at
            # 2^-100 the squared distances underflow and at 2^+100 they
            # overflow -- and it returns an EMPTY barcode rather than failing.
            # Measured on a 35x6 cloud (independent gpt-5.6-sol review,
            # 2026-09-06; reproduced here):
            #
            #   scale 1     mean H0 1.617843,  9 H1 bars
            #   2^-100      0.0,               0 bars
            #   2^+100      0.0,               0 bars
            #   GUDHI       correct at all three
            #
            # Dividing by the max entry only shifts exponents, so the barcode
            # this produces is exact and multiplying the lifetimes back by the
            # same factor restores the original scale.
            # A SEPARATE ARRAY. Rebinding `data_matrix` here silently broke every
            # `*_norm` output: `pdist(data_matrix)` further down then measured
            # the RESCALED cloud, so the mean pairwise distance came out `_hg`
            # times too small and every normalised summary `_hg` times too
            # large. Measured on real gpt2 (independent Opus review,
            # 2026-09-07): x23.782059 / x160.05917 / x191.996 for layers 0/6/11,
            # with `last_layer_mean_persistence_h0_norm` shipping as **140.33**
            # where the correct value is **0.73087684** -- and 0.7309 is exactly
            # the figure this file's own comment records for that quantity. It
            # also forced all eight `fine_*` columns to a constant zero, which
            # the file then misattributed to sample size.
            #
            # The output was self-evidently impossible and shipped anyway: a
            # mean H0 bar of 63.9 against a reported mean pairwise distance of
            # 0.4555.
            # CENTRE FIRST, then rescale.
            #
            # Vietoris-Rips persistence is exactly TRANSLATION invariant -- it
            # depends only on pairwise distances -- and this was not:
            # translating the seeded cloud by +1e6 returned **14 H1 bars with
            # mean H1 0.07011418** against 9 bars at offset 0 (gpt-5.6-sol
            # round 9, 2026-09-07). `pdist` on a float32 cloud far from the
            # origin cancels catastrophically, so genuinely distinct points
            # collapse onto each other and spurious bars appear.
            #
            # Subtracting the centroid is a mathematical no-op for Rips and
            # restores every digit, so it is pure gain. It must happen BEFORE
            # the magnitude rescale, or the rescale sizes itself against the
            # offset rather than the data -- the ordering lesson from
            # `matrix_entropy` the same day.
            #
            # `_hg` still describes the CENTRED cloud, which is what the
            # restore below must undo, and lifetimes are translation invariant
            # so no offset needs restoring.
            _centred = np.asarray(data_matrix, dtype=np.float64)
            # QUOTIENT DUPLICATE POINTS, exactly as topology_betti_curve does.
            #
            # A repeated point contributes an H0 bar of length ZERO (it dies at
            # radius 0), which is a fabricated finite bar: it drags the mean
            # persistence down without corresponding to any feature. On
            # [(0,0), (0,1), (1,0), (0,1)] the finite H0 bars are [0, 1, 1] and
            # the mean reads 0.6666666902601719 against the analytic and
            # released-reference 1.0 (-33.33%), reproduced under all 24 row
            # permutations; on fresh GPT-2 it reads 64.84628119915529 against
            # 89.78715679196964 at all 12 sampled layers across 4 checkpoints
            # (codex round 13, 2026-09-08).
            #
            # I applied this quotient to `betti_curve` in round 12 and did NOT
            # sweep it to this file, which is the recurring failure this repo
            # documents: fix one site of a class, leave the siblings.
            # Deduplication commutes with the centring below, since identical
            # rows stay identical under a common shift.
            _centred = np.unique(_centred, axis=0)
            _centred = _centred - _centred.mean(axis=0, keepdims=True)
            _hg = float(np.max(np.abs(_centred))) if _centred.size else 0.0
            if np.isfinite(_hg) and _hg > 0:
                _rips_input = _centred / _hg
            else:
                _hg = 1.0
                _rips_input = _centred

            # Compute Persistent Homology
            # maxdim=1 computes up to 1-dimensional holes (loops)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dgms = ripser(_rips_input, maxdim=1)['dgms']
            # Undo the rescale so every emitted lifetime is in the original
            # units. `_lifespan_summary` and the `_norm` twins below then see
            # exactly what they would have without the detour.
            if _hg != 1.0:
                dgms = [d * _hg for d in dgms]

            # dgms[0] contains 0-dimensional features (connected components)
            # dgms[1] contains 1-dimensional features (loops)

            # Summarize finite-bar lifespans (shared, testable helper).
            summary = _lifespan_summary(dgms)
            # `_meta_` on the absolute-index names: they are kept for
            # continuity with the archive and for single-model inspection, but
            # they must not enter a cross-model feature matrix under a name
            # that means different depths for different models.
            for _k in ("mean_persistence_h0", "max_persistence_h0",
                       "mean_persistence_h1", "num_loops_h1"):
                results[f"_meta_layer_{l_idx}_{_k}"] = summary[_k]
                if _r:
                    results[f"{_r}_layer_{_k}"] = summary[_k]

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
                    results[f"_meta_layer_{l_idx}_{key}_norm"] = summary[key] / mpd
                    if _r:
                        results[f"{_r}_layer_{key}_norm"] = summary[key] / mpd
                results[f"_meta_layer_{l_idx}_mean_pairwise_distance"] = mpd
                if _r:
                    results[f"_meta_{_r}_layer_mean_pairwise_distance"] = mpd
                for k, v in _scale_stratified_summary(dgms, mpd).items():
                    results[f"_meta_layer_{l_idx}_{k}"] = v
                    if _r:
                        # `h1_fine_*` is `_meta_`: measured exactly 0.0 on 3/3
                        # models, a zero-variance column that looks like a
                        # measurement. 20 mean-pooled points in R^768 cannot
                        # support a fine-scale H1 statistic (independent Opus
                        # review, 2026-09-06).
                        _pfx = "_meta_" if k.startswith("h1_fine") else ""
                        results[f"{_pfx}{_r}_layer_{k}"] = v

        # H1 AT THE SHIPPED SAMPLE SIZE IS NOISE, and says so.
        #
        # Over 30 random 20-of-200 draws against 4 models, between-model SD
        # divided by mean within-model SD is 6.4 for `mean_persistence_h0` but
        # **0.099** for `mean_persistence_h1` and 0.22 for its normalised twin;
        # `h1_fine_persistence_frac` was exactly 0.0000 for all four models, a
        # zero-variance column. 20 mean-pooled points in R^768 cannot support a
        # 1-dimensional homology estimate. The H1 keys stay (they are correct
        # computations of the estimand) but a caller must not read a
        # cross-model ordering off them at this N.
        results["_meta_h1_resolves_between_models"] = bool(
            len(layer_representations.get(_sorted_targets[0], [])) >= 200)
        return results
