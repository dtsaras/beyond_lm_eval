from ...tasks.base import DiagnosticTask
from ...registry import register_task
import torch
import numpy as np
from tqdm import tqdm
from .utils import collect_hidden_states
import logging
logger = logging.getLogger("blme")


@register_task("geometry_intrinsic_dim")
class IntrinsicDimensionTask(DiagnosticTask):
    """
    Estimates the Intrinsic Dimension (ID) of the embedding manifold
    using the Two-NN estimator.

    References:
      * Facco, d'Errico, Rodriguez, Laio 2017 — "Estimating the Intrinsic
        Dimension of Datasets by a Minimal Neighborhood Information",
        Scientific Reports 7, arXiv:1803.06992. The Two-NN estimator used
        here.
      * Ansuini, Laio, Macke, Zoccolan 2019 — "Intrinsic Dimension of
        Data Representations in Deep Neural Networks", NeurIPS 2019,
        arXiv:1905.12784. Characterises the per-layer ID profile that
        this task produces for language models.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Intrinsic Dimension Estimation (Two-NN)...")
        
        # Check mode: Embeddings (static) or Layer-wise Activations (dynamic)
        layerwise = self.config.get("layerwise", False)
        
        if layerwise:
            logger.info("  Mode: Layer-wise Activations")
            if dataset is None:
                # Mock dataset if missing
                dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]
            use_cache = self.config.get("use_cache", True)
                
            # Collect states from all layers
            logger.info("  Collecting hidden states...")
            # Use 'all' to get dict of {layer_idx: tensor}
            # THE 14TH INSTANCE of the post-final-norm defect, and the one
            # that reaches the most published numbers: `layerwise: true` is
            # set in the study's `configs/tasks.yaml`, so this branch runs for
            # every model in the archive, and all 66 `blme_verified_a/*.json`
            # carry a `lid_layer_L` computed on `ln_f(b_{L-1})` rather than on
            # a block output.
            #
            # Measured on the same 20k cached rows (red-team review,
            # 2026-09-05), shipped `lid_layer_L` vs the true block-L output:
            #
            #   gpt2         8.8603 -> 6.9776  (+27.0% high)
            #   distilgpt2   8.3975 -> 7.1351  (+17.7% high)
            #   pythia-160m  7.8042 -> 7.8883  ( -1.1%)
            #   pythia-70m   8.4167 -> 8.6839  ( -3.1%)
            #
            # The sign is architecture-family determined, so it does NOT
            # cancel in a cross-model comparison. It also manufactures a
            # qualitative claim: gpt2's reported final-layer ID EXPANSION
            # (lid_layer_10 7.365 -> lid_layer_11 9.364, +26.9%) disappears
            # entirely once the LayerNorm is removed -- true block-11 (6.978)
            # is statistically identical to block-10 (6.982), a -0.1% step.
            _id_status = {}
            if cache is not None and cache.is_populated and use_cache:
                layer_activations = cache.get_hidden_states(
                    layer_idx="all",
                    num_samples=self.config.get("num_samples", 100),
                    replace_final_norm=True,
                )
                _id_status["last_is_block"] = bool(
                    getattr(cache, "_last_block_substituted", False))
            else:
                layer_activations = collect_hidden_states(
                    model, tokenizer, dataset,
                    num_samples=self.config.get("num_samples", 100),
                    layer_idx="all", replace_final_norm=True,
                    status=_id_status)
            
            results = {}
            # Compute ID for each layer
            sorted_layers = sorted(layer_activations.keys())
            id_trend = []
            
            for layer_idx in tqdm(sorted_layers, desc="Computing Layer IDs"):
                # PRESERVE the incoming precision. `_compute_id` was repaired to
                # rescale by a power of two and keep float64, but this call site
                # still narrowed to float32 first, so the public layerwise route
                # discarded exactly the information the repaired estimator was
                # careful to keep: a perturbed lattice read 3.552080162654 against
                # the released Facco/Ansuini 4.606333343805 (-22.89%) identically
                # on all three evaluation routes (codex round 13, 2026-09-08).
                X = layer_activations[layer_idx].double().numpy()
                # Subsample if too large
                if len(X) > 20000:
                    rng = np.random.default_rng(
                        int(self.config.get("subsample_seed", 42))
                    )
                    indices = rng.choice(len(X), 20000, replace=False)
                    X = X[indices]
                    
                lid_result = self._compute_id(X)
                lid = lid_result["intrinsic_dimension"]
                # `_meta_` on the ABSOLUTE-INDEX names, plus depth-aligned
                # twins below. Over 66 models these produce **64 distinct
                # column names**, only 6 of which cover 100% of models, and
                # **18 alias two different depth roles** -- `lid_layer_11` is
                # the deepest layer at L=12 and the middle layer at L=22, so a
                # regression fits two different quantities to one coefficient.
                # `topology_homology` was fixed for exactly this on 2026-09-05
                # and this sibling was missed (independent Opus review, same
                # day).
                results[f"_meta_lid_layer_{layer_idx}"] = lid
                id_trend.append(lid)
                
            results["lid_trend"] = id_trend
            # Depth-aligned scalars: the same name means the same DEPTH ROLE on
            # every model, which is what a cross-model feature matrix needs.
            if id_trend:
                _n = len(id_trend)
                results["lid_first_layer"] = float(id_trend[0])
                results["lid_mid_layer"] = float(id_trend[_n // 2])
                results["lid_last_layer"] = float(id_trend[-1])
                results["lid_depth_slope"] = (
                    float(np.polyfit(np.linspace(0.0, 1.0, _n),
                                     np.asarray(id_trend, dtype=float), 1)[0])
                    if _n > 1 and np.all(np.isfinite(id_trend)) else float("nan"))
            # Provenance: True only if the deepest entry is really the raw
            # block output. Without this nothing in the output distinguished a
            # LayerNorm-contaminated profile from a clean one -- which is why
            # 66 archived envelopes carry one and no verdict file noticed.
            results["_meta_last_entry_is_block_output"] = bool(
                _id_status.get("last_is_block", False))
            return results
            
        else:
             logger.info("  Mode: Static Embeddings")
             # 1. Get Embeddings
             from ..common import get_embeddings as _get_emb
             E = _get_emb(model)
             if E is None:
                 return {"error": "Could not extract embeddings"}
             
             # PRESERVE the incoming precision on the STATIC route too. The
             # layerwise call site was repaired first and this one was left, so
             # the static and model-collection routes still returned
             # 3.552080162654 against the released Facco/Ansuini 4.606333343805
             # (-22.89%) on the perturbed lattice while the layerwise route was
             # already correct -- the same class fixed at one site and not swept
             # (codex round 14, 2026-09-09).
             E_np = E.double().cpu().numpy()
             # `sample_size`, not `_meta_sample_size`: the `_meta_` prefix marks
             # OUTPUT provenance keys, so the old spelling could never be set
             # by a config and this subsample knob was permanently dead (both
             # reviewers, round 1). `None` keeps the previous behaviour.
             return self._compute_id(
                 E_np, sample_size=self.config.get("sample_size", None))

    def _compute_id(self, X, sample_size=None):
        """Two-NN intrinsic dimension of the point cloud ``X``.

        Uses GPU-batched ``torch.cdist + torch.topk`` when CUDA is
        available (100-1000× faster than sklearn's CPU kNN on
        vocab-scale embeddings). Falls back to sklearn on CPU-only
        hosts. Returns the same ``{intrinsic_dimension, sample_size}``
        dict either way.
        """
        # RESCALE TO UNIT MAX FIRST, both paths.
        #
        # Two-NN depends only on `mu = r2/r1`, so a positive global rescale
        # cancels EXACTLY -- Facco's statistic cannot depend on it. The CPU
        # fallback did: on a max-normalised `default_rng(0)` 500x32 Gaussian it
        # returned 24.233986018 at scale 1, **16.927948590 at 1e-22 (-30.15%)**,
        # and NaN at 1e-30, 1e-25, 1e25 and 2.406e38, because sklearn's kNN
        # squares the coordinates and underflows. The CUDA path was unaffected,
        # which is why this went unreproduced when it was first reported: the
        # check was run on a GPU host and never exercised the fallback
        # (gpt-5.6-sol review, 2026-09-06 and 2026-09-07; the second pass
        # supplied the fixture and the CPU detail, and it reproduces to nine
        # digits).
        #
        # THE DIVISOR MUST BE A POWER OF TWO.
        #
        # The line above used to read "dividing by the max only shifts
        # exponents, so the ratios below are bit-for-bit what they would have
        # been". That is true ONLY when the divisor is a power of two. Dividing
        # by an arbitrary max ROUNDS every coordinate, and Two-NN is not a
        # smooth function of the coordinates: it excludes duplicate points and
        # equal-radius (r1 == r2) queries by EXACT comparison, so re-rounding
        # silently changes which queries are excluded.
        #
        # Counterexample (lossless integer lattice, so nothing is lost to the
        # fixture itself): G = meshgrid(arange(-16,17), arange(-16,17)); take
        # 500 distinct rows as float32 / 16, then translate by +2. Every
        # pairwise distance is bit-identical and all 500 rows stay distinct,
        # yet evaluate() returned 2.170755946319 untranslated and
        # 3.765359458874 translated (+73.4584%) -- because the untranslated max
        # is exactly 1 (division is a no-op) while the translated max is 3
        # (division rounds), moving the valid-ratio count from 166 to 335.
        # Released pdist + Facco/Ansuini returns 2.170755946319 on BOTH, and
        # reproduces BLME's wrong value when fed BLME's divided coordinates --
        # so the estimator was never at fault, the preprocessing was. Over all
        # 65 integer offsets the released answer is constant while BLME
        # exceeded 1% on 58, median +46.7954%, max +85.6091% at offset 9. The
        # first failure is at coordinate max 3, i.e. 1014.7x BELOW the measured
        # GPT-2 hidden max of 3044.142334 (codex round 12, 2026-09-08).
        #
        # Scaling to a power of two keeps the mantissas untouched, which is what
        # the original comment promised, and still fixes the sklearn
        # underflow/overflow the rescale was introduced for.
        _g = float(np.max(np.abs(np.asarray(
            X.detach().cpu() if isinstance(X, torch.Tensor) else X, dtype=np.float64))))
        if np.isfinite(_g) and _g > 0:
            _s = float(2.0 ** np.ceil(np.log2(_g)))   # exact: mantissa preserved
            if np.isfinite(_s) and _s > 0:
                X = (X / _s) if isinstance(X, torch.Tensor) else (np.asarray(X) / _s)

        # PRESERVE float64. Narrowing here re-quantises the coordinates that the
        # power-of-two rescale was just careful not to disturb, and the
        # `.double()` calls further down then promote an already-damaged tensor
        # -- a fix that is present in the source and has no effect.
        if isinstance(X, torch.Tensor):
            X_tensor = X.detach().double()
        else:
            X_tensor = torch.from_numpy(np.asarray(X, dtype=np.float64))

        # DROP NON-FINITE ROWS, AND SAY SO.
        #
        # A matrix carrying inf/NaN previously produced a perfectly plausible
        # number with no signal whatsoever: on real gpt2 layer 0 scaled to the
        # top of float32, 51 of 307968 entries were inf and `_compute_id`
        # returned **16.342547** against the clean 16.404143 -- publishable
        # looking, and wrong. Layers 5 and 11 (38.1% and 81.3% non-finite)
        # returned NaN, which is the correct answer for a destroyed input; it
        # was the layer that did NOT fail that was dangerous.
        #
        # Note this is NOT a scale-invariance defect: the input overflowed
        # float32 before the task ever saw it, so no rescale inside the task
        # could recover it (this refines the round-9 report, which read the
        # NaN as a surviving scale break). What the task owes the caller is a
        # count, not a silent estimate over whatever rows happened to survive.
        _finite_rows = torch.isfinite(X_tensor).all(dim=1)
        _n_nonfinite = int((~_finite_rows).sum().item())
        if _n_nonfinite:
            logger.warning(
                "intrinsic_dim: dropping %d/%d rows with non-finite entries",
                _n_nonfinite, int(X_tensor.shape[0]))
            X_tensor = X_tensor[_finite_rows]
            if X_tensor.shape[0] < 4:
                return {"intrinsic_dimension": float("nan"),
                        "_meta_sample_size": int(X_tensor.shape[0]),
                        "_meta_n_nonfinite_rows": _n_nonfinite}

        n_vocab = X_tensor.shape[0]
        if sample_size and sample_size < n_vocab:
            rng = np.random.default_rng(42)
            indices = rng.choice(n_vocab, sample_size, replace=False)
            X_tensor = X_tensor[torch.from_numpy(indices)]

        N = int(X_tensor.shape[0])
        if N < 4:
            return {"intrinsic_dimension": float("nan"), "_meta_sample_size": N}

        r1, r2 = _two_nearest_neighbor_distances(X_tensor)
        if r1 is None:
            return {"intrinsic_dimension": float("nan"), "_meta_sample_size": N}

        # Drop exact duplicates (r1 = 0 → mu undefined). Near-duplicates
        # are kept; they're valid density samples.
        valid = r1 > 0
        r1, r2 = r1[valid], r2[valid]
        if r1.size < 4:
            return {"intrinsic_dimension": float("nan"), "_meta_sample_size": N}

        mus = r2 / r1
        # Filter numerical noise: mu must be ≥ 1 by construction.
        mus = mus[mus > 1.0]
        if mus.size < 4:
            return {"intrinsic_dimension": float("nan"), "_meta_sample_size": N}

        intrinsic_dim = _twonn_linear_fit(mus)
        return {
            "intrinsic_dimension": float(intrinsic_dim),
            "_meta_sample_size": N,
            "_meta_n_nonfinite_rows": _n_nonfinite,
        }


def _two_nearest_neighbor_distances(
    X: torch.Tensor,
    chunk_rows: int = 4096,
) -> tuple:
    """Return ``(r1, r2)`` — first and second nearest-neighbor distances
    for every row of ``X`` — as 1-D numpy arrays.

    Uses a GPU ``torch.cdist`` + ``torch.topk`` implementation when
    CUDA is available; otherwise falls back to ``sklearn.neighbors``.
    Chunked along rows so the (Q, N) distance matrix doesn't blow up
    memory on 150 k-vocab embeddings.
    """
    # Prefer GPU (Tesla/RTX, ~1000× faster than sklearn on vocab-scale).
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = None

    if device is not None:
        try:
            # float64 + no-mm compute mode (audit 2026-07-16): the default
            # f32 mm-mode cdist (x^2+y^2-2xy) catastrophically cancels when
            # r1 << ||x||, inflating TwoNN by up to +23% on dense clouds and
            # making results host-dependent (GPU vs CPU). f64 + direct
            # euclidean brings the error to ~1e-11.
            X_dev = X.to(device).double()
            N = X_dev.shape[0]
            r1 = np.zeros(N, dtype=np.float64)
            r2 = np.zeros(N, dtype=np.float64)
            for s in range(0, N, chunk_rows):
                e = min(N, s + chunk_rows)
                d = torch.cdist(X_dev[s:e], X_dev, p=2,
                                compute_mode="donot_use_mm_for_euclid_dist")
                # Take top-3 smallest; index 0 is self (distance 0).
                vals, _ = torch.topk(d, k=3, dim=1, largest=False, sorted=True)
                r1[s:e] = vals[:, 1].detach().cpu().double().numpy()
                r2[s:e] = vals[:, 2].detach().cpu().double().numpy()
                del d, vals
            return r1, r2
        except Exception as e:
            logger.info(
                f"GPU kNN failed ({type(e).__name__}: {e}); "
                "falling back to sklearn."
            )

    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        logger.info("sklearn not installed, skipping ID estimation")
        return None, None

    # float64 + a TREE algorithm, matching the GPU branch's arithmetic.
    #
    # The GPU branch above uses double precision and
    # `donot_use_mm_for_euclid_dist` precisely because the Gram identity
    # (x^2 + y^2 - 2xy) cancels catastrophically when r1 << ||x||. This
    # fallback did neither: `.float()` plus `algorithm="auto"`, which resolves
    # to brute force and uses that same identity. The result was a host-
    # dependent estimate -- the exact failure the GPU comment says it fixed --
    # reading -5.12% low with exact duplicates at r1 = 3.576e-07 on d=768
    # (Opus round 11, 2026-09-08).
    #
    # Measured against the GPU branch's own reference (torch double, direct
    # differences) on a 400x768 cloud offset by 50, worst relative r1 error:
    #     auto/brute  float32  1.708e-07
    #     auto/brute  float64  1.692e-12   <- Gram residue survives promotion
    #     ball_tree   float32  1.356e-07
    #     ball_tree   float64  0.000e+00   <- exact
    # so the algorithm matters as much as the dtype: promoting alone leaves a
    # picometre of the identity behind, and only a tree computes the true
    # distance. `n_jobs` is dropped because tree queries are already threaded
    # and it is not accepted for construction.
    #
    # This is the SECOND time this fallback has been missed: the 2026-09-07
    # unit-max rescale was applied to both paths, and the dtype/algorithm was
    # left GPU-only. A branch that only runs on hosts without CUDA is invisible
    # to every check run on a GPU box.
    X_np = X.detach().cpu().double().numpy()
    nbrs = NearestNeighbors(
        n_neighbors=3, algorithm="ball_tree", metric="euclidean",
    ).fit(X_np)
    d, _ = nbrs.kneighbors(X_np)
    return d[:, 1], d[:, 2]


def _twonn_linear_fit(mus: np.ndarray, tail_trim: float = 0.1) -> float:
    """Two-NN estimator via the linear-regression form of Facco et al.
    2017, following the five-step algorithm on p. 3 of the paper and
    the reference implementation in ``scikit-dimension``.

    The method fits the empirical CDF
    ``log(1 − F(μ)) = −d · log(μ)`` by OLS through the origin,
    discarding the largest ``tail_trim`` fraction of μ values
    (the paper and skdim use 10 %) to dampen the influence of the
    heavy upper tail on the slope. We do NOT trim the lower tail —
    small μ's (μ ≈ 1) are the most informative points, not noise.

    We also do NOT floor the estimate at 1: the Two-NN Pareto model has
    support d in (0, inf), so estimates below 1 are valid outputs on
    near-degenerate geometries and carry diagnostic signal. (The paper
    itself never discusses sub-1 estimates — no floor is implied by it.)
    If the downstream analysis wants a lower bound it can apply one
    explicitly.
    """
    if mus.size == 0:
        return float("nan")

    mus_sorted = np.sort(mus)
    n = mus_sorted.size
    # Empirical CDF F_i = i / N, matching Facco step 4 and
    # scikit-dimension's TwoNN implementation.
    F = np.arange(1, n + 1, dtype=np.float64) / n

    # Trim only the top `tail_trim` fraction (paper default: 10 %).
    # The last index has F = 1 → log(1 − F) = −∞; the trim also drops
    # that singularity.
    n_keep = int(n * (1.0 - tail_trim))
    if n_keep < 4:
        n_keep = n  # tiny sample — keep everything, accept noise
    mus_keep = mus_sorted[:n_keep]
    F_keep = F[:n_keep]

    # Guard against μ = 1 exactly (log μ = 0) — that's a valid (0, 0)
    # point on the regression line, but some samples may also hit
    # F = 1 at the top. We've already trimmed the top; for any
    # remaining F = 1 (can't happen with i/N and i < n), skip.
    valid = (mus_keep > 1.0) & (F_keep < 1.0)
    if valid.sum() < 4:
        return float("nan")

    x = np.log(mus_keep[valid])
    y = -np.log(1.0 - F_keep[valid])

    # OLS through the origin: d = Σ xy / Σ x²
    denom = float(np.sum(x * x))
    if denom <= 0:
        return float("nan")
    d_est = float(np.sum(x * y) / denom)
    if not np.isfinite(d_est) or d_est <= 0:
        return float("nan")
    return d_est
