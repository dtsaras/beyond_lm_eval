from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings
import torch
import numpy as np
from scipy.stats import skew
from tqdm import tqdm
import logging
logger = logging.getLogger("blme")


def _gini_from_counts(counts: np.ndarray) -> float:
    counts = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(counts)
    total = counts.sum()
    if n == 0 or total <= 0:
        return 0.0
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float((2 * np.sum(ranks * counts)) / (n * total) - (n + 1) / n)


def _hubness_stats_from_occurrences(n_occ, k: int) -> dict:
    """Hubness skew/max/top-1%/Gini from k-NN occurrence counts."""
    n_occ_cpu = np.asarray(n_occ, dtype=np.int64)
    if n_occ_cpu.size == 0:
        hub_skew = float("nan")
    elif n_occ_cpu.std() == 0:
        # NaN, not 0.0. S_Nk is the standardized third moment, so a constant
        # occurrence vector makes it 0/0 -- undefined -- and scipy.stats.skew
        # returns NaN there. Reporting 0.0 says "perfectly symmetric", which is
        # a measurement the data does not support: 20 equal counts returned
        # 0.000000 (adversarial review round 2, 2026-09-04).
        hub_skew = float("nan")
    else:
        hub_skew = float(skew(n_occ_cpu))
    hub_max = int(n_occ_cpu.max()) if n_occ_cpu.size else 0
    denom = n_occ_cpu.sum()
    # Mass of the top 1% of points, taken as EXACTLY the m = ceil(0.01 n)
    # largest occurrence counts.
    #
    # The previous form selected everything >= percentile(counts, 99), which
    # over-selects whenever counts tie -- and k-NN occurrence counts are small
    # integers, so ties are the norm, not an edge case. Measured over-selection
    # on real tables: gpt2 540 of 50257 points selected instead of 503
    # (+0.77% mass), pythia-70m 599 vs 504 (+16.0%), pythia-160m 602 vs 504
    # (+16.6%), Qwen2.5-0.5B 1528 vs 1520 (+0.24%). Degenerate case: 100 nodes
    # all with the same count selected all 100 and reported mass 1.0 for what
    # is a perfectly uniform, hub-free distribution.
    n_pts = int(n_occ_cpu.size)
    if denom > 0 and n_pts > 0:
        m = int(np.ceil(0.01 * n_pts))
        m = max(1, min(m, n_pts))
        # np.partition puts the m largest in the tail; no full sort needed.
        top_m = np.partition(n_occ_cpu, n_pts - m)[n_pts - m:]
        top_1pct_mass = float(top_m.sum() / denom)
    else:
        top_1pct_mass = 0.0
    return {
        f"hubness_k{k}_skew": float(hub_skew),
        f"hubness_k{k}_max": hub_max,
        f"hubness_k{k}_top1pct": float(top_1pct_mass),
        f"hubness_k{k}_gini": _gini_from_counts(n_occ_cpu),
    }


@register_task("geometry_hubness")
class GlobalHubnessTask(DiagnosticTask):
    """
    Analyzes the "hubness" of the embedding space: the skewness of the distribution
    of k-nearest neighbor occurrences.

    References:
      * Tomašev, Radovanović, Mladenić, Ivanović 2014 — "The Role of
        Hubness in Clustering High-Dimensional Data", IEEE TKDE 26 (3),
        IEEE TKDE 26(3):739-751, 2014 (DOI 10.1109/TKDE.2013.25; no arXiv).
        Formalises the k-NN occurrence-skewness measure S_Nk used here.

        NOT the source of the Gini variant. Both cited papers define hubness
        through N_k and S_Nk; neither mentions a Gini coefficient (searched
        both PDFs, 0 occurrences -- codex round 9, 2026-09-07). The docstring
        previously credited them with "the k-NN skewness / Gini hubness
        measures", which is a citation to a thing they do not contain.
        ``hubness_k*_gini`` is a BLME add-on: the standard Gini coefficient
        applied to the same self-excluded N_k vector. It is internally
        correct (0.5146666666666668 against the independent pairwise
        definition's 0.5146666666666667 on the pinned vector), but it is ours
        to defend, not Tomasev's.
      * Radovanović, Nanopoulos, Ivanović 2010 — "Hubs in Space: Popular
        Nearest Neighbors in High-Dimensional Data", JMLR 11. The
        originator paper for k-NN hubness in high-D.

    ``hubness_k10_gini`` is BLME's top-7 partial predictor of composite
    benchmark capability beyond scale (+0.74 partial ρ; see
    `docs/TOP_PREDICTORS.md` §2).

    Implementation runs the k-NN search on GPU. Neither ``torch.topk`` nor the
    similarity matmul this docstring used to describe survives: ``topk`` left
    exact ties unordered (making the answer depend on which k values were also
    requested) and was replaced by a stable full argsort, and the normalized
    dot product lost neighbour resolution on anisotropic vocabularies and was
    replaced by direct chord distances -- see the two long notes in
    ``evaluate``. The numpy-on-CPU original was infeasible for VLM-scale vocabs
    (Gemma 4's 262k tokens × 6k hidden, ~hours on CPU); the GPU version is
    ~26 s for gpt2's 50257.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Global Hubness Analysis...")
        k_values = self.config.get("k_values", [10, 50, 100])
        batch_size = self.config.get("batch_size", 2000)

        E = get_embeddings(model)
        if E is None:
            return {"error": "Could not extract embeddings"}

        # Choose a CUDA device when available; the embedding's own device is
        # ideal because the weights are already there. Fall back to CPU.
        if E.is_cuda:
            device = E.device
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        # Cast to float32 for the cosine similarity (bf16 has too little
        # precision for the small differences between top-k similarities) and
        # normalize for cosine similarity in one shot. We materialise this on
        # the chosen device.
        # float64, NOT float32. The ranking below differences nearly-identical
        # unit rows; float32 normalisation injects a ~6e-8 relative error into
        # each row, and when the vocabulary shares a strong common direction
        # that error is LARGER than the row-to-row separation being ranked.
        # Upcasting from a float32/bf16 checkpoint is exact, so this only ever
        # removes error -- it never invents precision the checkpoint lacks.
        E_norm = E.detach().to(device=device, dtype=torch.float64)
        # DIVIDE by the norm; do not ADD to it.
        #
        # `norm + 1e-10` is not cosine normalisation for small nonzero rows --
        # it shrinks them toward the origin and changes the neighbour geometry.
        # TinyLlama-intermediate has 111 such rows, where current-vs-exact k100
        # skew differs by -0.345%, and a direct 1e-12 row-rescaling test moved
        # k50 skew by 3.19814 (adversarial review, 2026-09-04). Only an exactly
        # zero row has no direction; clamp the divisor at the smallest positive
        # value instead of biasing every row.
        # TWO-STAGE rescale: divide each row by its own largest-magnitude
        # coordinate BEFORE norming. `finfo.tiny` guards the divisor but not
        # the norm's own arithmetic -- the norm squares in float32, so at
        # alpha = 1e-23 all 24 stored coordinates were still nonzero while the
        # norm underflowed to 0.0, changing 77 of 257 k50 occurrence counts and
        # moving skew 0.393048896 -> 7.301942076 (adversarial review round 2,
        # 2026-09-04).
        #
        # This used to claim "scaling by the row max is exact in binary
        # floating point and yields the identical unit vector". It is NOT
        # exact -- only a POWER-OF-TWO divisor leaves the mantissas alone, and
        # an arbitrary row max re-rounds every coordinate. Measured on a
        # 2000x256 standard normal: 0 of 2000 unit rows are bit-identical to
        # the power-of-two-scaled result, with max coordinate deviation
        # 5.960e-08. It is kept anyway because what this task CONSUMES is the
        # k-NN ranking, and that is unchanged: 2000 of 2000 rows have identical
        # k=10 neighbour lists under both scalings. The same false "only shifts
        # exponents" claim WAS load-bearing in intrinsic_dim, where Two-NN
        # compares radii for exact equality -- see the power-of-two rescale
        # there (codex round 12, 2026-09-08).
        _rmax = E_norm.abs().amax(dim=1, keepdim=True)
        E_norm = E_norm / _rmax.clamp_min(torch.finfo(E_norm.dtype).tiny)
        _n = E_norm.norm(dim=1, keepdim=True)
        E_norm = E_norm / _n.clamp_min(torch.finfo(E_norm.dtype).tiny)
        # A row that was exactly zero stays exactly zero (0/tiny = 0), which is
        # what "no direction" means.
        #
        # This comment used to end "it is reported below, not silently ranked"
        # -- and nothing was reported (codex round 9, 2026-09-07). The counts
        # below make the sentence true. Note `_rmax > 0` is False for NaN, so
        # a NaN row also lands here and is counted separately: a NaN row and a
        # zero row are NOT the same condition, and collapsing both to the zero
        # vector without saying so hides a corrupt checkpoint.
        # Counted on the row max BEFORE the where(), so the two conditions
        # stay distinguishable. `+inf` is deliberately counted here too: it
        # PASSES `> 0`, so it is not zeroed, and inf/inf leaves NaN in the
        # ranked row -- a silently different failure from an all-zero row.
        _rm = _rmax.squeeze(-1)
        _n_zero = int((_rm == 0).sum().item())
        _n_nonfinite = int((~torch.isfinite(_rm)).sum().item())
        if _n_zero or _n_nonfinite:
            logger.warning(
                "hubness: %d zero-norm and %d non-finite embedding rows; "
                "they have no direction and their neighbour ranks are not "
                "meaningful", _n_zero, _n_nonfinite)
        E_norm = torch.where(_rmax > 0, E_norm, torch.zeros_like(E_norm))
        n_vocab = E_norm.shape[0]
        max_k = max(k_values)

        # Cap the batch so the float64 (B, V) distance slice stays inside a
        # 512 MB budget. At gpt2's V=50257 the configured 2000 would ask for
        # 804 MB in one allocation; this trims it to 1336 rows. Purely a
        # memory guard -- the stable tie rule makes the result batch-invariant,
        # so trimming cannot change any number.
        _bytes = E_norm.element_size() * max(1, n_vocab)
        batch_size = max(1, min(int(batch_size), (512 << 20) // _bytes))

        # Per-token occurrence counters, one per k-value, kept on the same
        # device so the topk results can be scattered without round-trips.
        n_occ = {k: torch.zeros(n_vocab, dtype=torch.int64, device=device)
                 for k in k_values}

        with torch.no_grad():
            for i in tqdm(range(0, n_vocab, batch_size), desc="Hubness"):
                end = min(i + batch_size, n_vocab)
                # (B, V) slice ranked by DIRECT chord distance, not by the
                # dot product.
                #
                # For unit rows ||u-v||^2 = 2 - 2*u.v, so the two orderings are
                # mathematically identical -- but not numerically. When the
                # vocabulary shares a common direction every u.v sits just
                # below 1, and the quantity that actually separates neighbours
                # is 1 - u.v, obtained by cancelling two ~1.0 sums. Differencing
                # the coordinates first is exact for nearby floats (Sterbenz),
                # so the common component cancels before anything is squared.
                #
                # Measured on a 257x24 fixture given a common offset of 1e3:
                # the dot-product ranking left only 41 distinct off-diagonal
                # similarities out of 32888 and put 212/257 occupancy bins wrong
                # (L1=1172, k10 skew 1.854701 -> 2.905081). At offset 1e6 only
                # 10 distinct similarities survived (242/257 bins, L1=3288).
                # Direct differencing gives 0 wrong bins at both offsets.
                # Upcasting the DOT PRODUCT to float64 does not fix it (186 and
                # 242 bins still wrong): the cancellation destroys the signal
                # regardless of accumulator width, so the differencing -- not
                # the precision -- is the load-bearing half of this fix.
                #
                # Real numbers DO move, slightly. Measured end-to-end on the
                # full gpt2 vocabulary (V=50257, k=10/50/100), worst relative
                # change 3.602e-05:
                #     hubness_k10_skew   34.0717888947 -> 34.0705616505
                #     hubness_k100_max   41027         -> 41028
                #     hubness_k50_max    30325         -> 30326
                #     hubness_k100_top1pct 0.531816861333 -> 0.531816065424
                # The new values are the accurate ones. gpt2 is mildly
                # anisotropic (mean/spread 0.605), enough to cost the float32
                # dot product a few digits even without an artificial offset.
                # Any archived hubness field is stale at the 1e-5 level; no
                # qualitative claim depends on it. Verified 2026-09-07.
                #
                # Cost: 1.1 s -> 26.1 s for the full gpt2 vocabulary. float64
                # is load-bearing -- normalising in float64 but ranking in
                # float32 is exact at offset 1e3 yet wrong in 89 bins at 1e6,
                # for a saving of ~7 s per model. Correctness wins.
                sims = torch.cdist(
                    E_norm[i:end], E_norm,
                    compute_mode="donot_use_mm_for_euclid_dist")
                # Mask self-distance (i+j is the diagonal entry of row j)
                rows = torch.arange(end - i, device=device)
                cols = torch.arange(i, end, device=device)
                sims[rows, cols] = float("inf")
                # STABLE ordering over the FULL row, ranked by
                # (-similarity, index). `topk` leaves exact ties unordered, so
                # which tied neighbour landed inside k depended on max_k -- the
                # SAME k=10 skew on gpt2 read 34.070209318 / 34.070394611 /
                # 34.070093558 for max_k = 10 / 100 / 101, and an all-identical
                # 12-row fixture moved by 1.29 (adversarial review round 2,
                # 2026-09-04). A stable descending argsort breaks ties by index,
                # so the answer no longer depends on which k values were also
                # requested. It costs V log V per row instead of V; `sims` is
                # already materialised, so no extra memory.
                # ASCENDING now: nearest neighbour = smallest distance. The
                # stable tie rule is unchanged -- ties still resolve by index,
                # so the result stays batch- and max_k-invariant.
                top_idx_sorted = torch.argsort(
                    sims, dim=1, descending=False, stable=True)[:, :max_k]
                for k in k_values:
                    flat = top_idx_sorted[:, :k].reshape(-1)
                    n_occ[k].scatter_add_(0, flat, torch.ones_like(flat))

        results = {
            "_meta_n_zero_rows": _n_zero,
            "_meta_n_nonfinite_rows": _n_nonfinite,
        }
        for k in k_values:
            n_occ_cpu = n_occ[k].cpu().numpy().astype(np.int64)
            results.update(_hubness_stats_from_occurrences(n_occ_cpu, k))

        return results
