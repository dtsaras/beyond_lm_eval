from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collect_hidden_states, collected_layer_count, safe_svdvals
import numpy as np
import torch
from tqdm import tqdm
import logging
logger = logging.getLogger("blme")


@register_task("geometry_collapse")
class RepresentationCollapseTask(DiagnosticTask):
    """
    Detects representation collapse by tracking Effective Rank across layers.
    A sharp drop in effective rank indicates dimensional collapse.

    References:
      * Jing, Vincent, LeCun, Tian 2021 — "Understanding Dimensional
        Collapse in Contrastive Self-supervised Learning", ICLR 2022,
        arXiv:2110.09348. Motivating paper for collapse-ratio / erank
        diagnostics.
      * Roy, Vetterli 2007 — "The Effective Rank: A Measure of Effective
        Dimensionality", European Signal Processing Conference. The
        exp(Shannon-entropy-of-normalised-singular-values) erank formula
        this task reports under ``erank_per_layer``.
      * Queipo-de-Llano, Arroyo, Barbero, Dong, Bronstein, LeCun,
        Shwartz-Ziv 2025 —
        "Attention Sinks and Compression Valleys in LLMs are Two Sides
        of the Same Coin", arXiv:2510.06477. The erank-vs-depth profile
        is a BLME collapse/valley proxy, not the full paper pipeline.

    ``erank_per_layer.q75`` and ``collapse_ratio`` are BLME's top-17 and
    top-22 partial predictors beyond scale (+0.71 and +0.68 partial ρ;
    see `docs/TOP_PREDICTORS.md` §2).
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Representation Collapse Detection...")

        if dataset is None:
            dataset = [
                {"text": "The quick brown fox jumps over the lazy dog."}
                for _ in range(50)
            ]

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)

        # Collect hidden states from all layers
        # `_last_is_block` records whether the deepest profile point is a real
        # block output. The shared cache stores HF's raw tuple, whose last
        # entry is ln_f(b_{L-1}); a fresh collection can hook the block and
        # recover its pre-norm output instead.
        if cache is not None and cache.is_populated and use_cache:
            # This is the branch production TAKES: geometry_collapse is in
            # core.py's cache_hidden_tasks and defaults.yaml sets no use_cache,
            # so a default run never reaches the else-branch. Omitting the flag
            # here made the whole fix inert end-to-end.
            layer_activations = cache.get_hidden_states(
                layer_idx="all", num_samples=num_samples, replace_final_norm=True)
            # Derived from whether the swap ACTUALLY fired, not from the
            # mere presence of a hooked tensor (a shape mismatch is
            # rejected inside get_hidden_states).
            _last_is_block = bool(getattr(cache, "_last_block_substituted", False))
        else:
            _st = {}
            layer_activations = collect_hidden_states(
                model, tokenizer, dataset, num_samples=num_samples, layer_idx="all",
                replace_final_norm=True, status=_st,
            )
            _last_is_block = bool(_st.get("last_is_block", False))

        layers = sorted(layer_activations.keys())
        # RESOLVED (2026-09-03): the profile's last point is now the deepest
        # block's RAW output, recovered by a forward hook on layers[-1].
        # Verified exactly: ln_f(hooked) reproduces HF's hidden_states[-1] to
        # 0.0e+00 on gpt2, distilgpt2 and pythia-70m, so the hook captures
        # precisely the pre-norm block output. That removes the whole dilemma
        # below -- there is no LayerNorm entry to include OR delete, and the
        # profile spans blocks 0..L-1 with max_drop over block transitions
        # only.
        #
        # BOTH paths now serve the hooked entry: `ModelOutputCache` stores
        # `_last_block_raw` and accepts `replace_final_norm=True`, so the trim
        # below is dead on any cache built by the current code and survives
        # only for a cache built before that capability existed.
        #
        # HISTORICAL ONLY -- do NOT act on the paragraph below. It records why
        # the two PRE-HOOK options were both wrong, and its instruction ("do
        # not truncate the profile") describes no live code path. A reader who
        # followed it would undo the fix.
        #
        # HF's slot for block L-1 holds `ln_f(b_{L-1})`, so dropping the last
        # key DELETES the deepest block rather than recovering it. Since
        # `collapse_ratio` is a min/max over the profile and that block IS the
        # profile maximum, truncation is worse than leaving it alone: on gpt2
        # the untruncated value is **0.4662 against a hooked truth of 0.4626 --
        # 0.78% off -- while the truncated value is 0.5200, 12.4% off**, and
        # truncation collapsed gpt2 and gpt2-medium to 0.5200 / 0.5196 (0.08%
        # apart) where the truth is 12% apart (adversarial review, 2026-09-03).
        #
        # `max_drop` is the exception and the trim IS right there: a
        # block-to-LayerNorm "drop" is not a layer transition, and that entry
        # was producing max_drop = -562.65 at "layer 23" (the norm) against
        # -290.65 at layer 3. It is computed on the truncated profile below.
        _n_block_layers = collected_layer_count(model)
        erank_per_layer = []

        erank_raw_per_layer = []
        for idx in tqdm(layers, desc="Computing Effective Rank per Layer"):
            X = layer_activations[idx].float().numpy()
            # Filter NaN/Inf rows (fp16 models may produce extreme values)
            finite_mask = np.all(np.isfinite(X), axis=1)
            if not np.all(finite_mask):
                X = X[finite_mask]
            if len(X) < 5:
                # NOTE: appending 0.0 on failure silently drags min_erank to 0
                # and collapse_ratio to 0 with no error flag. Kept for shape
                # alignment; the nan_layers count below makes it visible.
                erank_per_layer.append(float("nan"))
                erank_raw_per_layer.append(float("nan"))
                continue
            # CENTRE IN DOUBLE. In float32 the subtraction itself overflows
            # near the top of the range -- a unit-max cloud scaled to 2.4e38
            # has entries whose centred values exceed 3.4e38, so `X - mean`
            # returns inf and `safe_svdvals` (correctly) refuses a non-finite
            # matrix. The overflow is in the CENTRING, not in the SVD, so
            # wiring `safe_svdvals` alone did not fix it; the same ordering
            # defect was found in `matrix_entropy` the same day.
            X = np.asarray(X, dtype=np.float64)
            X = X - np.mean(X, axis=0)

            # `safe_svdvals`, not a raw SVD.
            #
            # numpy/torch compute the singular values internally and then CAST
            # them back to the input dtype, so for float32 input near the top
            # of the range that cast overflows to `inf` and every downstream
            # consumer filters it out and returns NaN. Roy-Vetterli Property 2
            # requires erank(cA) = erank(A), and this violated it concretely:
            # on a three-layer fixture the erank profile agreed with float64
            # references to 3.64e-9 from 1e-30 through 1e25, then returned
            # NaN / 5.213422550 / NaN at 2.406159650500376e38 instead of
            # 15.276807901 / 12.719444962 / 7.806865804, reporting
            # collapse_ratio = 1.0 instead of 0.511027294 (gpt-5.6-sol round 9,
            # 2026-09-07).
            #
            # `safe_svdvals` factors out the largest magnitude first (exact --
            # it only shifts exponents) and computes in double. The helper
            # already had 28 call sites; this one was simply never wired, which
            # is why a defect that was supposedly fixed campaign-wide was still
            # live here.
            try:
                S = safe_svdvals(X)
            except Exception:
                erank_per_layer.append(float("nan"))
                erank_raw_per_layer.append(float("nan"))
                continue

            # The cited dimensional-collapse papers define the entropy over
            # the COVARIANCE spectrum, not the raw singular values. Jing et al.
            # 2110.09348 Eq. 1 works on C = (1/N) X^T X, and
            # Queipo-de-Llano et al. 2510.06477 Eq. (1) is explicit:
            # p_j = sigma_j^2 / ||X||_F^2. Feeding raw sigma (adversarial
            # review, 2026-09-02) differed by a mean factor of 145x on gpt2,
            # moved collapse_ratio from 0.0143 to 0.4228 and the argmin layer
            # from 2 to 11 -- and on pythia-410m it MASKED the very phenomenon
            # the papers are about (correct erank collapses to ~1.2 across 17 of
            # 24 layers; the raw-sigma version only dipped 827 -> 307).
            #
            # STALE-COMMENT FIX. This paragraph described a convention that no
            # longer holds twice over: RAW sigma (Roy & Vetterli Eq. (1), see
            # below) is the primary output, not the covariance spectrum, and
            # the retained sigma^2 profile is emitted as
            # `erank_sigma2_per_layer` -- there is no `*_rawsigma` key. The
            # local variable is still named erank_raw_per_layer for continuity.
            from .utils import effective_rank
            # Roy & Vetterli 2007 (EUSIPCO), Sec. 2.1 Eq. (1) + Definition 1:
            #     p_k = sigma_k / ||sigma||_1        (RAW singular values, L1)
            #     erank(A) = exp{H(p)}
            # RAW singular values, not squared. On 2026-09-02 this was changed
            # to effective_rank(S**2) on the theory that the covariance
            # spectrum was intended; a second review read the primary source
            # and that was WRONG. RankMe (Garrido et al. 2210.02885 Eq. 1-2),
            # which explicitly attributes to Roy & Vetterli, also uses raw
            # sigma. The squared convention is nobody's published erank: the
            # one paper that does normalize by ||X||_F^2 calls it "matrix-based
            # entropy" and never exponentiates.
            #
            # The mistaken change was not cosmetic -- it moved collapse_ratio
            # on gpt2 from 0.436 to 0.014 (a ~150x shift in the spectrum),
            # contradicted the paper's own equation in metrics_geometry.tex,
            # falsified task_metadata.py:50, violated the explicit contract in
            # utils.effective_rank's docstring, and FLIPPED THE SIGN of EDG
            # (-0.042 -> +0.133), which is derived from this profile.
            erank_per_layer.append(effective_rank(S))
            # The sigma^2 spectrum is retained under a name that says what it
            # is. An earlier comment here claimed geometry_matrix_entropy
            # "already reports this quantity" -- that is WRONG. Measured across
            # gpt2 layers the two are Pearson 0.343, and matrix_entropy is
            # parity-exact against the official Diff-eRank kernel (max 5.7e-5
            # over 600 model/document pairs), so they are distinct measurements
            # (adversarial review, 2026-09-03).
            erank_raw_per_layer.append(
                effective_rank(np.asarray(S, dtype=np.float64) ** 2))

        # Detect collapse: ratio of min erank to max erank
        erank_arr = np.array(erank_per_layer, dtype=np.float64)
        n_failed = int(np.isnan(erank_arr).sum())
        max_erank = float(np.nanmax(erank_arr)) if n_failed < len(erank_arr) else float("nan")
        min_erank = float(np.nanmin(erank_arr)) if n_failed < len(erank_arr) else float("nan")
        # EXACT. The last surviving absolute epsilon in the repo: a ratio of
        # two effective ranks is scale-free, and `+1e-12` biases it -- two
        # identical layers give 0.9999999999989999 instead of exactly 1.0,
        # an absolute error of 1.000089e-12 (independent Opus review,
        # 2026-09-04). Only a genuinely zero maximum is undefined.
        collapse_ratio = (min_erank / max_erank if max_erank > 0
                          else float("nan"))

        # Largest single-layer drop.
        # np.nanargmin RAISES ValueError("All-NaN slice encountered") when every
        # entry is NaN, which became reachable once failed layers started
        # recording NaN instead of 0.0. np.diff also turns ONE NaN layer into
        # TWO NaN diffs, so an all-NaN diffs array needs fewer failures than it
        # looks. Guard on the finite count, not on len().
        # max_drop EXCLUDES the post-final-norm entry (see above); the rest of
        # the profile keeps it.
        if _last_is_block:
            # Every point is a block output, so every diff is a block
            # transition: use the whole profile.
            _prof_for_drop = erank_arr
        else:
            _prof_for_drop = (erank_arr[:_n_block_layers]
                              if _n_block_layers and len(erank_arr) > _n_block_layers
                              else erank_arr)
        diffs = np.diff(_prof_for_drop)
        n_finite_diffs = int(np.count_nonzero(np.isfinite(diffs)))
        if n_finite_diffs > 0:
            max_drop = float(np.nanmin(diffs))
            max_drop_layer = int(np.nanargmin(diffs)) + 1
        else:
            max_drop = float("nan")
            max_drop_layer = -1

        # max_drop is a raw DIMENSION COUNT, so it is not comparable across
        # models of different width (-270 at d_model 768 vs -482 at 1024 is not
        # a bigger drop, just a wider model). Emit the width-normalized form
        # alongside it; erank <= d_model bounds both.
        d_model = int(layer_activations[layers[0]].shape[-1])
        max_drop_frac = max_drop / d_model if d_model else float("nan")

        return {
            "erank_per_layer": erank_per_layer,
            "max_erank": max_erank,
            "min_erank": min_erank,
            "collapse_ratio": collapse_ratio,
            "max_drop": max_drop,
            # DEMOTED to _meta_ (2026-09-04). These are /d_model twins and
            # they fail the same mechanism test that removed all eleven
            # registered normalizers: isometric zero-padding from d=8 to d=512
            # leaves max_erank / min_erank / max_drop / collapse_ratio
            # BIT-IDENTICAL (7.810457319 / 3.954113194 / -1.942833987 /
            # 0.506258857) while the /d_model forms move by a factor of 64.
            # `normalization.py` registers zero twins; these were the last
            # first-class ones still shipping (adversarial review round 2,
            # 2026-09-04).
            "_meta_max_drop_frac": max_drop_frac,
            "_meta_max_erank_frac": max_erank / d_model if d_model else float("nan"),
            "_meta_min_erank_frac": min_erank / d_model if d_model else float("nan"),
            # Raw-singular-value spectrum, retained for continuity with the
            # pre-2026-09 collection. NOT the papers' quantity.
            "erank_sigma2_per_layer": erank_raw_per_layer,
            # Per-layer WIDTH-NORMALIZED profile. The aggregator reduces
            # erank_per_layer to .{mean,q75,slope} columns, which were raw
            # dimension counts and so scaled with d_model; these do not.
            "_meta_erank_per_layer_frac": [
                (float(v) / d_model if (d_model and np.isfinite(v)) else float("nan"))
                for v in erank_arr
            ],
            "_meta_max_drop_layer": max_drop_layer,
            "_meta_n_failed_layers": n_failed,
            # Provenance: True = deepest point is the hooked raw block output;
            # False = cache path, deepest point is ln_f(b_{L-1}).
            "_meta_last_entry_is_block_output": bool(_last_is_block),
            # `max_drop` is an unclamped nanmin over diffs, so on a
            # monotonically INCREASING profile it reports the smallest rise
            # under a name that says drop. Latent on every real model measured,
            # but flag it rather than let the name lie.
            "_meta_max_drop_is_actually_increase": bool(
                np.isfinite(max_drop) and max_drop > 0),
        }
