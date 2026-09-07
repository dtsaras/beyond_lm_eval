from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collected_layer_count
from .utils import collect_hidden_states
import torch
import numpy as np
from tqdm import tqdm
import logging
logger = logging.getLogger("blme")


def _centered_linear_gram(X: torch.Tensor) -> torch.Tensor:
    """Linear kernel Gram matrix centered by H K H.

    Centering is applied to X BEFORE the Gram is formed. HKH(XX^T) equals
    (HX)(HX)^T exactly in real arithmetic, but not in floating point: forming
    XX^T first squares any common offset into the entries and then subtracts
    it back, by which point the signal is gone. A fixed pair with exact CKA
    0.105577850292 returned 0.131100457707 at offset 1e3 (+24.17%) and
    -0.000873075136 at 1e4 under the old order (adversarial review,
    2026-09-04). float64 for the same reason -- the Gram is a sum over the
    full width.
    """
    X = X.double()
    X = X - X.mean(dim=0, keepdim=True)
    return X @ X.t()


def _nhsic_from_grams(K_i: torch.Tensor, K_j: torch.Tensor,
                      self_i: "float | None" = None,
                      self_j: "float | None" = None) -> float:
    """Normalized HSIC from (centered) gram matrices:
    tr(K_i K_j) / sqrt(tr(K_i K_i) tr(K_j K_j)). This is THE production
    kernel — evaluate() and the parity-pinned feature-space wrapper both
    call it, so the reference-value parity test covers the shipped path
    (previously evaluate() duplicated the formula inline and only the
    unused wrapper was pinned)."""
    hsic_ij = float(torch.sum(K_i * K_j))
    if self_i is None:
        self_i = float(torch.sum(K_i * K_i))
    if self_j is None:
        self_j = float(torch.sum(K_j * K_j))
    # Kornblith's CKA is invariant to isotropic rescaling of either view; a
    # fixed +1e-12 in the denominator is not. A nondegenerate pair scored
    # 0.132094427492 at unit scale but 0.101702161272 at 1e-4 (-23.01%) and
    # 4.41881768414e-5 at 1e-5 (-99.97%) (adversarial review, 2026-09-04).
    # Only an exactly-zero self term is undefined.
    # sqrt(a)*sqrt(b), NOT sqrt(a*b).
    #
    # The Gram entries are already squared, and the self terms square them
    # again, so the PRODUCT of two self terms overflows float64 long before
    # either one does: at input magnitude 2.4e38 each self term is ~1.2e157 and
    # their product ~1.5e314 is `inf`, which made CKA return exactly 0.0 for a
    # perfectly ordinary pair (found by extending the scale-invariance contract
    # to the float32 boundary, 2026-09-05). Multiplying the roots is
    # algebraically identical and has half the dynamic range.
    denom = np.sqrt(self_i) * np.sqrt(self_j)
    if not (denom > 0.0) or not np.isfinite(denom):
        return float("nan")
    return float(hsic_ij / denom)


def _normalized_linear_hsic(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Normalized linear HSIC, equivalent to linear CKA."""
    return _nhsic_from_grams(_centered_linear_gram(X), _centered_linear_gram(Y))


@register_task("geometry_hsic")
class HSICDependenceTask(DiagnosticTask):
    """
    Measures statistical dependence between layer representations using
    normalized HSIC (Hilbert-Schmidt Independence Criterion) with a linear
    kernel. Mathematically equivalent to Linear CKA from Kornblith 2019
    when normalised by √(HSIC(X,X)·HSIC(Y,Y)).

    References:
      * Gretton, Bousquet, Smola, Schölkopf 2005 — "Measuring Statistical
        Dependence with Hilbert-Schmidt Norms", ALT 2005. The original
        HSIC formulation used here.
      * Kornblith, Norouzi, Lee, Hinton 2019 — "Similarity of Neural
        Network Representations Revisited", ICML 2019, arXiv:1905.00414.
        HSIC → CKA normalisation in Section 3.

    ``input_to_layer_hsic.mean`` enters BLME's top-24 partial predictors
    at +0.68 partial ρ (`docs/TOP_PREDICTORS.md` §2), measuring how
    strongly per-layer activations still depend on the FIRST TRANSFORMER
    BLOCK's output (the shared cache drops the embedding output hs[0],
    so layer 0 here is block-1 output, not the input embedding).
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running HSIC Dependence Analysis...")

        if dataset is None:
            dataset = [
                {"text": "The quick brown fox jumps over the lazy dog."}
                for _ in range(50)
            ]

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)

        # Collect all layer activations
        if cache is not None and cache.is_populated and use_cache:
            layer_activations = cache.get_hidden_states(
                layer_idx="all", num_samples=num_samples, replace_final_norm=True)
            # Derived from whether the swap ACTUALLY fired, not from the
            # mere presence of a hooked tensor (a shape mismatch is
            # rejected inside get_hidden_states).
            _last_is_block = bool(getattr(cache, "_last_block_substituted", False))
        else:
            _st = {}
            layer_activations = collect_hidden_states(
                model, tokenizer, dataset, num_samples=num_samples,
                layer_idx="all", replace_final_norm=True, status=_st)
            _last_is_block = bool(_st.get("last_is_block", False))

        layers = sorted(layer_activations.keys())
        # RECOVER the deepest block rather than DELETING it.
        #
        # HF's last `hidden_states` slot holds `ln_f(b_{L-1})`, so the old trim
        # threw away block L-1 and left only L-2 block-to-block transitions
        # (4/10/22/34 at L=6/12/24/36) -- and `min_adjacent_hsic` and the
        # numerator of `hsic_compression_ratio` both live at the deep end.
        # A forward hook on `layers[-1]` returns the raw pre-norm output
        # instead, validated as ln_f(hooked) == hs[-1] to 0.0e+00. The trim
        # below now only fires for a cache built before that capability
        # existed (adversarial review, 2026-09-04).
        if not _last_is_block:
            _keep = collected_layer_count(model)
            if _keep and len(layers) > _keep:
                layers = layers[:_keep]
        n_layers = len(layers)

        if n_layers < 2:
            return {"error": "Need at least 2 layers"}

        # Subsample tokens for speed — use SAME indices across all layers
        max_tokens = self.config.get("max_hsic_tokens", 2000)
        n_tokens = layer_activations[layers[0]].shape[0]
        if n_tokens > max_tokens:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(int(self.config.get("seed", 42)))
            shared_perm = torch.randperm(n_tokens, generator=rng)[:max_tokens]
        else:
            shared_perm = None

        # Precompute centered Gram matrices for each layer
        logger.info("  Computing Gram matrices...")
        gram_matrices = {}
        for idx in tqdm(layers, desc="Gram Matrices"):
            X = layer_activations[idx].float()
            if shared_perm is not None:
                X = X[shared_perm]

            gram_matrices[idx] = _centered_linear_gram(X).cpu()

        # Compute HSIC between pairs of layers
        # HSIC(X, Y) = (1/(n-1)^2) * trace(K_X @ K_Y)
        # Normalized HSIC (CKA-like): HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
        logger.info("  Computing pairwise HSIC...")

        # Self-HSIC for normalization
        self_hsic = {}
        for idx in layers:
            K = gram_matrices[idx]
            self_hsic[idx] = float(torch.sum(K * K))

        # Adjacent layer HSIC (shared, parity-pinned kernel).
        adjacent_hsic = []
        for i in range(n_layers - 1):
            adjacent_hsic.append(_nhsic_from_grams(
                gram_matrices[layers[i]], gram_matrices[layers[i + 1]],
                self_hsic[layers[i]], self_hsic[layers[i + 1]],
            ))

        # Input-to-layer HSIC (first layer vs every *later* layer).
        # Historic bug: the loop started at ``i = 0``, including the
        # self-pair ``HSIC(input, input) = 1.0``. That self-pair had no
        # information content but pinned the list's ``max`` to 1.0 for
        # every model and skewed ``hsic_compression_ratio`` by anchoring
        # the denominator at a tautological 1.
        input_hsic = []
        K_input = gram_matrices[layers[0]]
        for i in range(1, n_layers):
            input_hsic.append(_nhsic_from_grams(
                K_input, gram_matrices[layers[i]],
                self_hsic[layers[0]], self_hsic[layers[i]],
            ))

        # RAW (uncentred-normalisation-free) HSIC endpoints for the
        # manuscript's ratio: tr(K_0 K_{L-1}) and tr(K_0 K_0) on the CENTRED
        # Gram matrices, which is exactly Gretton's biased HSIC up to the
        # common 1/(n-1)^2 factor that cancels in the ratio.
        _K0 = gram_matrices[layers[0]]
        _KL = gram_matrices[layers[-1]]
        _raw_hsic_first_last = float(torch.sum(_K0 * _KL))
        _raw_hsic_first_self = float(torch.sum(_K0 * _K0))

        return {
            "adjacent_hsic": adjacent_hsic,
            "avg_adjacent_hsic": float(np.mean(adjacent_hsic)),
            "min_adjacent_hsic": float(np.min(adjacent_hsic)),
            "input_to_layer_hsic": input_hsic,
            # NAMED for what it is, and NOT claimed to be bounded.
            #
            # The manuscript (sections/metrics_geometry.tex:227-238) defines
            # this as HSIC(0, L-1) / HSIC(0, 0) and states a [0, 1] range.
            # What the code computes is nCKA(anchor, L-1) / nCKA(anchor, 0) --
            # the anchor is the FIRST BLOCK's output, never the self-pair,
            # because the shared cache drops the embedding entry. Those are
            # different quantities, and the ratio is genuinely unbounded: a
            # three-layer [X, Y, X] stack emits 9.401857715774 (adversarial
            # review round 2, 2026-09-04). Renamed so no reader takes the
            # bound on faith; the denominator epsilon is gone with it.
            # THE PAPER TEXT MUST BE RECONCILED (tracked in RERUN_LEDGER.md).
            "hsic_deep_to_shallow_ratio": (
                float(input_hsic[-1] / input_hsic[0])
                if input_hsic and input_hsic[0] != 0.0 else float("nan")
            ),
            # THE MANUSCRIPT'S QUANTITY, computed from RAW HSIC.
            #
            # The round-3 note below was wrong, and an independent review
            # caught it: substituting normalised CKA for HSIC does NOT give
            # the manuscript's ratio. On [X, Y, 3X] the manuscript's formula
            # gives 8.999999999999998 while the CKA form gives 1.000000000000,
            # and across four real models the CKA form (0.1126-0.3448) ORDERS
            # THE MODELS EXACTLY BACKWARDS against the manuscript's
            # (3.4176-10.4108) -- Spearman -1.0 (independent Opus review,
            # 2026-09-04). HSIC(0, L-1)/HSIC(0, 0) = tr(K_0 K_{L-1}) /
            # tr(K_0 K_0), which is a projection coefficient and is NOT bounded
            # by 1; the manuscript's "[0, 1]" claim is false FOR ITS OWN
            # FORMULA, and that is a paper correction, not a code one.
            "hsic_compression_ratio_raw": (
                float(_raw_hsic_first_last / _raw_hsic_first_self)
                if _raw_hsic_first_self else float("nan")),
            "hsic_compression_ratio": (
                float(input_hsic[-1]) if input_hsic else float("nan")
            ),
        }
