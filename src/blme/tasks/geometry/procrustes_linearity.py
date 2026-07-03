"""Procrustes linearity — is each transformer block a near-linear map?

Reference (official):
    Razzhigaev, Mikhalchuk, Goncharova, Gerasimenko, Oseledets, Dimitrov,
    Kuznetsov (2024). "Your Transformer is Secretly Linear."
    ACL 2024 (long), arXiv:2405.12250.
    Official repo: AIRI-Institute/LLM-Microscope (commit b6db939), pip
    package ``llm-microscope`` (v0.0.7).

Metric definition (verified against the reference code — quoted below):
    Their "linearity score" for a pair of consecutive-layer embedding
    clouds ``X`` (layer l) and ``Y`` (layer l+1), each ``(N, D)`` with a
    SHARED token order across the two layers, is:

      1. Center each cloud by its column (feature) mean:
             X = x - x.mean(0);  Y = y - y.mean(0)
      2. Normalize each by its Frobenius norm:
             X = X / ||X||_F;    Y = Y / ||Y||_F
      3. Fit the best UNCONSTRAINED linear map A that sends X -> Y in the
         least-squares sense, via the Moore-Penrose pseudo-inverse of X:
             X = U S Vh  (thin SVD)
             A = Vh^T diag(1/S) U^T Y   (== X^+ Y)
             Y_est = X A
      4. Report the normalized residual as a similarity:
             sim = 1 - ||Y_est - Y||_F^2

    Because X, Y are Frobenius-normalized, ||Y||_F^2 = 1, so this is
    exactly the normalized-residual "linearity" in [-inf, 1], reaching
    1.0 when Y lies perfectly in the column space of X (i.e. Y = X A for
    some linear A). The paper reports these values are ~0.99 for adjacent
    transformer layers ("secretly linear").

    IMPORTANT — the transform is NOT orthogonal Procrustes. Despite the
    function name ``procrustes_similarity`` in the repo, ``get_est_svd``
    forms the pseudo-inverse solution A = X^+ Y (a general linear map),
    not the orthogonal A = U V^T from the SVD of X^T Y. Reference code
    (LLM_microscope.ipynb cell 3 / llm_microscope/functions.py:86-115):

        def get_est_svd(X, Y):
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            A_estimation = Vh.T * (1 / S)[None, ...] @ U.T @ Y  # Y=XA
            Y_est =  X @ A_estimation
            return Y_est

        def procrustes_similarity(x, y):
            with torch.no_grad():
                X = x - x.mean(dim=0, keepdim=True)
                Y = y - y.mean(dim=0, keepdim=True)
                X = X / X.norm()
                Y = Y / Y.norm()
                Y_estimation = get_est_svd(X, Y)
                y_error = (Y_estimation - Y).square().sum()
                sim = float(1 - y_error)
            return sim

    (The repo also ships ``procrustes_similarity_centered(x, y0)`` which
    replaces ``y`` with the residual ``y0 - x`` before the same pipeline;
    that "residual linearity" variant is intentionally NOT reproduced
    here — this task pins the headline ``procrustes_similarity``.)

Implementation notes (BLME):
    * Tier 2 — consumes the shared cache via
      ``cache.get_hidden_states(layer_idx="all", per_sample=False)``,
      which returns one ``(N, D)`` token cloud per layer with a MATCHED
      token order across layers (same forward pass, same flatten order),
      exactly the synchronized collection CKA relies on. Adjacent layers
      therefore share token order, which is required for a meaningful
      per-token linear map. Without a cache the task runs its own forward
      pass through a private ``ModelOutputCache``.
    * The core ``_procrustes_similarity`` is a BIT-EXACT port: it runs the
      SAME ``torch.linalg.svd`` + unguarded ``1 / S`` as the reference
      (promoted to float64 for determinism) and matches it to < 1e-6 on a
      shared toy pair (see the parity test). BLME adds exactly ONE guard —
      returning NaN for a degenerate constant / exactly-singular cloud
      where the reference would emit NaN/inf. It deliberately does NOT use
      numpy's SVD or an rcond-truncated pinv: those are equally-valid least
      squares but resolve the null space differently, so they would NOT
      reproduce the reference number on real, ill-conditioned clouds.
    * CONDITIONING CAVEAT (publication-relevant, verified on real layer
      clouds): when X is near-rank-deficient — token clouds whose effective
      rank is well below D, i.e. N >~ D but with ~epsilon-tiny trailing
      singular values — the score is dominated by how the SVD backend
      resolves those ~1e-17 singular values, because the unguarded
      ``1 / S`` amplifies them by ~1e17. In this regime numpy and torch SVD
      disagree by up to ~0.4 on the SAME input, and a numerically-stable
      solver returns ~1.0 (Y is truly in X's column space). The reference —
      and therefore BLME — reports torch's value, deterministic per torch
      build but an artifact of the conditioning, not a robust linearity
      measure. The metric is only well-posed when X is reasonably
      conditioned (the paper's toy example is (1000, 10); their real runs
      use ~512-token clouds with low intrinsic dimension). Interpret the
      absolute value with care; the across-depth PROFILE is the signal.
    * evaluate() computes the similarity for every adjacent layer pair
      (l, l+1) and summarizes the straightening / linearity profile:
      mean/min/max over depth, plus the OLS slope on NORMALIZED depth
      l/(n_pairs-1) (house convention for cross-model comparability).
    * Only summary scalars are emitted (no per-pair list) so the
      aggregator never sees absolute layer indices; ``_meta_`` counts are
      excluded from the feature matrix.
"""

import logging

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")

# Frobenius norms below this mean a degenerate (constant) cloud after
# centering; the pair is skipped rather than producing NaN/inf. This is the
# ONLY guard BLME adds: everything else reproduces the reference verbatim.
_ZERO_NORM_EPS = 1e-12


def _procrustes_similarity(X, Y) -> float:
    """Linearity score of Razzhigaev et al. 2024 (arXiv:2405.12250).

    Bit-exact port of ``llm_microscope.procrustes_similarity`` (repo commit
    b6db939 / pip v0.0.7). The reference is torch-native and its value in
    the near-rank-deficient regime is determined by how ``torch.linalg.svd``
    resolves the tiny singular values that the unguarded ``1 / S`` then
    amplifies — so BLME runs the SAME torch SVD (promoted to float64 for
    determinism) rather than numpy, whose SVD backend resolves the null
    space differently and would NOT reproduce the reference number (see the
    parity report / test: numpy and torch disagree by up to ~0.4 on real,
    ill-conditioned layer clouds even though both are "correct" least
    squares; the reference is defined by torch).

        center columns -> Frobenius-normalize (x / x.norm(), Frobenius) ->
        A = Vh^T diag(1/S) U^T Y  (== X^+ Y, unconstrained linear map) ->
        sim = 1 - ||X A - Y||_F^2   (||Y||_F = 1 after normalization).

    Args:
        X: array-like / tensor ``(N, D)`` — token cloud at layer l.
        Y: array-like / tensor ``(N, D)`` — token cloud at layer l+1, SAME
           token order (row i of X and Y is the same token position).

    Returns:
        Float linearity in (-inf, 1]; ~1.0 => the layer transition is
        near-linear. Returns NaN for degenerate (constant / mismatched /
        too-small) clouds. See the CONDITIONING CAVEAT in the module
        docstring: the score is only well-posed when X is reasonably
        conditioned (low effective rank relative to N); this matches the
        reference exactly, artifact and all.
    """
    x = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(Y), dtype=torch.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape or x.shape[0] < 2:
        return float("nan")

    with torch.no_grad():
        # 1. Center columns (subtract per-feature mean).
        Xc = x - x.mean(dim=0, keepdim=True)
        Yc = y - y.mean(dim=0, keepdim=True)

        # 2. Frobenius-normalize each cloud (default torch.norm == Frobenius).
        nx = Xc.norm()
        ny = Yc.norm()
        if float(nx) <= _ZERO_NORM_EPS or float(ny) <= _ZERO_NORM_EPS:
            return float("nan")
        Xc = Xc / nx
        Yc = Yc / ny

        # 3. Unconstrained linear map A = X^+ Y via the thin SVD, then Y_est.
        #    Reproduces get_est_svd verbatim (unguarded 1 / S — matching the
        #    reference; a genuinely singular S is caught below).
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        if S.numel() == 0 or float(S[-1]) <= 0.0:
            # Exactly-singular X: 1/S would be inf. The reference would
            # divide by zero here; BLME reports NaN instead of NaN/inf
            # propagating silently. (Does not occur for real token clouds.)
            return float("nan")
        A = Vh.T * (1.0 / S)[None, ...] @ U.T @ Yc     # (D, D)
        Y_est = Xc @ A

        # 4. Normalized residual as similarity (||Y||_F = 1 => already norm.).
        y_error = (Y_est - Yc).square().sum()
        return float(1.0 - y_error)


@register_task("geometry_procrustes_linearity")
class ProcrustesLinearityTask(DiagnosticTask):
    """
    Computes the Razzhigaev et al. (ACL 2024, arXiv:2405.12250) linearity
    (a.k.a. "procrustes") score for every adjacent layer pair and
    summarizes the linearity profile across depth.

    The paper's headline finding is that transformer blocks are "secretly
    linear": consecutive-layer embeddings are related by a near-perfect
    linear transform (scores ~0.99), and the profile of this linearity
    across depth is a fingerprint of the architecture / training.

    Outputs (flat floats):
        procrustes_linearity_mean   — mean over adjacent pairs
        procrustes_linearity_min    — least-linear transition
        procrustes_linearity_max    — most-linear transition
        procrustes_linearity_first  — pair (0, 1)
        procrustes_linearity_last   — pair (L-2, L-1)
        procrustes_linearity_slope  — OLS slope on normalized depth
        procrustes_linearity_q25/q50/q75 — quartiles over pairs
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Procrustes Linearity Analysis...")

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)

        if cache is not None and cache.is_populated and use_cache:
            layer_states = cache.get_hidden_states(
                layer_idx="all", num_samples=num_samples, per_sample=False,
            )
        else:
            from ...cache import ModelOutputCache, load_default_corpus

            if dataset is None:
                dataset = load_default_corpus(num_samples)
            local_cache = ModelOutputCache(
                model, tokenizer, dataset=dataset, num_samples=num_samples,
            )
            local_cache.populate(need_hidden=True)
            layer_states = local_cache.get_hidden_states(
                layer_idx="all", num_samples=num_samples, per_sample=False,
            )

        if not layer_states:
            return {"error": "No hidden states available for procrustes linearity"}

        layers = sorted(layer_states.keys())
        n_layers = len(layers)
        if n_layers < 2:
            return {"error": "Need at least 2 layers for procrustes linearity"}

        n_tokens = 0
        sims = []
        for a, b in zip(layers[:-1], layers[1:]):
            X = layer_states[a]
            Y = layer_states[b]
            if isinstance(X, torch.Tensor):
                X = X.detach().float().cpu().numpy()
            if isinstance(Y, torch.Tensor):
                Y = Y.detach().float().cpu().numpy()
            X = np.asarray(X, dtype=np.float64)
            Y = np.asarray(Y, dtype=np.float64)
            if X.ndim != 2 or Y.ndim != 2 or X.shape != Y.shape:
                sims.append(float("nan"))
                continue
            n_tokens = max(n_tokens, X.shape[0])
            sims.append(_procrustes_similarity(X, Y))

        sim = np.asarray(sims, dtype=np.float64)
        finite = np.isfinite(sim)
        if not np.any(finite):
            return {"error": "No adjacent layer pair yielded a valid linearity score"}

        n_pairs = sim.size
        # Slope on NORMALIZED depth l/(n_pairs-1) — house convention for
        # cross-model comparability.
        if n_pairs >= 2 and finite.sum() >= 2:
            depth = np.arange(n_pairs, dtype=np.float64) / (n_pairs - 1)
            slope = float(np.polyfit(depth[finite], sim[finite], 1)[0])
        else:
            slope = float("nan")

        finite_sim = sim[finite]
        first = sim[0] if np.isfinite(sim[0]) else float("nan")
        last = sim[-1] if np.isfinite(sim[-1]) else float("nan")
        return {
            "procrustes_linearity_mean": float(np.mean(finite_sim)),
            "procrustes_linearity_min": float(np.min(finite_sim)),
            "procrustes_linearity_max": float(np.max(finite_sim)),
            "procrustes_linearity_first": float(first),
            "procrustes_linearity_last": float(last),
            "procrustes_linearity_slope": slope,
            "procrustes_linearity_q25": float(np.percentile(finite_sim, 25)),
            "procrustes_linearity_q50": float(np.percentile(finite_sim, 50)),
            "procrustes_linearity_q75": float(np.percentile(finite_sim, 75)),
            # _meta_ prefix => excluded from the analysis feature matrix so
            # these architecture/sampling counts cannot leak in as size
            # proxies (Audit-V2).
            "_meta_n_layers": int(n_layers),
            "_meta_n_pairs": int(n_pairs),
            "_meta_n_tokens": int(n_tokens),
        }
