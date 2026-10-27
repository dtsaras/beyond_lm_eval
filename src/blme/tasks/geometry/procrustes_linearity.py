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
    * One ``(N, D)`` token cloud per layer with a shared token order (one
      forward pass), from the shared cache or a private ``ModelOutputCache``.
      The layer set starts at the embedding output and ends at the raw
      deepest block output, so all L block transitions are scored (the
      notebook skips the last one only because HF's last entry is post-norm).
    * ``_procrustes_similarity`` ports the reference in float64 (parity 0.0
      against the reference evaluated in float64,
      tests/tasks/parity/test_procrustes_linearity_parity.py); its only added
      guard is NaN for a constant or exactly singular cloud.
    * Conditioning: the unguarded ``1 / S`` amplifies near-zero singular
      values, so on a rank-deficient cloud the reference rule depends on how
      the SVD resolved the null space. ``rank_aware=True`` applies the
      Moore-Penrose cutoff at numpy's pinv tolerance.
    * Only summary scalars are emitted; the slope is on normalised depth.
"""

import logging

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from .._tokens import hidden_states_for_task

logger = logging.getLogger("blme")



def _procrustes_similarity(X, Y, rank_aware: bool = False, return_rank: bool = False):
    """Linearity score of Razzhigaev et al. 2024 (arXiv:2405.12250).

    Port of ``llm_microscope.procrustes_similarity`` (b6db939 / pip 0.0.7)
    evaluated in float64, with the same torch SVD (numpy's resolves the null
    space differently):

        center columns -> Frobenius-normalize ->
        A = Vh^T diag(1/S) U^T Y  (== X^+ Y, unconstrained linear map) ->
        sim = 1 - ||X A - Y||_F^2   (||Y||_F = 1 after normalization).

    Args:
        X: ``(N, D)`` token cloud at layer l.
        Y: ``(N, D)`` token cloud at layer l+1, same token order.
        rank_aware: invert only singular values above numpy's pinv tolerance
            (Moore-Penrose); otherwise every positive one, as the reference.
        return_rank: with ``rank_aware``, return ``(sim, r)`` where r is the
            numerical rank of the centred X at that tolerance.

    Returns:
        Linearity in (-inf, 1]; NaN for degenerate (constant / mismatched /
        too-small) clouds.
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
        # Exact zero only: the score is scale-free, so an absolute floor is not.
        if float(nx) <= 0.0 or float(ny) <= 0.0:
            return float("nan")
        Xc = Xc / nx
        Yc = Yc / ny

        # 3. Unconstrained linear map A = X^+ Y via the thin SVD, then Y_est.
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        if S.numel() == 0:
            return float("nan")
        if rank_aware:
            # Before the singularity guard: an exactly rank-deficient X is the
            # case this branch exists for.
            tol = float(S[0]) * max(Xc.shape) * torch.finfo(Xc.dtype).eps
            Sinv = torch.where(S > tol, 1.0 / S, torch.zeros_like(S))
            A_r = Vh.T * Sinv[None, ...] @ U.T @ Yc
            sim_r = float(1.0 - (Xc @ A_r - Yc).square().sum())
            if return_rank:
                return sim_r, int((S > tol).sum())
            return sim_r
        if float(S[-1]) <= 0.0:
            # Exactly singular X: 1/S would be inf; NaN instead.
            return float("nan")
        A = Vh.T * (1.0 / S)[None, ...] @ U.T @ Yc     # (D, D)
        Y_est = Xc @ A

        # 4. Normalized residual as similarity (||Y||_F = 1 => already norm.).
        y_error = (Y_est - Yc).square().sum()
        sim = float(1.0 - y_error)
        return sim


def _adjusted_linearity(sim: float, n: int, r: int) -> float:
    """Adjusted R^2 of the fit (Wherry 1931): 1 - (1 - sim)(N - 1)/(N - 1 - r).

    Least squares on r regressors explains r/(N-1) of an independent centred
    target by chance; this removes exactly that term. NaN when N - 1 <= r.
    """
    dof = n - 1 - r
    if not np.isfinite(sim) or dof <= 0:
        return float("nan")
    return float(1.0 - (1.0 - sim) * (n - 1) / dof)


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

    The fit is the Moore-Penrose least-squares map X^+ Y with numpy's pinv
    cutoff. The released ``get_est_svd`` inverts every positive singular
    value; on a numerically rank-deficient cloud that amplifies ~1e-15
    values, so an exact isometric embedding moved it 0.816 -> 0.694 on gpt2
    (dossier). That is a numerical artefact, not the paper's least-squares
    definition (Q2), so the released rule is kept only as ``_meta_``; on real
    full-rank pairs the two agree exactly.

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

        if dataset is None and not (cache is not None and use_cache
                                    and getattr(cache, "is_populated", False)):
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)
        # Embedding output first, raw deepest block last: all L transitions.
        layer_states, _info = hidden_states_for_task(
            model, tokenizer, dataset, cache, num_samples=num_samples,
            replace_final_norm=True, layer_idx="all", use_cache=use_cache,
            include_embedding=True, fallback="private_cache")

        if not layer_states:
            return {"error": "No hidden states available for procrustes linearity"}

        layers = sorted(layer_states.keys())
        n_layers = len(layers)
        if n_layers < 2:
            return {"error": "Need at least 2 layers for procrustes linearity"}

        n_tokens = 0
        sims = []          # headline: rank-aware least-squares fit
        sims_ref = []      # released get_est_svd rule (unguarded 1/S)
        sims_adj = []      # adjusted-R^2 twin of the headline
        ranks = []
        for a, b in zip(layers[:-1], layers[1:]):
            X = layer_states[a]
            Y = layer_states[b]
            if isinstance(X, torch.Tensor):
                # numpy then float64: never narrow a float64 input to float32.
                X = X.detach().cpu().numpy()
            if isinstance(Y, torch.Tensor):
                Y = Y.detach().cpu().numpy()
            X = np.asarray(X, dtype=np.float64)
            Y = np.asarray(Y, dtype=np.float64)
            if X.ndim != 2 or Y.ndim != 2 or X.shape != Y.shape:
                sims.append(float("nan"))
                sims_ref.append(float("nan"))
                sims_adj.append(float("nan"))
                continue
            n_tokens = max(n_tokens, X.shape[0])
            sims_ref.append(_procrustes_similarity(X, Y))
            _fit = _procrustes_similarity(X, Y, rank_aware=True, return_rank=True)
            _s, _r = _fit if isinstance(_fit, tuple) else (_fit, X.shape[0])
            sims.append(_s)
            sims_adj.append(_adjusted_linearity(_s, X.shape[0], _r))
            ranks.append(_r)

        sim = np.asarray(sims, dtype=np.float64)
        _sref = np.asarray(sims_ref, dtype=np.float64)
        finite = np.isfinite(sim)
        if not np.any(finite):
            return {"error": "No adjacent layer pair yielded a valid linearity score"}

        n_pairs = sim.size
        finite_sim = sim[finite]
        _d_model = int(layer_states[layers[0]].shape[-1]) if n_tokens else 0
        _d_over_n = (float(_d_model) / float(n_tokens)) if n_tokens else float("nan")

        out = _profile_summary(sim, "procrustes_linearity")
        # Size-free twin: the chance fit r/(N-1) removed pair by pair (the
        # slope's twin is not defined; test_procrustes_linearity.py N test).
        _adj = _profile_summary(np.asarray(sims_adj, dtype=np.float64),
                                "procrustes_linearity_adjusted")
        _adj.pop("procrustes_linearity_adjusted_slope")
        out.update(_adj)
        out["_meta_fit_rank_max"] = int(max(ranks)) if ranks else 0
        _ref_fin = _sref[np.isfinite(_sref)]
        out.update({
            "_meta_n_layers": int(n_layers),
            "_meta_n_pairs": int(n_pairs),
            "_meta_n_tokens": int(n_tokens),
            # Ambient-d correction, _meta_ only: zero-padding (an exact
            # isometry) moves it, because the chance term follows the fitted
            # rank, not d.
            "_meta_procrustes_nonlinearity_corrected_mean": (
                float((1.0 - np.mean(finite_sim)) / (1.0 - _d_over_n))
                if 0.0 < _d_over_n < 1.0 else float("nan")),
            "_meta_procrustes_nonlinearity_corrected_max": (
                float((1.0 - np.min(finite_sim)) / (1.0 - _d_over_n))
                if 0.0 < _d_over_n < 1.0 else float("nan")),
            # The score moves with N at fixed model, so record the regime.
            "_meta_d_model": int(_d_model),
            "_meta_d_over_n_tokens": float(_d_over_n),
            # The released rule, kept for parity: it equals the headline
            # unless a cloud is numerically rank-deficient (gap reported).
            "_meta_reference_rule_procrustes_linearity_mean": (
                float(np.mean(_ref_fin)) if _ref_fin.size else float("nan")),
            "_meta_reference_rule_procrustes_linearity_min": (
                float(np.min(_ref_fin)) if _ref_fin.size else float("nan")),
            "_meta_procrustes_rank_aware_gap_max": (
                float(np.nanmax(np.abs(_sref - sim)))
                if np.any(np.isfinite(_sref) & finite) else float("nan")),
        })
        return out


def _profile_summary(sim: np.ndarray, prefix: str) -> dict:
    """mean/min/max/first/last/slope/quartiles over the adjacent-pair profile;
    the slope is on normalised depth l/(n_pairs-1)."""
    finite = np.isfinite(sim)
    if not np.any(finite):
        return {f"{prefix}_{k}": float("nan") for k in
                ("mean", "min", "max", "first", "last", "slope", "q25", "q50", "q75")}
    fs = sim[finite]
    n_pairs = sim.size
    if n_pairs >= 2 and finite.sum() >= 2:
        depth = np.arange(n_pairs, dtype=np.float64) / (n_pairs - 1)
        slope = float(np.polyfit(depth[finite], sim[finite], 1)[0])
    else:
        slope = float("nan")
    return {
        f"{prefix}_mean": float(np.mean(fs)),
        f"{prefix}_min": float(np.min(fs)),
        f"{prefix}_max": float(np.max(fs)),
        f"{prefix}_first": float(sim[0]) if np.isfinite(sim[0]) else float("nan"),
        f"{prefix}_last": float(sim[-1]) if np.isfinite(sim[-1]) else float("nan"),
        f"{prefix}_slope": slope,
        f"{prefix}_q25": float(np.percentile(fs, 25)),
        f"{prefix}_q50": float(np.percentile(fs, 50)),
        f"{prefix}_q75": float(np.percentile(fs, 75)),
    }
