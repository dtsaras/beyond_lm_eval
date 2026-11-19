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
    * Rows: the headline drops position 0 and the massive-activation rows
      (``TOKEN_POLICY``), one mask per token shared by every layer. Those rows
      hold most of the centred sum of squares, so with them the fit scores the
      sink and not the content tokens. The released notebook keeps every
      position; that value is emitted as ``_meta_incl_pos0_*``.
    * An additional BLME score selects a truncated-SVD affine map using
      training documents and measures predictive R-squared against the training
      target mean on unseen documents. The splits, cutoffs and ranks are recorded.
"""

import logging

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_num_layers
from .utils import as_numpy
from .._tokens import (hidden_states_for_task, massive_token_share, middle_depth_layers,
                       position0_share, select_rows, sink_keep_masks, sink_meta)

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


def _pair_profile(layer_states, layers, reference_rule: bool = True):
    """Per adjacent pair: rank-aware sim, released-rule sim, adjusted twin,
    fit rank; plus the largest token count seen."""
    n_tokens = 0
    sims, sims_ref, sims_adj, ranks = [], [], [], []
    for a, b in zip(layers[:-1], layers[1:]):
        X = layer_states[a]
        Y = layer_states[b]
        if isinstance(X, torch.Tensor):
            # numpy then float64: never narrow a float64 input to float32.
            X = as_numpy(X)
        if isinstance(Y, torch.Tensor):
            Y = as_numpy(Y)
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.ndim != 2 or Y.ndim != 2 or X.shape != Y.shape:
            sims.append(float("nan"))
            sims_ref.append(float("nan"))
            sims_adj.append(float("nan"))
            continue
        n_tokens = max(n_tokens, X.shape[0])
        sims_ref.append(_procrustes_similarity(X, Y) if reference_rule else float("nan"))
        _fit = _procrustes_similarity(X, Y, rank_aware=True, return_rank=True)
        _s, _r = _fit if isinstance(_fit, tuple) else (_fit, X.shape[0])
        sims.append(_s)
        sims_adj.append(_adjusted_linearity(_s, X.shape[0], _r))
        ranks.append(_r)
    return sims, sims_ref, sims_adj, ranks, n_tokens


def _adjusted_linearity(sim: float, n: int, r: int) -> float:
    """Adjusted R^2 of the fit (Wherry 1931): 1 - (1 - sim)(N - 1)/(N - 1 - r).

    Least squares on r regressors explains r/(N-1) of an independent centred
    target by chance; this removes exactly that term. NaN when N - 1 <= r.
    """
    dof = n - 1 - r
    if not np.isfinite(sim) or dof <= 0:
        return float("nan")
    return float(1.0 - (1.0 - sim) * (n - 1) / dof)


_SI_NODE = ("tests/tasks/invariance/test_procrustes_linearity.py::"
            "test_size_invariant_entry[{}]")


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

    Rows follow ``TOKEN_POLICY``: position 0 and the massive-activation rows
    are dropped with one mask per token, taken from the middle block outputs
    and shared by every layer. The every-position values of the released
    notebook are ``_meta_incl_pos0_*`` (tests/tasks/invariance/
    test_procrustes_linearity.py).

    Outputs (flat floats):
        procrustes_linearity_mean   — mean over adjacent pairs
        procrustes_linearity_min    — least-linear transition
        procrustes_linearity_max    — most-linear transition
        procrustes_linearity_first  — pair (0, 1)
        procrustes_linearity_last   — pair (L-2, L-1)
        procrustes_linearity_slope  — OLS slope on normalized depth
        procrustes_linearity_q25/q50/q75 — quartiles over pairs
    """

    TOKEN_POLICY = "exclude_sinks"
    SOURCES = {
        "paper": "arXiv:2405.12250",
        "code": "AIRI-Institute/LLM-Microscope@b6db939b2696845ce1f88cf69cddc41e808eea17",
        "llm_precedent": "arXiv:2405.12250 (proposed for LLMs); adjusted R^2: "
                         "DOI:10.1214/aoms/1177732951 (Wherry 1931)",
    }
    HEADLINE_KEYS = tuple(
        f"procrustes_linearity_{s}" for s in
        ("mean", "min", "max", "first", "last", "slope", "q25", "q50", "q75")
    ) + tuple(
        f"procrustes_linearity_adjusted_{s}" for s in
        ("mean", "min", "max", "first", "last", "q25", "q50", "q75")) + (
        "procrustes_linearity_heldout_r2_mean",)
    # Raw keys are width- and scale-free but carry the r/(N-1) chance fit; the
    # adjusted twins add N. Identity blocks add pairs scoring exactly 1, so only
    # min/first/last survive them.
    SIZE_INVARIANT = {
        **{f"procrustes_linearity_{s}": (
            ("identity_blocks_interior",) + ends + ("zero_pad", "isometry", "rescale"),
            _SI_NODE.format(f"procrustes_linearity_{s}"))
           for s, ends in (("min", ("identity_blocks_leading", "identity_blocks_trailing")),
                           ("first", ("identity_blocks_trailing",)),
                           ("last", ("identity_blocks_leading",)))},
        **{f"procrustes_linearity_adjusted_{s}": (
            ("N", "identity_blocks_interior") + ends + ("zero_pad", "isometry", "rescale"),
            _SI_NODE.format(f"procrustes_linearity_adjusted_{s}"))
           for s, ends in (("min", ("identity_blocks_leading", "identity_blocks_trailing")),
                           ("first", ("identity_blocks_trailing",)),
                           ("last", ("identity_blocks_leading",)))},
        **{f"procrustes_linearity_adjusted_{s}": (
            ("N", "zero_pad", "isometry", "rescale"),
            _SI_NODE.format(f"procrustes_linearity_adjusted_{s}"))
           for s in ("mean", "max", "q25", "q50", "q75")},
    }

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Procrustes Linearity Analysis...")

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)

        if dataset is None and not (cache is not None and use_cache
                                    and getattr(cache, "is_populated", False)):
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples, include_document_ids=True)
        # Embedding output first, raw deepest block last: all L transitions.
        per_doc, _info = hidden_states_for_task(
            model, tokenizer, dataset, cache, num_samples=num_samples,
            replace_final_norm=True, layer_idx="all", use_cache=use_cache,
            include_embedding=True, fallback="private_cache", per_sample=True)

        if not per_doc:
            return {"error": "No hidden states available for procrustes linearity"}

        layers = sorted(per_doc.keys())
        n_layers = len(layers)
        if n_layers < 2:
            return {"error": "Need at least 2 layers for procrustes linearity"}

        # One keep mask per token from the middle block outputs (the embedding
        # is not a depth, so the mask is the cross-layer tasks'), used at every
        # layer: both clouds of a pair hold the same rows.
        n_blocks = int(get_num_layers(model) or 0)
        _blocks = layers[1:] if (n_blocks > 0 and n_layers == n_blocks + 1) else layers
        _keys = middle_depth_layers({k: per_doc[k] for k in _blocks})
        # Backfilling the held-out readout does not require the three in-sample SVDs.
        heldout = _document_heldout_profile(
            per_doc, layers, _keys, gram_device=self.config.get("heldout_gram_device"))
        if self.config.get("heldout_only", False):
            return heldout
        masks, _sink_info = sink_keep_masks([per_doc[k] for k in _keys])
        layer_states, _rows_meta = {}, None
        for li in layers:
            layer_states[li], _rows_meta = select_rows(
                per_doc[li], self.TOKEN_POLICY, keep=masks)
        sims, sims_ref, sims_adj, ranks, n_tokens = _pair_profile(layer_states, layers)

        sim = np.asarray(sims, dtype=np.float64)
        _sref = np.asarray(sims_ref, dtype=np.float64)
        finite = np.isfinite(sim)
        if not np.any(finite):
            return {"error": "No adjacent layer pair yielded a valid linearity score", **heldout}

        n_pairs = sim.size
        finite_sim = sim[finite]
        _d_model = int(layer_states[layers[0]].shape[-1]) if n_tokens else 0
        _d_over_n = (float(_d_model) / float(n_tokens)) if n_tokens else float("nan")

        out = _profile_summary(sim, "procrustes_linearity")
        out.update(heldout)
        # Size-free twin: the chance fit r/(N-1) removed pair by pair (the
        # slope's twin is not defined; test_procrustes_linearity.py N test).
        _adj = _profile_summary(np.asarray(sims_adj, dtype=np.float64),
                                "procrustes_linearity_adjusted")
        _adj.pop("procrustes_linearity_adjusted_slope")
        out.update(_adj)
        out["_meta_fit_rank_max"] = int(max(ranks)) if ranks else 0
        # The released notebook's rows (every position): the same 17 summaries.
        if self.config.get("meta_incl_pos0", True):
            _all = {li: select_rows(per_doc[li], "all")[0] for li in layers}
            _is, _, _ia, _, _n_all = _pair_profile(_all, layers, reference_rule=False)
            _incl = _profile_summary(np.asarray(_is, dtype=np.float64),
                                     "_meta_incl_pos0_procrustes_linearity")
            _incl_adj = _profile_summary(np.asarray(_ia, dtype=np.float64),
                                         "_meta_incl_pos0_procrustes_linearity_adjusted")
            _incl_adj.pop("_meta_incl_pos0_procrustes_linearity_adjusted_slope")
            out.update(_incl)
            out.update(_incl_adj)
            out["_meta_incl_pos0_n_tokens"] = int(_n_all)
        # Sink diagnostics on the full (inclusive) middle-layer cloud.
        _mid = per_doc[layers[n_layers // 2]]
        out["_meta_position0_share"] = position0_share(_mid)
        out["_meta_massive_token_share"] = massive_token_share(_mid)
        out.update(sink_meta(_sink_info))
        if _rows_meta is not None:
            out.update(_rows_meta.as_meta())
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


# Cached activations are float32. Keep training covariance eigenvalues above
# float32 epsilon relative to the largest, using their square roots in the SVD.
# Coarser half-decade cutoffs are candidates for document validation.
_HELDOUT_RCOND = torch.finfo(torch.float32).eps ** 0.5
_VALIDATION_CUTOFFS = (1e-3, 10**-2.5, 1e-2, 10**-1.5, 1e-1)


def _rank_errors(x_train, y_train, x_test, y_test, cutoffs):
    """Squared prediction errors at several truncations of one training SVD."""
    x_train, y_train, x_test, y_test = (
        t.detach().to("cpu", torch.float64) for t in (x_train, y_train, x_test, y_test)
    )
    x_mean, y_mean = x_train.mean(0), y_train.mean(0)
    target = y_test - y_mean
    baseline_sse = float(target.square().sum())
    u, s, vh = torch.linalg.svd(x_train - x_mean, full_matrices=False)
    left = (x_test - x_mean) @ vh.T
    right = u.T @ (y_train - y_mean)
    errors_by_rank = {0: baseline_sse}
    fits = {}
    for cutoff in cutoffs:
        rank = int((s > s[0] * cutoff).sum())
        if rank not in errors_by_rank:
            prediction = (left[:, :rank] / s[:rank]) @ right[:rank]
            errors_by_rank[rank] = float((target - prediction).square().sum())
        fits[cutoff] = (rank, errors_by_rank[rank])
    return baseline_sse, fits


def _gram_rank_errors(x_train, y_train, x_test, y_test, cutoffs, device):
    """Same truncated affine fit, using the float64 training Gram eigensystem.

    The held-out numerical floor keeps eigenvalues above eps32 times the
    leading eigenvalue. This path is only used for the held-out fit.
    """
    x, y, x_test, y_test = (a.detach().to(device=device, dtype=torch.float64)
                            for a in (x_train, y_train, x_test, y_test))
    x_mean, y_mean = x.mean(0), y.mean(0)
    x = x - x_mean
    y = y - y_mean
    residual = y_test - y_mean
    baseline = float(residual.square().sum())
    eigenvalues, vectors = torch.linalg.eigh(x.T @ x)
    # A singular value on a cutoff boundary can round to either side after
    # squaring. Use the reference decision for such spectra.
    top = eigenvalues[-1]
    thresholds = top * eigenvalues.new_tensor(cutoffs).square()
    tolerance = top * (8 * max(x.shape) * torch.finfo(x.dtype).eps)
    if top > 0 and ((eigenvalues[:, None] - thresholds).abs() <= tolerance).any():
        return _rank_errors(x_train, y_train, x_test, y_test, cutoffs)
    ranks = {c: int((eigenvalues > eigenvalues[-1] * c**2).sum()) for c in cutoffs}
    max_rank = max(ranks.values())
    errors = {0: baseline}
    if max_rank:
        # Accumulate descending eigenspaces once across nested candidate ranks.
        vectors = vectors[:, -max_rank:].flip((1,))
        values = eigenvalues[-max_rank:].flip((0,))
        left = (x_test - x_mean) @ vectors
        right = (vectors.T @ (x.T @ y)) / values[:, None]
        previous = 0
        for rank in sorted(set(ranks.values()) - {0}):
            residual.addmm_(left[:, previous:rank], right[previous:rank], alpha=-1)
            errors[rank] = float(residual.square().sum())
            previous = rank
    return baseline, {c: (rank, errors[rank]) for c, rank in ranks.items()}


def _split_layer_rows(per_doc, layers, mask_layers, groups, fit_docs, score_docs):
    """Yield aligned fit and score tensors with sink thresholds learned on fit_docs."""
    fit_chunks = [group in fit_docs for group in groups]
    score_chunks = [group in score_docs for group in groups]
    masks, _ = sink_keep_masks([per_doc[k] for k in mask_layers], reference_chunks=fit_chunks)
    first = None
    for layer in layers:
        states, rows = select_rows(per_doc[layer], "exclude_sinks", doc_ids=range(len(groups)), keep=masks)
        if first is None:
            first = rows
            fit, score = torch.tensor(fit_chunks)[rows.doc], torch.tensor(score_chunks)[rows.doc]
        elif not (torch.equal(first.doc, rows.doc) and torch.equal(first.position, rows.position)):
            raise ValueError("Procrustes requires matching document and token positions across layers")
        yield states[fit], states[score]


def _document_heldout_profile(per_doc, layers, mask_layers, *, rank_floor=_HELDOUT_RCOND, gram_device=None):
    """Use the same deterministic document split for every block transition."""
    first = per_doc[layers[0]]
    index = range(len(first))
    def identifiers(chunks):
        ids = getattr(chunks, "doc_ids", None)
        ids = tuple(int(i) for i in ids) if ids is not None else tuple(range(len(chunks)))
        parents = getattr(chunks, "document_groups", None)
        groups = tuple(parents) if parents is not None else tuple(f"record:{i}" for i in ids)
        if len(ids) != len(chunks) or len(groups) != len(chunks):
            raise ValueError("Each Procrustes chunk needs one document identifier")
        return ids, groups

    row_ids, groups = identifiers(first)
    if any(identifiers(per_doc[layer]) != (row_ids, groups) for layer in layers[1:]):
        raise ValueError("Procrustes requires identical document order across layers")
    _, last = select_rows(first, "last", doc_ids=index)
    eligible = sorted({groups[i] for i in last.doc[last.position > 0].tolist()})
    test = set(eligible[::5])
    train = set(eligible) - test
    inner_val = set(sorted(train)[::5]) if len(train) >= 2 else set()
    inner_train = train - inner_val
    candidates = (rank_floor, *(c for c in _VALIDATION_CUTOFFS if c > rank_floor))
    if gram_device is None:
        solve = _rank_errors
    else:
        from functools import partial
        if rank_floor < _HELDOUT_RCOND:
            raise ValueError("The Gram solver requires at least the float32 covariance floor")
        solve = partial(_gram_rank_errors, device=gram_device)
    key = "procrustes_linearity_heldout_r2_mean"
    meta = {
        "split": "every fifth sorted eligible document ID, starting with the first",
        "train_documents": sorted(train), "test_documents": sorted(test),
        "train_tokens": 0, "test_tokens": 0, "pair_r2": [],
        "rank_relative_floor": rank_floor, "candidate_cutoffs": list(candidates),
        "rank_fit_documents": sorted(inner_train), "rank_validation_documents": sorted(inner_val),
        "rank_fit_tokens": 0, "rank_validation_tokens": 0, "rank_selection": "numerical floor only",
        "pair_relative_cutoff": [], "pair_fit_rank": [],
        "baseline": "training target mean", "status": "fewer than two eligible documents",
    }
    out = {key: float("nan"), "_meta_procrustes_heldout": meta}
    if not train or not test:
        return out
    outer = _split_layer_rows(per_doc, layers, mask_layers, groups, train, test)
    x_fit, x_test = next(outer)
    meta.update(train_tokens=len(x_fit), test_tokens=len(x_test))
    if meta["train_tokens"] < 2 or meta["test_tokens"] == 0:
        meta["status"] = "insufficient rows after token selection"
        return out
    inner = None
    if inner_val:
        inner = _split_layer_rows(per_doc, layers, mask_layers, groups, inner_train, inner_val)
        x_inner, x_val = next(inner)
        meta.update(rank_fit_tokens=len(x_inner), rank_validation_tokens=len(x_val))
        if len(x_inner) >= 2 and len(x_val):
            meta["rank_selection"] = "minimum inner-document squared prediction error"
        else:
            inner = None
    valid = []
    for y_fit, y_test in outer:
        cutoff = rank_floor
        if inner is not None:
            y_inner, y_val = next(inner)
            baseline, fits = solve(x_inner, y_inner, x_val, y_val, candidates)
            best = min(error for _, error in fits.values())
            # Increasing cutoffs alter the estimator only when validation
            # distinguishes them beyond double-precision roundoff.
            tolerance = 64 * torch.finfo(torch.float64).eps * max(best, baseline)
            cutoff = next(c for c in candidates if fits[c][1] <= best + tolerance)
            x_inner, x_val = y_inner, y_val
        baseline, fits = solve(x_fit, y_fit, x_test, y_test, (cutoff,))
        rank, error = fits[cutoff]
        r2 = 1 - error / baseline if baseline else float("nan")
        meta["pair_relative_cutoff"].append(cutoff)
        meta["pair_fit_rank"].append(rank)
        meta["pair_r2"].append(r2 if np.isfinite(r2) else None)
        if np.isfinite(r2):
            valid.append(r2)
        x_fit, x_test = y_fit, y_test
    if valid:
        out[key] = float(np.mean(valid))
        meta["status"] = "ok"
    else:
        meta["status"] = "zero held-out baseline sum of squares"
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
