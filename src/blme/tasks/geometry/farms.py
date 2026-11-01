"""``geometry_farms_alpha``: FARMS, the heavy-tail exponent of every weight
matrix measured on fixed-aspect-ratio windows (Hu, Goel, Killiakov & Yang,
"Eigenspectrum Analysis of Neural Networks without Aspect Ratio Bias",
ICML 2025, arXiv:2506.06280).

A whole-matrix power-law alpha moves with the matrix's aspect ratio: the
Marchenko-Pastur bulk of an m x n matrix narrows as m/n leaves 1, which alone
raises alpha. FARMS fits the power law to the pooled eigenvalues of square
(Q = 1) windows cut from the matrix on a regular grid instead.

The estimator follows the released code (HUST-AI-HYZ/FARMS@e0a2ded, LLM_Pruning)
in its LLM configuration (``scripts/llm_pruning_main.txt``: ``alpha_peak_farms``,
``use_sliding_window=True``, ``num_row_samples=2000``, ``Q_ratio=1``,
``sampling_ops_per_dim=15``; ``net_esd_estimator_farms`` defaults ``bins=100``,
``filter_zeros=False``, ``eigs_num_thresh=5``):

* windows (``sampling.fixed_sampling``): ``r = min(2000, min(rows, cols))``
  rows by ``int(r * Q)`` columns, at most 15 x 15 of them, row starts
  ``min(i * row_step, rows - r)`` with ``row_step = max(1, (rows - r) // 14)``
  (columns alike); the squared float32 singular values of every window are
  pooled and sorted. ``r`` is capped by the matrix's narrow side, so a narrow
  model gets smaller windows than a wide one;
* fit (``esd_utils_farms.analysis_esd``, ``fix_fingers='xmin_peak'``): a
  100-bin histogram of ``log10`` of the pooled eigenvalues locates the peak
  bin's left edge ``p``; every eigenvalue ``lambda_i`` (except the largest)
  with ``log10(0.95 p) <= lambda_i <= 1.5 p`` is a candidate xmin, with Hill
  exponent ``alpha_i = 1 + n_i / (sum_{j>=i} log lambda_j - n_i log lambda_i)``
  and KS distance ``D_i = max_{j>=i} |1 - (lambda_j/lambda_i)^(1-alpha_i) -
  (j-i)/n_i|``; the alpha of the smallest ``D_i`` is the matrix's alpha
  (``argmin`` over an array that holds alpha 0 / D 1 for non-candidates).

The lower bound of the candidate range is the release's: ``log10(0.95 p)`` is
in log10 units and is compared with linear eigenvalues. It is reproduced as
written (it is the released method): it is not a scale-free bound, it is no
bound at all when ``p < 1/0.95`` (it is then negative) and it excludes small
eigenvalues when ``p`` is larger (Qwen3-8B-Base layer 10 q_proj, 2000-windows:
p = 1.26, the 95,065 smallest of 450,000 eigenvalues are not candidates).

Two properties of the released fit, measured and kept: the KS optimum usually
sits at the top of the candidate range (gpt2: within 1% of the range below the
1.5 p cap for 36 of 48 matrices with 768-windows and 39 of 48 with 512-windows,
median 0.2% and 0.02%), so alpha is mostly the Hill exponent above about
1.5 p; and p is the left edge of a histogram bin whose range starts at the
smallest pooled eigenvalue, a near-zero singular value that float32 SVD
resolves only to ~1e-7 of the largest. When the two highest bins nearly tie,
rounding of that one eigenvalue moves p by a whole bin: on gpt2's
``h.6.mlp.c_proj`` with 512-windows, multiplying the smallest of 115,200
eigenvalues (1.9e-10) by 1.0065, the difference between the SVD of W and of
W.T, moves alpha from 3.275 to 2.857 in the release and here alike. Over
gpt2's 48 matrices, measuring W instead of W.T moves the mean by 1.3e-4
(relative; 2000-windows, at most 0.8% per matrix) and by 6.2e-3 (512-windows,
at most 14% per matrix): a rounding-noise floor of each headline.
``_meta_n_peak_bin_near_tie`` (runner-up bin within 1% of the peak count; 19
and 21 of gpt2's 48) and ``_meta_n_xmin_at_candidate_cap`` (xmin within 1% of
the range below the cap) count the matrices exposed to this.

The release evaluates each candidate in a Python loop (O(N k) for N pooled
eigenvalues and k candidates, up to 225 x 2000 eigenvalues per matrix at
d >= 2000). Here every candidate's alpha comes from float64 suffix sums of
``log lambda`` (rounded to float32, the release's dtype) and ``D_i`` is the
release's float32 expression, evaluated exactly for every candidate that can
win: lower bounds of ``D_i`` on coarse grids of ``j`` discard the rest
(``_min_ks``). tests/tasks/parity/test_farms_parity.py compares alpha, D and
the selected xmin index with the released ``analysis_esd`` on the same pooled
eigenvalues.

Population: every ``torch.nn.Linear`` inside the transformer blocks
(``get_layers(model)``; the release runs on ``model.model.layers``), plus HF
``Conv1D`` weights (GPT-2 family), which the release cannot see. A Conv1D
weight is stored (in, out); it is measured as its Linear equivalent ``W.T``.
"""
import logging
import math

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_layers

try:
    from transformers.pytorch_utils import Conv1D as _HFConv1D
except Exception:  # pragma: no cover - transformers is a hard dependency in practice
    _HFConv1D = None

logger = logging.getLogger("blme")

# The release's LLM configuration (scripts/llm_pruning_main.txt, alpha_peak_farms rows).
NUM_ROW_SAMPLES = 2000
Q_RATIO = 1.0
SAMPLING_OPS_PER_DIM = 15
# net_esd_estimator_farms defaults.
BINS = 100
EIGS_NUM_THRESH = 5
# The size-independent twin (BLME): one window size for every model.
TWIN_NUM_ROW_SAMPLES = 512
TWIN_Q_RATIO = 1.0
TWIN_SAMPLING_OPS_PER_DIM = 15

# _min_ks: grids of j on which lower bounds of D_i prune candidates, the number
# of best-bound candidates evaluated exactly per level, the pruning slack (D is
# in [0, 1); 1e-6 is ~100 float32 ulps at D ~ 0.05, so a bound and an exact value
# computed in different vector lanes cannot prune the true minimum), and the
# elements per evaluation chunk.
_GRIDS = (256, 2048, 16384, 131072)
_TOP = 16
_SLACK = 1e-6
_CHUNK = 1 << 22
# Windows per batched SVD: at most this many float32 elements.
_SVD_BATCH_ELEMS = 1 << 25


def _window_grid(rows: int, cols: int, num_row_samples: int, q_ratio: float,
                 sampling_ops_per_dim: int):
    """``(r, c, row_starts, col_starts)`` of ``sampling.fixed_sampling``."""
    r = min(int(num_row_samples), min(rows, cols))
    c = int(r * q_ratio)
    ops = int(sampling_ops_per_dim)
    row_step = max(1, (rows - r) // (ops - 1))
    col_step = max(1, (cols - c) // (ops - 1))
    n_row = min(ops, max(1, (rows - r) // row_step + 1))
    n_col = min(ops, max(1, (cols - c) // col_step + 1))
    row_starts = [min(i * row_step, rows - r) for i in range(n_row)]
    col_starts = [min(j * col_step, cols - c) for j in range(n_col)]
    return r, c, row_starts, col_starts


def _pooled_window_eigs(matrix: torch.Tensor, num_row_samples: int = NUM_ROW_SAMPLES,
                        q_ratio: float = Q_RATIO,
                        sampling_ops_per_dim: int = SAMPLING_OPS_PER_DIM):
    """Sorted pooled eigenvalues of the window grid and the number of windows.

    Squared float32 singular values of every window ``W[rs:rs+r, cs:cs+c]``
    (Python slice semantics, as the release slices), batched by window shape;
    the SVD runs on the matrix's device, the result is on the CPU.
    """
    W = matrix.detach().to(dtype=torch.float32)
    rows, cols = W.shape
    r, c, row_starts, col_starts = _window_grid(rows, cols, num_row_samples, q_ratio,
                                                sampling_ops_per_dim)
    windows = [W[rs:rs + r, cs:cs + c] for rs in row_starts for cs in col_starts]
    by_shape = {}
    for w in windows:
        by_shape.setdefault(tuple(w.shape), []).append(w)
    eigs = []
    for shape, group in by_shape.items():
        per = max(1, _SVD_BATCH_ELEMS // max(1, shape[0] * shape[1]))
        for s in range(0, len(group), per):
            sv = torch.linalg.svdvals(torch.stack(group[s:s + per]))
            eigs.append(torch.square(sv).flatten().cpu())
    pooled = torch.cat(eigs) if eigs else torch.zeros(0)
    return torch.sort(pooled).values, len(windows)


def _ks(eigs: torch.Tensor, idx: torch.Tensor, expo: torch.Tensor, cols: torch.Tensor):
    """``max_{j in cols, j >= i} |1 - (eigs[j]/eigs[i])**expo_i - (j-i)/(N-i)|`` per
    candidate ``i``, in float32: ``analysis_esd``'s D with ``expo = -alpha + 1``."""
    N = eigs.numel()
    out = torch.empty(idx.numel(), dtype=torch.float32)
    if idx.numel() == 0:
        return out
    per = max(1, _CHUNK // max(1, cols.numel()))
    for s in range(0, idx.numel(), per):
        ii = idx[s:s + per]
        cc = cols[int(torch.searchsorted(cols, ii.min())):]
        ratio = eigs[cc][None, :] / eigs[ii][:, None]
        p = ratio ** expo[s:s + per][:, None]
        seq = (cc[None, :] - ii[:, None]).to(torch.float32)
        n = (N - ii).to(torch.float32)[:, None]
        v = torch.abs(1 - p - seq / n)
        # j < i is not in the tail; 0 never raises a max (the j = i term is 0).
        v = torch.where(cc[None, :] >= ii[:, None], v, torch.zeros((), dtype=v.dtype))
        out[s:s + per] = v.amax(dim=1)
    return out


def _min_ks(eigs: torch.Tensor, idx: torch.Tensor, alpha: torch.Tensor):
    """D values for the candidates ``idx`` such that ``argmin`` over them is the
    argmin of their exact D: exact for every candidate that can attain the
    minimum, a lower bound above it for the rest.

    A D computed on a subset of j is a lower bound of the exact D (same float32
    expression). Each level bounds the live candidates on a coarser grid of j,
    evaluates the best-bounded few exactly, and drops every candidate whose
    bound exceeds the best exact D by more than ``_SLACK``; the survivors are
    evaluated exactly at the end.
    """
    N = eigs.numel()
    k = idx.numel()
    expo = -alpha + 1
    full = torch.arange(N)
    D = torch.zeros(k, dtype=torch.float32)
    exact = torch.zeros(k, dtype=torch.bool)
    alive = torch.arange(k)
    best = math.inf
    lo = int(idx.min()) if k else 0
    for g in _GRIDS:
        if alive.numel() <= 4 * _TOP or g >= (N - lo) // 2:
            break
        grid = torch.unique(torch.linspace(lo, N - 1, g).round().long())
        lb = _ks(eigs, idx[alive], expo[alive], grid)
        D[alive] = torch.maximum(D[alive], lb)
        top = alive[torch.argsort(D[alive])[:_TOP]]
        top = top[~exact[top]]
        if top.numel():
            D[top] = _ks(eigs, idx[top], expo[top], full)
            exact[top] = True
        if bool(exact.any()):
            best = min(best, float(D[exact].min()))
        alive = alive[(~exact[alive]) & (D[alive] <= best + _SLACK)]
    if alive.numel():
        alive = alive[torch.argsort(idx[alive])]
        D[alive] = _ks(eigs, idx[alive], expo[alive], full)
    return D


def _xmin_peak_fit(eigs: torch.Tensor, bins: int = BINS):
    """``analysis_esd(fix_fingers='xmin_peak', filter_zeros=False)`` on pooled
    eigenvalues: ``{alpha, D, xmin_index, xmin, n_eigs, n_candidates}``.

    Non-positive or non-finite eigenvalues make the release's ``torch.histc``
    raise (its log10 range is not finite); here they give alpha = NaN with a
    reason.
    """
    eigs = torch.sort(eigs.detach().to(device="cpu", dtype=torch.float32).flatten()).values
    N = int(eigs.numel())
    out = {"alpha": float("nan"), "D": float("nan"), "xmin_index": -1, "xmin": float("nan"),
           "n_eigs": N, "n_candidates": 0}
    if N < 2:
        out["reason"] = f"{N} eigenvalues"
        return out
    log_e = torch.log(eigs)
    h = torch.log10(eigs)
    min_e, max_e = h.min(), h.max()
    if not (bool(torch.isfinite(min_e)) and bool(torch.isfinite(max_e))):
        out["reason"] = "non-positive or non-finite eigenvalue (the release's histc raises)"
        return out
    counts = torch.histc(h, bins, min=min_e, max=max_e)
    boundaries = torch.linspace(min_e, max_e, bins + 1)
    xmin2 = 10 ** boundaries[torch.argmax(counts)]
    # sic: the released lower bound is log10(0.95 * peak), in log10 units, and is
    # compared with the linear eigenvalues below. Reproduced as written.
    xmin_min = torch.log10(0.95 * xmin2)
    xmin_max = 1.5 * xmin2
    head = eigs[:-1]
    # The release loops i over eigs[:-1]: `continue` while lambda_i < xmin_min,
    # `break` at the first lambda_i > xmin_max; on sorted eigenvalues the
    # candidates are one index range.
    lo = int(torch.searchsorted(head, xmin_min.reshape(1), right=False))
    hi = max(lo, int(torch.searchsorted(head, xmin_max.reshape(1), right=True)))
    alphas = torch.zeros(N - 1, dtype=torch.float32)
    Ds = torch.ones(N - 1, dtype=torch.float32)
    if hi > lo:
        idx = torch.arange(lo, hi)
        log64 = log_e.to(torch.float64)
        suffix = torch.flip(torch.cumsum(torch.flip(log64, (0,)), 0), (0,))
        n = (N - idx).to(torch.float64)
        a = (1 + n / (suffix[idx] - n * log64[idx])).to(torch.float32)
        alphas[idx] = a
        fit = a > 1      # the release computes D only when alpha > 1 (D stays 1)
        if bool(fit.any()):
            Ds[idx[fit]] = _min_ks(eigs, idx[fit], a[fit])
    j = int(torch.argmin(Ds))
    # Diagnostics of the release's fragility (not part of its output): the KS
    # optimum usually sits at the 1.5 * peak cap, so alpha follows the peak bin,
    # and a runner-up bin within 1% of the peak count can take over when float32
    # rounding moves the smallest eigenvalue, which sets the histogram's range.
    top2 = torch.topk(counts, min(2, bins)).values
    near_tie = bool(top2.numel() > 1 and float(top2[1]) >= 0.99 * float(top2[0]))
    at_cap = bool(hi > lo and j >= hi - max(1, math.ceil(0.01 * (hi - lo))))
    out.update(alpha=float(alphas[j]), D=float(Ds[j]), xmin_index=j, xmin=float(eigs[j]),
               n_candidates=hi - lo, candidate_range=(lo, hi), peak_near_tie=near_tie,
               xmin_at_cap=at_cap)
    return out


def _farms_matrices(model):
    """``(names, matrices, counts)``: every nn.Linear in the transformer blocks
    (the release's population), plus HF Conv1D weights as their Linear
    equivalent ``W.T``.

    Counted, not measured: Conv2d weights (the release has a separate conv path;
    no LLM block in the study has one) and every other weight tensor of 2 or
    more dimensions in the blocks (packed MoE experts, torch Conv1d kernels),
    which the release's ``isinstance(m, nn.Linear)`` does not see either.
    """
    counts = {"conv1d_included": 0, "conv2d_skipped": 0, "other_weight_tensors_unmeasured": 0}
    blocks = get_layers(model)
    if blocks is None:
        return [], [], counts
    prefix = next((n for n, m in model.named_modules() if m is blocks), "")
    names, mats = [], []
    for rel, m in blocks.named_modules():
        name = f"{prefix}.{rel}" if prefix and rel else (prefix or rel)
        if isinstance(m, torch.nn.Linear):
            names.append(name)
            mats.append(m.weight.detach())
        elif _HFConv1D is not None and isinstance(m, _HFConv1D):
            names.append(name)
            mats.append(m.weight.detach().t())
            counts["conv1d_included"] += 1
        elif isinstance(m, torch.nn.Conv2d):
            counts["conv2d_skipped"] += 1
        else:
            counts["other_weight_tensors_unmeasured"] += sum(
                1 for _, p in m.named_parameters(recurse=False) if p.dim() >= 2)
    return names, mats, counts


def _farms_alpha(matrix: torch.Tensor, num_row_samples: int, q_ratio: float,
                 sampling_ops_per_dim: int):
    """FARMS fit of one matrix, or None when it pools fewer than 5 eigenvalues
    (``eigs_num_thresh``: the release skips the matrix)."""
    eigs, n_windows = _pooled_window_eigs(matrix, num_row_samples, q_ratio,
                                          sampling_ops_per_dim)
    if eigs.numel() < EIGS_NUM_THRESH:
        return None
    fit = _xmin_peak_fit(eigs)
    fit["n_windows"] = n_windows
    fit["window_rows"] = min(int(num_row_samples), min(matrix.shape))
    return fit


def _summary(alphas):
    arr = np.asarray(alphas, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return (float(finite.mean()) if finite.size else float("nan")), int(arr.size - finite.size)


@register_task("geometry_farms_alpha")
class FARMSAlphaTask(DiagnosticTask):
    """FARMS alpha: the xmin_peak power-law exponent of each weight matrix's
    pooled fixed-aspect-ratio window spectrum, averaged over the matrices."""

    TOKEN_POLICY = "not_token_based"
    SOURCES = {
        "paper": ("arXiv:2506.06280 (Hu, Goel, Killiakov & Yang, ICML 2025, Eigenspectrum "
                  "Analysis of Neural Networks without Aspect Ratio Bias)"),
        "code": "HUST-AI-HYZ/FARMS@e0a2ded79a4436259e1adb32fe2fa0adc16b1620",
        "llm_precedent": ("arXiv:2506.06280 (the release's LLM_Pruning: alpha_peak_farms over every "
                          "nn.Linear of LLaMA-7B/13B's blocks, allocating AlphaPruning's "
                          "layer-wise sparsity, arXiv:2410.10912)"),
    }
    ADAPTATION_NOTE = (
        "Each weight matrix's alpha is the released FARMS estimator in its LLM configuration: "
        "squared singular values of up to 15 x 15 square windows of side min(2000, rows, cols), "
        "pooled, with the xmin_peak KS-optimal Hill fit, whose candidate lower bound "
        "log10(0.95 * peak) is compared with linear eigenvalues as the release does. "
        "The population is the release's, every nn.Linear in the transformer blocks, plus HF "
        "Conv1D weights (GPT-2), which the release cannot see, measured as their Linear "
        "equivalent W.T. "
        "farms_alpha_mean_w512 is a BLME twin (class 2): the same estimator with 512 x 512 "
        "windows for every model, because the release's window side min(2000, rows, cols) "
        "grows with width and changes the spectrum being fitted. "
        "It does not fix matrices narrower than 512, which still get their own smaller "
        "window, the window covering a smaller share of a wider matrix, or the mean's "
        "dependence on how many matrices a model has. "
        "Neither headline is invariant to zero-padding or isometric embedding, which change "
        "the windows' contents. "
        "The released fit is fragile and kept as released: its KS optimum usually sits at the cap of "
        "1.5 times the peak histogram bin, and when two bins nearly tie, float32 rounding of the "
        "smallest eigenvalue can move one matrix's alpha by over 10% (gpt2 h.6.mlp.c_proj, "
        "512-windows: 3.27 or 2.86), which _meta_ counts as near-ties."
    )
    HEADLINE_KEYS = ("farms_alpha_mean", "farms_alpha_mean_w512")
    SIZE_INVARIANT = {}

    def _settings(self):
        nrs = int(self.config.get("num_row_samples", NUM_ROW_SAMPLES))
        q = float(self.config.get("q_ratio", Q_RATIO))
        ops = int(self.config.get("sampling_ops_per_dim", SAMPLING_OPS_PER_DIM))
        if nrs < 1 or q <= 0 or ops < 2:
            raise ValueError(f"geometry_farms_alpha: need num_row_samples >= 1, q_ratio > 0 and "
                             f"sampling_ops_per_dim >= 2, got {nrs}, {q}, {ops}")
        return {"release": (nrs, q, ops),
                "w512": (TWIN_NUM_ROW_SAMPLES, TWIN_Q_RATIO, TWIN_SAMPLING_OPS_PER_DIM)}

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running FARMS (fixed-aspect-ratio window) alpha Analysis...")
        settings = self._settings()
        names, mats, counts = _farms_matrices(model)
        if not mats:
            return {"error": "No nn.Linear / Conv1D weights found in the transformer blocks"}
        out = {}
        with torch.no_grad():
            for tag, (nrs, q, ops) in settings.items():
                kept, alphas, Ds, xmins, n_eigs, n_cand, win_rows, n_win = [], [], [], [], [], [], [], []
                reasons, skipped, near_ties, at_cap = {}, 0, 0, 0
                for name, W in zip(names, mats):
                    fit = _farms_alpha(W, nrs, q, ops)
                    if fit is None:
                        skipped += 1
                        continue
                    near_ties += int(fit.get("peak_near_tie", False))
                    at_cap += int(fit.get("xmin_at_cap", False))
                    kept.append(name)
                    alphas.append(fit["alpha"])
                    Ds.append(fit["D"])
                    xmins.append(fit["xmin"])
                    n_eigs.append(fit["n_eigs"])
                    n_cand.append(fit["n_candidates"])
                    win_rows.append(fit["window_rows"])
                    n_win.append(fit["n_windows"])
                    if "reason" in fit:
                        reasons[name] = fit["reason"]
                mean, n_nonfinite = _summary(alphas)
                sfx = "" if tag == "release" else "_" + tag
                out["farms_alpha_mean" + sfx] = mean
                out[f"_meta_farms_alphas{sfx}"] = alphas
                out[f"_meta_farms_modules{sfx}"] = kept
                out[f"_meta_farms_ks_D{sfx}"] = Ds
                out[f"_meta_farms_xmin{sfx}"] = xmins
                out[f"_meta_farms_n_eigenvalues{sfx}"] = n_eigs
                out[f"_meta_farms_n_candidates{sfx}"] = n_cand
                out[f"_meta_farms_window_rows{sfx}"] = win_rows
                out[f"_meta_farms_n_windows{sfx}"] = n_win
                out[f"_meta_n_matrices_fitted{sfx}"] = len(kept)
                out[f"_meta_n_skipped_fewer_than_5_eigenvalues{sfx}"] = skipped
                out[f"_meta_n_alpha_nonfinite{sfx}"] = n_nonfinite
                out[f"_meta_n_peak_bin_near_tie{sfx}"] = near_ties
                out[f"_meta_n_xmin_at_candidate_cap{sfx}"] = at_cap
                # [num_row_samples, q_ratio, sampling_ops_per_dim]
                out[f"_meta_window_config{sfx}"] = [nrs, q, ops]
                if reasons:
                    out[f"_meta_alpha_nan_reasons{sfx}"] = [f"{k}: {v}" for k, v in reasons.items()]
        out["_meta_n_matrices"] = len(mats)
        out["_meta_n_conv1d_included"] = counts["conv1d_included"]
        out["_meta_n_conv2d_skipped"] = counts["conv2d_skipped"]
        out["_meta_n_other_weight_tensors_unmeasured"] = counts["other_weight_tensors_unmeasured"]
        return out
