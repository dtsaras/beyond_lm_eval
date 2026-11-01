"""``geometry_spectral_signature``: the PL_Alpha_Hill "spectral signature" of
Zhang et al. 2026 (arXiv:2607.03377), the heavy-tail estimator of the HT-SR
LLM literature (AlphaPruning, Lu et al. 2024, arXiv:2410.10912).

For every weight matrix, the eigenvalues of ``W^T W`` (squared float32
singular values) above 1e-5 are sorted ascending, ``xmin`` is the one at
index ``N // 2`` (``fix_fingers='xmin_mid'``), and the Hill estimate over the
upper half is ``alpha = 1 + n / (sum_{j>=i} log lambda_j - n log lambda_i)``.
The signature is the vector of these alphas over the model's matrices; the
release compares two models by the Spearman correlation of their aligned
alphas, which needs both models, so BLME reports the per-model profile
(``_meta_``) and its mean and median.

The matrix population follows the released ``get_module_names_shapes`` and
``net_esd_estimator`` (Ingrid-505/Spectral_Signature@d04eacd): Linear, HF
Conv1D and Conv2d weights, Linear weights with an aspect ratio of 8 or more
skipped (the LM head), fused QKV weights of attention modules split into
three, and packed 3-D expert weights split per expert. Two departures, each
counted in ``_meta_``: torch convolution kernels are skipped (the release's
packed-expert branch would read Qwen3.5's depthwise ``conv1d`` (C, 1, 4) as C
one-eigenvalue "experts", each with alpha = inf), and only the language model
is scanned (``get_decoder()`` plus the output embedding), so a multimodal
checkpoint's vision tower is left out, as in ``geometry_spectral``.
"""
import logging

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from .spectral import _language_model_modules

logger = logging.getLogger("blme")

# net_esd_estimator defaults: EVALS_THRESH=0.00001, fix_fingers="xmin_mid", xmin_pos=2.
EVALS_THRESH = 1e-5
XMIN_POS = 2
# net_esd_estimator: "ignore classifier-like layers (very elongated linear layers)".
MAX_ASPECT = 8


def _hill_alpha_xmin_mid(matrix: torch.Tensor, evals_thresh: float = EVALS_THRESH,
                         xmin_pos: int = XMIN_POS):
    """``(alpha, n_eigenvalues)`` of ``matrix_esd_estimator(fix_fingers='xmin_mid')``."""
    eigs = torch.square(torch.linalg.svdvals(matrix.to(dtype=torch.float32)).flatten())
    eigs, _ = torch.sort(eigs, descending=False)
    nz_eigs = eigs[eigs > evals_thresh]
    if len(nz_eigs) == 0:
        nz_eigs = eigs
    N = len(nz_eigs)
    log_nz_eigs = torch.log(nz_eigs)
    i = int(N / xmin_pos)
    n = float(N - i)
    alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
    return float(alpha.item()), int(N)


def _split_qkv(name: str, weight: torch.Tensor):
    """``attn_split_qkv`` on a 2-D attention weight: thirds along the long axis
    when one side is exactly three times the other."""
    m, n = weight.shape
    if m == n // 3:
        return [(name + "_q", weight[:, :m]), (name + "_k", weight[:, m:2 * m]),
                (name + "_v", weight[:, 2 * m:])]
    if n == m // 3:
        return [(name + "_q", weight[:n, :]), (name + "_k", weight[n:2 * n, :]),
                (name + "_v", weight[2 * n:, :])]
    return [(name, weight)]


def _signature_matrices(model):
    """``(names, matrices, counts)`` of the matrices the signature fits."""
    try:
        out_emb = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    except Exception:
        out_emb = None
    lm_ids = _language_model_modules(model, out_emb)
    names, mats = [], []
    counts = {"conv_kernels_skipped": 0, "modules_outside_lm": 0, "elongated_skipped": 0}
    for name, module in model.named_modules():
        own = [p for _, p in module.named_parameters(recurse=False) if p.dim() >= 2]
        if lm_ids is not None and id(module) not in lm_ids:
            counts["modules_outside_lm"] += bool(own)
            continue
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            counts["conv_kernels_skipped"] += 1
            continue
        packed = {pname: p for pname, p in module.named_parameters(recurse=False) if p.dim() == 3}
        if packed:
            for pname, p in packed.items():
                for e in range(p.shape[0]):
                    pieces = [(f"{name}.expert_{e}.{pname}", p[e].detach())]
                    _keep(pieces, "linear", names, mats, counts)
            continue
        w = getattr(module, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dim() < 2:
            continue
        if any(isinstance(getattr(c, "weight", None), torch.Tensor)
               for _, c in module.named_children()):
            continue
        pieces = (_split_qkv(name, w.detach())
                  if ("attn" in name or "attention" in name) and w.dim() == 2
                  else [(name, w.detach())])
        _keep(pieces, type(module).__name__.lower(), names, mats, counts)
    return names, mats, counts


def _keep(pieces, type_name, names, mats, counts):
    """net_esd_estimator's per-module filter, applied after the QKV split."""
    if type_name not in ("linear", "conv1d", "conv2d"):
        return
    for nm, w in pieces:
        if w.dim() > 2:          # Conv2d kernels: none in the language models scanned
            continue
        if type_name == "linear" and max(w.shape) / min(w.shape) >= MAX_ASPECT:
            counts["elongated_skipped"] += 1
            continue
        names.append(nm)
        mats.append(w)


@register_task("geometry_spectral_signature")
class SpectralSignatureTask(DiagnosticTask):
    """PL_Alpha_Hill (xmin = median eigenvalue) profile over the weight matrices."""

    TOKEN_POLICY = "not_token_based"
    SOURCES = {
        "paper": "arXiv:2607.03377 (Zhang et al. 2026, Spectral Signatures of Large Language Models)",
        "code": "Ingrid-505/Spectral_Signature@d04eacd9689b8922074e7bc2188bdac1d52e6a85",
        "llm_precedent": ("arXiv:2410.10912 (Lu et al. 2024, AlphaPruning: the same xmin_mid "
                          "PL_Alpha_Hill per weight matrix of LLaMA-family LLMs)"),
    }
    ADAPTATION_NOTE = (
        "Each weight matrix's heavy-tail exponent is the Hill estimate over the upper half of "
        "the eigenvalues of W^T W above 1e-5 (xmin = the median eigenvalue), the PL_Alpha_Hill "
        "of AlphaPruning and Spectral Signatures, over the matrices the released Spectral "
        "Signature code selects. The release compares two models through their aligned "
        "profiles; one model's profile is kept as _meta_ and its mean and median are the "
        "headline. Convolution kernels are skipped, because the release's expert branch would "
        "read Qwen3.5's depthwise conv1d as one-eigenvalue matrices with alpha = inf, and only "
        "the language model is scanned. A matrix's alpha is unchanged by zero-padding and by "
        "an isometric embedding of it; the mean moves when blocks are added (an identity block "
        "still carries copied attention and MLP matrices), and the absolute 1e-5 eigenvalue "
        "cut is not scale-free."
    )
    HEADLINE_KEYS = ("alpha_hill_mean", "alpha_hill_median")
    SIZE_INVARIANT = {}

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Spectral Signature (PL_Alpha_Hill) Analysis...")
        names, mats, counts = _signature_matrices(model)
        if not mats:
            return {"error": "No weight matrices found for the spectral signature"}
        alphas, n_eigs = [], []
        with torch.no_grad():
            for w in mats:
                a, n = _hill_alpha_xmin_mid(w)
                alphas.append(a)
                n_eigs.append(n)
        arr = np.asarray(alphas, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        return {
            "alpha_hill_mean": float(finite.mean()) if finite.size else float("nan"),
            "alpha_hill_median": float(np.median(finite)) if finite.size else float("nan"),
            "_meta_alpha_hill_profile": [float(a) for a in alphas],
            "_meta_alpha_hill_modules": names,
            "_meta_n_matrices": len(mats),
            "_meta_n_alpha_nonfinite": int(arr.size - finite.size),
            "_meta_median_n_eigenvalues": float(np.median(n_eigs)),
            "_meta_n_conv_kernels_skipped": counts["conv_kernels_skipped"],
            "_meta_n_modules_outside_lm": counts["modules_outside_lm"],
            "_meta_n_elongated_skipped": counts["elongated_skipped"],
        }
