"""Weight-Activation Alignment (WAA)
──────────────────────────────────────────────────────────────────────
For each transformer block, compare the top singular direction of the
MLP output projection's weight matrix (in its *input* feature space)
with the top principal component of the *actual* MLP intermediate
activations observed during inference. High alignment means the model
is using the feature directions it stored in its weights.

References:
  * Park, Choe, Veitch 2024 — "The Linear Representation
    Hypothesis and the Geometry of Large Language Models", arXiv:
    2311.03658 — formalises the input-space / output-space duality
    this task measures.
  * Elhage et al. 2022 — "Toy Models of Superposition" — motivates
    why alignment between weight singular directions and activation
    principal components is a useful interpretability signal.

Rewrite (2026-04-17 audit):
  * Hooks every target projection simultaneously and captures the
    activations in a single forward pass over the corpus (the previous
    implementation ran ``num_layers × num_samples`` forward passes, so
    on Llama 3 8B with 32 layers × 5 samples it tripped the 600 s task
    timeout — 22/32 models failed).
  * Extended the projection name table to cover Pythia / GPT-NeoX
    (``dense_4h_to_h``), OLMo (``ff_out``), and Phi-2 (``fc2``).
  * Uses ``torch.linalg.svd`` instead of the deprecated ``torch.svd``.
  * Subsamples tokens before SVD (default 4096) so the eigh step stays
    tractable on ~14 k-dim intermediates.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_layers

logger = logging.getLogger("blme")


try:
    from transformers.pytorch_utils import Conv1D as _HFConv1D
except Exception:  # pragma: no cover - optional dep
    _HFConv1D = None


# Projection attribute names in the order we prefer them. The "output"
# projection of an MLP is always the one that maps the expanded
# intermediate space back down to the residual-stream width.
_OUT_PROJ_NAMES = (
    "down_proj",       # Llama, Qwen, Gemma, Mistral (gated MLP)
    "dense_4h_to_h",   # Pythia / GPT-NeoX / GPT-J
    "ff_out",          # OLMo
    "fc2",             # Phi, OPT
    "c_proj",          # GPT-2 / CodeGen (transformers.Conv1D)
    "output_proj",     # some adapters
    "dense",           # BERT-style second linear
)


def _deterministic_top_singular_vector(M: torch.Tensor, side: str) -> torch.Tensor:
    """Top ``side`` ('left'/'right') singular vector of ``M``, computed
    DETERMINISTICALLY and accurately.

    The previous implementation used ``torch.svd_lowrank(q=1, niter=2)``
    seeded from the global (un-reseeded) RNG, so repeated runs on identical
    input gave wildly different vectors and, on flat spectra, |cos| with the
    true top vector as low as ~0.07. This helper:
      * uses an exact SVD when the matrix is small enough; otherwise
      * a randomized SVD with oversampling (q=8) and niter=7 — which reaches
        |cos| ~ 0.99 with the exact top vector — SEEDED via save/restore of
        the RNG state so global determinism elsewhere is untouched.

    Sign is irrelevant downstream (the task uses |cos|).
    """
    k = int(min(M.shape))
    if k <= 1:
        return None
    # Exact path when cheap (<= ~4M elements).
    if int(M.shape[0]) * int(M.shape[1]) <= 4_000_000:
        U, _S, Vh = torch.linalg.svd(M, full_matrices=False)
        return U[:, 0] if side == "left" else Vh[0]
    # Large matrix: seeded, oversampled randomized SVD.
    cpu_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        torch.manual_seed(0)
        q = min(8, k)
        U, _S, V = torch.svd_lowrank(M, q=q, niter=7)
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    return U[:, 0] if side == "left" else V[:, 0]


def _is_projection(m) -> bool:
    if isinstance(m, torch.nn.Linear):
        return True
    if _HFConv1D is not None and isinstance(m, _HFConv1D):
        return True
    return False


def _find_mlp_projection(mlp) -> Optional[torch.nn.Module]:
    """Return the MLP's output projection module, or None."""
    for name in _OUT_PROJ_NAMES:
        proj = getattr(mlp, name, None)
        if proj is not None and _is_projection(proj):
            return proj
    # Fallback: last projection submodule in the MLP.
    last = None
    for sub in mlp.modules():
        if _is_projection(sub):
            last = sub
    return last


def _weight_top_left_singular(proj: torch.nn.Module) -> Optional[torch.Tensor]:
    """Top left-singular vector of the projection's weight, in the
    module's *input* feature space, computed on the projection's own
    device via a randomised rank-1 SVD (much faster than a full SVD on
    large intermediate widths).

    ``torch.nn.Linear`` stores ``weight`` with shape ``(out, in)`` and
    applies ``x @ W.T``. ``transformers.Conv1D`` stores ``weight`` with
    shape ``(in, out)`` and applies ``x @ W``. We flip the Linear
    weight to ``(in, out)`` so ``U[:, 0]`` always lives in the input
    feature axis regardless of the layer type.
    """
    W = proj.weight.detach().float()
    if W.dim() != 2:
        return None
    if _HFConv1D is not None and isinstance(proj, _HFConv1D):
        pass  # (in, out) already
    else:
        W = W.T  # (out, in) → (in, out)
    try:
        # Deterministic, accurate top left-singular vector (in input space).
        return _deterministic_top_singular_vector(W, side="left")
    except Exception as e:
        logger.info(f"  WAA SVD failed: {type(e).__name__}: {e}")
        return None


def _activation_top_principal(acts: torch.Tensor) -> Optional[torch.Tensor]:
    """Top principal component of the centered activation matrix.

    ``acts`` has shape ``(N, D)``. Uses randomised rank-1 SVD for speed
    on large intermediate widths — a full SVD of (4k, 14k) on CPU
    takes tens of seconds per layer, which was the dominant cost of
    this task before the fix.
    """
    if acts.numel() == 0 or acts.shape[0] < 2:
        return None
    acts = acts.float()
    acts = acts - acts.mean(dim=0, keepdim=True)
    try:
        # Deterministic, accurate top right-singular vector (max-variance
        # direction in feature space).
        return _deterministic_top_singular_vector(acts, side="right")
    except Exception as e:
        logger.info(f"  WAA activation SVD failed: {type(e).__name__}: {e}")
        return None


@register_task("interpretability_waa")
class WeightActivationAlignmentTask(DiagnosticTask):
    """Cosine between weight SVD and activation PCA per layer."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Weight-Activation Alignment (single-pass)...")
        num_samples = int(self.config.get("num_samples", 5))
        max_tokens_per_layer = int(self.config.get("max_tokens", 4096))
        max_length = int(self.config.get("max_length", 128))

        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)
        samples = list(dataset)[:num_samples]
        if not samples:
            return {"error": "Need at least 1 sample."}

        device = next(model.parameters()).device
        layers = get_layers(model)
        if layers is None:
            return {"error": "Could not detect model layers."}

        targets: List[Tuple[int, torch.nn.Module]] = []
        for l_idx, layer in enumerate(layers):
            mlp = (
                getattr(layer, "mlp", None)
                or getattr(layer, "feed_forward", None)
                or getattr(layer, "output", None)
            )
            if mlp is None:
                continue
            proj = _find_mlp_projection(mlp)
            if proj is not None:
                targets.append((l_idx, proj))

        if not targets:
            return {
                "error": (
                    "Could not identify MLP output projections for "
                    "WAA on this architecture."
                )
            }

        # Hook every target simultaneously. We use a pre-forward hook so
        # we intercept the projection's INPUT — the MLP intermediate
        # activation, in the same feature space as U[:, 0] above.
        collected: Dict[int, List[torch.Tensor]] = {li: [] for li, _ in targets}
        # Seeded RNG so the sub-sampled token set (and therefore the
        # reported alignment) is deterministic across reruns of the
        # same model/corpus. Without this, `torch.randperm` draws from
        # the global PyTorch RNG and the result drifts between
        # invocations even when `set_global_seed` has been called at
        # the start of the pipeline.
        rng = torch.Generator(device="cpu").manual_seed(
            int(self.config.get("seed", 0))
        )

        def make_hook(li: int, budget: int):
            def pre_hook(module, args):
                if not args:
                    return
                x = args[0]
                if not isinstance(x, torch.Tensor):
                    return
                # Keep activations on the model's device so the SVD can
                # run on GPU (100-1000× faster than CPU SVD of a
                # (4k, 14k) matrix — which was the dominant cost of
                # this task prior to the speedup fix).
                flat = x.detach().float().reshape(-1, x.shape[-1])
                # Early subsample per batch to bound memory.
                if flat.shape[0] > budget:
                    cpu_rand = torch.randperm(flat.shape[0], generator=rng)[:budget]
                    idx = cpu_rand.to(flat.device)
                    flat = flat[idx]
                collected[li].append(flat)
            return pre_hook

        # Per-layer budget: total ≤ max_tokens_per_layer across the
        # corpus, so a model with more tokens doesn't blow up memory.
        handles = [
            proj.register_forward_pre_hook(make_hook(li, max_tokens_per_layer))
            for li, proj in targets
        ]

        try:
            with torch.no_grad():
                for s in samples:
                    text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                    inputs = tokenizer(
                        text, return_tensors="pt",
                        truncation=True, max_length=max_length,
                    ).to(device)
                    model(**inputs)
        finally:
            for h in handles:
                h.remove()

        alignments: Dict[int, float] = {}
        for l_idx, proj in targets:
            chunks = collected.get(l_idx, [])
            if not chunks:
                continue
            acts = torch.cat(chunks, dim=0)
            if acts.shape[0] > max_tokens_per_layer:
                cpu_rand = torch.randperm(
                    acts.shape[0], generator=rng,
                )[:max_tokens_per_layer]
                acts = acts[cpu_rand.to(acts.device)]

            u_weight = _weight_top_left_singular(proj)
            v_act = _activation_top_principal(acts)
            if u_weight is None or v_act is None:
                continue
            if u_weight.shape[-1] != v_act.shape[-1]:
                logger.info(
                    f"  WAA layer {l_idx}: dimension mismatch "
                    f"(weight={u_weight.shape[-1]}, act={v_act.shape[-1]})"
                )
                continue

            # Both vectors can now live on any device — move them to a
            # common device before the dot product.
            u = u_weight.detach().float().flatten()
            v = v_act.detach().float().flatten().to(u.device)
            cos_sim = torch.dot(u, v)
            # |cos| — the sign is arbitrary; we only care about the axis.
            alignments[l_idx] = float(torch.abs(cos_sim).item())

        if not alignments:
            return {"error": "Failed to collect activations for WAA."}

        return {
            "mean_waa_alignment": float(np.mean(list(alignments.values()))),
            "layer_waa_alignments": {str(k): v for k, v in alignments.items()},
            "n_layers_analyzed": len(alignments),
        }
