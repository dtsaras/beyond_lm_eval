"""Matrix Entropy (Wei et al. 2024, arXiv:2401.17139).

Computes the **per-sentence** von Neumann entropy of the centered-and-
row-normalised token covariance, then averages over sentences and
normalises by ``log d`` (paper Def. 4.3).

Rewrite history:
  * 2026-04-17: added per-sentence aggregation, **row-wise L2
    normalisation after centering**, and ``/ log d`` normalisation to
    match Def. 4.1 and Def. 4.3 of Wei et al. The original BLME
    implementation pooled every token from every sample into a single
    covariance matrix per layer and skipped both the row normalisation
    and the ``log d`` divisor — so the reported headline number was
    architecture-dependent and dominated by whichever tokens happened to
    have large activation norm (a known issue for LLM hidden states).

References:
  * Wei, Tan, Li, Wang, Huang, "Large Language Model Evaluation via
    Matrix Entropy", arXiv:2401.17139 (2024).
  * Official reference code: https://github.com/waltonfuture/Matrix-Entropy
    (now `Diff-eRank`).
"""

import logging
import math

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from .utils import pow2_rescale as _pow2_rescale

logger = logging.getLogger("blme")


def _matrix_entropy(Z: torch.Tensor) -> dict:
    """Compute Wei et al. matrix entropy for a single sentence's hidden
    states ``Z ∈ R^{N_tokens × d}``.

    Steps, matching Def. 4.1–4.3 of the paper:
      1. Center: ``Z ← Z − mean(Z, dim=0)``.
      2. Row L2 normalise: ``Z ← Z / ‖Z_row‖₂``.
      3. Covariance: ``A = Zᵀ Z / N`` (d × d).
      4. Unit trace: ``A ← A / tr(A)``.
      5. Entropy: ``H = − Σ λ log λ`` over eigenvalues of ``A``.
      6. Normalised entropy: ``H / log d`` (Def. 4.3).

    Returns a dict with both the raw and normalised entropies so the
    caller can decide which to report. NaN is returned when the input
    is degenerate (< 2 tokens, zero rows after normalisation, zero
    trace, etc.).
    """
    if Z.numel() == 0 or Z.shape[0] < 2:
        return {"entropy": float("nan"), "entropy_normalized": float("nan")}
    d = int(Z.shape[1])
    if d < 2:
        return {"entropy": float("nan"), "entropy_normalized": float("nan")}

    # DOUBLE, not float. This used to cast to float32 here, which quantised
    # the data before the centring below could cancel a common offset -- the
    # payload of `1e8 + 0.125` is simply not present in float32. Promoting a
    # float32/bf16 input to double is exact and only ever removes error.
    Z = Z.double()
    # If the caller handed us a CPU tensor and CUDA is available,
    # move to GPU for the SVD — at D = 8192 (Qwen-9B) a CPU SVD of
    # (128, D) takes ~1 s while the GPU one takes <10 ms; this was
    # the dominant cost when the task timed out on the v2 run.
    if not Z.is_cuda and torch.cuda.is_available():
        try:
            Z = Z.cuda(non_blocking=True)
        except Exception:
            pass
    # RESCALE BEFORE CENTRING. The row-max rescale below is applied after the
    # mean subtraction, so centring itself can lose the data: at
    # q = FLT_TRUE_MIN the headline read 0.0743019 against 0.4525896 (-83.6%)
    # -- the same ordering bug round 4 fixed in `schatten._normalise_matrix`
    # (independent Opus review, 2026-09-04). Dividing by the global maximum
    # only shifts exponents, so it is exact and cancels in the entropy.
    # CENTRE IN DOUBLE, FIRST. Then rescale, then come back down.
    #
    # The order here has been wrong in both directions. Centring after the
    # float32 rescale lost the data at denormal scales (round 4); centring
    # after the float32 CAST loses it under a common offset, which is the
    # opposite failure and the one that survived. Wei et al. centre every token
    # vector, so adding a common vector must leave the entropy unchanged. It
    # did not: on the 4x2 fixture [[-0.125,-0.375],[-0.3125,-0.1875],
    # [-0.5,0.125],[-0.25,-0.3125]] the entropy went 0.467692434788 ->
    # 0.092731282115 at +1e6 (**-80.17%**) and NaN at +1e8, while an
    # independent evaluation is unchanged (gpt-5.6-sol round 9, 2026-09-07).
    #
    # The cause is quantisation, not the centring: in float32, 1e6 + 0.125
    # keeps ~7 digits, so the payload is gone before the mean is subtracted.
    # Centred in float32 the fixture gives [0.1875, -0.1875, 0.0, 0.0]; in
    # float64 it gives [0.171875, -0.1875, -0.015625, 0.0].
    #
    # Doing the subtraction in double fixes the offset case, and doing it
    # BEFORE the rescale (rather than after) leaves the rescale free to bring
    # the CENTRED data into range, which is what the denormal case needed. One
    # order now satisfies both, so neither fix has to be traded against the
    # other.
    Z = Z - Z.mean(dim=0, keepdim=True)

    _fi0 = torch.finfo(torch.float32)
    _gmax0 = Z.abs().max()
    _lo0 = float(_fi0.tiny) ** 0.5 * 4.0
    _hi0 = (float(_fi0.max) / max(1, Z.shape[1])) ** 0.5 / 4.0
    if float(_gmax0) > 0 and (float(_gmax0) < _lo0 or float(_gmax0) > _hi0):
        # POWER OF TWO. Dividing by the actual maximum introduces rounding, so
        # whether the guard fired became observable in the result -- the same
        # fixture gave a different entropy depending on which SCALE ROUTE
        # reached it (codex round 5, 2026-09-05). A power-of-two divisor shifts
        # exponents only, so the two routes now agree exactly.
        Z = _pow2_rescale(Z, float(_gmax0))
    # Back to float32 for the SVD, now that the data is centred and in range.
    Z = Z.float()

    # Row L2 normalisation — Def. 4.1.
    # Scale each row by its own largest coordinate BEFORE norming: the norm
    # squares, so it underflows in float32 long before the data does. At
    # alpha = 3e-23 the tiny-row norms came out 3.743e-23 and the normalised
    # entropy read 0.482747648 instead of its scale-free value (adversarial
    # review round 2, 2026-09-04). This is exact -- it only shifts exponents.
    _rmax = Z.abs().amax(dim=1, keepdim=True)
    Z = Z / _rmax.clamp_min(torch.finfo(Z.dtype).tiny)
    Z = torch.where(_rmax > 0, Z, torch.zeros_like(Z))
    row_norms = Z.norm(p=2, dim=1, keepdim=True)
    # Def. 4.1 normalises every row and the released code divides directly, so
    # an absolute 1e-12 floor makes the metric discontinuous in the data scale:
    # at eps=1.1e-12 the four-row anchor is 1.000000003, and at 1.0e-12 it
    # collapsed to 9.97e-29 while the reference stayed 1.000000003
    # (adversarial review, 2026-09-04). Only an exactly zero row is absent.
    keep = row_norms.squeeze(-1) > 0
    if int(keep.sum().item()) < 2:
        return {"entropy": float("nan"), "entropy_normalized": float("nan")}
    Z = Z[keep] / row_norms[keep]

    # A = Zᵀ Z / N. Use SVD on Z to avoid materialising a d×d matrix
    # when N < d. Squared singular values / N are the eigenvalues of A.
    S = torch.linalg.svdvals(Z)
    lam = (S ** 2) / float(Z.shape[0])
    trace = float(lam.sum().item())
    if not math.isfinite(trace) or trace <= 0:
        return {"entropy": float("nan"), "entropy_normalized": float("nan")}
    lam = lam / trace

    # − Σ λ log λ with the 0·log 0 = 0 convention.
    lam = lam.clamp(min=1e-30)
    H = float(-torch.sum(lam * torch.log(lam)).item())

    return {
        "entropy": H,
        "entropy_normalized": H / math.log(max(d, 2)),
    }


def _collect_per_sample_layer_tokens(cache, num_samples, use_cache,
                                     replace_final_norm=True):
    """Return ``{layer_idx: List[Tensor(T_i, D)]}`` — one tensor per
    sentence per layer — from the shared cache if available.

    ``replace_final_norm`` is a PARAMETER here, and this task needs BOTH
    settings, which is why. See the note at the call site.
    """
    if cache is None or not cache.is_populated or not use_cache:
        return None
    hs = cache.get_hidden_states(
        layer_idx="all", num_samples=num_samples, per_sample=True,
        replace_final_norm=replace_final_norm,
    )
    if not hs:
        return None
    return {li: chunks for li, chunks in hs.items() if chunks}


def _collect_per_sample_layer_tokens_fresh(model, tokenizer, samples, max_length):
    """Fallback path: run our own forward pass sample-by-sample and
    return ``{layer_idx: List[Tensor(T_i, D)]}``.

    Keeps the collected tensors on the model's device so the downstream
    per-sentence SVD runs on GPU — GPU SVD of a ``(128, D)`` matrix is
    100-1000× faster than CPU SVD at D=8192+, which was the dominant
    cost when this task timed out on the v2 run.
    """
    device = next(model.parameters()).device
    per_layer: dict[int, list[torch.Tensor]] = {}
    # post-final-norm: BOTH tensors are returned, because this task needs both.
    # `per_layer` is the hooked, homogeneous PROFILE (every entry a block
    # output); `post_norm_last` is the untouched `ln_f(b_{L-1})` entry, which is
    # what Wei et al.'s released recipe measures and what the HEADLINE must
    # stay. See the note at the call site.
    post_norm_last: list[torch.Tensor] = []
    _sink = {}
    _hook = None
    try:
        from ..common import get_layers
        _blocks = get_layers(model)
        if _blocks is not None and len(_blocks) > 0:
            _hook = _blocks[-1].register_forward_hook(
                lambda mod, inp, outp: _sink.__setitem__(
                    "h", (outp[0] if isinstance(outp, tuple) else outp).detach()))
    except Exception as e:  # pragma: no cover - defensive
        logger.info(f"matrix_entropy: could not hook last block ({e})")
    # try/FINALLY. The straight-line remove below covered the happy path
    # only: an exception anywhere in the loop left the hook attached to
    # `layers[-1]` for the life of the process. Confirmed by an
    # independent review on gpt2 and pythia-70m (0 -> 1 after a raise) --
    # the same bug fixed in `perturbation_growth` the same day, in a task
    # the leak test claimed to cover but did not.
    try:
        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(
                    text, return_tensors="pt",
                    truncation=True, max_length=max_length,
                ).to(device)
                _sink.pop("h", None)
                out = model(**inputs, output_hidden_states=True)
                if not getattr(out, "hidden_states", None):
                    continue
                # post-final-norm: BOTH. `per_layer` is substituted (homogeneous
                # profile); `post_norm_last` keeps the untouched `ln_f` entry,
                # which is the headline the released reference defines.
                hs = list(out.hidden_states)
                post_norm_last.append(hs[-1][0].detach().float())
                _h_raw = _sink.get("h")
                if (_h_raw is not None and len(hs) >= 2
                        and _h_raw.shape == hs[-1].shape):
                    hs[-1] = _h_raw
                for li, h in enumerate(hs[1:]):  # skip embedding
                    per_layer.setdefault(li, []).append(
                        h[0].detach().float()
                    )
    finally:
        if _hook is not None:
            _hook.remove()
    return per_layer, post_norm_last


@register_task("geometry_matrix_entropy")
class MatrixEntropyTask(DiagnosticTask):
    """Per-sentence-averaged Wei et al. 2024 matrix entropy per layer.

    Canonical "Wei et al. number" is the entropy of the **last** hidden
    layer; we report every layer too as a convenient extension but flag
    the last-layer value as the headline metric.
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Matrix Entropy (Wei et al. 2024)...")
        num_samples = int(self.config.get("num_samples", 10))
        use_cache = self.config.get("use_cache", True)
        # Default 512 to match the shared prediction cache's truncation
        # (cache.py:352) so the fresh and cached paths produce the same entropy
        # for the same config. (The official Wei et al. protocol used 2048; the
        # cache fixes 512 for all consumers, so we align to it.)
        max_length = int(self.config.get("max_length", 512))

        # THE FLAG IS SPLIT, AND THIS TASK IS WHY THE RULE HAS TWO HALVES.
        #
        # The 2026-09-05 sweep put `replace_final_norm=True` on both routes and
        # BROKE THE HEADLINE. Wei et al.'s released recipe (waltonfuture/
        # Diff-eRank README, verbatim) is
        #     model = AutoModel.from_pretrained(path); R = model(input_ids)[0]
        # i.e. `last_hidden_state`, which is BIT-IDENTICAL to
        # `hidden_states[-1] = ln_f(b_{L-1})` -- max abs diff 0.0 on gpt2 and
        # pythia-70m. Running that recipe on 10 WikiText passages gives gpt2
        # 0.2555779059 and pythia-70m 0.5925999933; BLME matched to rel 4.4e-06
        # and 2.5e-05 BEFORE the sweep and returned 0.4939811964 (+93.3%) and
        # 0.4573149317 (-22.8%) AFTER it. It also destroyed the one published
        # comparison the metadata claims: the pythia 70m-vs-160m gap went from
        # 0.0450 (+6.3% vs the published 0.0423) to 0.0869 (+105.5%)
        # (independent Opus review, 2026-09-05).
        #
        # Both halves of the sweep's own rule apply here, to DIFFERENT outputs:
        #   * the per-layer PROFILE must be homogeneous  -> hooked
        #   * the HEADLINE is a SINGLE-LAYER estimand defined by a released
        #     reference as the post-norm state              -> NOT hooked
        # So both tensors are collected. Emitting one and calling it the other
        # is the actual defect, and it is not fixed by picking a side.
        per_layer_tokens = _collect_per_sample_layer_tokens(
            cache, num_samples, use_cache, replace_final_norm=True,
        )
        _postnorm_tokens = None

        if per_layer_tokens is None:
            if dataset is None:
                from ...cache import load_default_corpus
                dataset = load_default_corpus(num_samples)
            samples = list(dataset)[:num_samples]
            if not samples:
                return {"error": "Need at least 1 sample."}
            per_layer_tokens, _postnorm_tokens = (
                _collect_per_sample_layer_tokens_fresh(
                    model, tokenizer, samples, max_length))
        else:
            # Same cache, same documents, the untouched deepest entry.
            _pn = _collect_per_sample_layer_tokens(
                cache, num_samples, use_cache, replace_final_norm=False)
            if _pn:
                _postnorm_tokens = _pn[max(_pn.keys())]

        if not per_layer_tokens:
            return {"error": "Could not collect hidden states."}

        layer_entropies: dict[str, float] = {}
        layer_entropies_norm: dict[str, float] = {}
        for li in sorted(per_layer_tokens.keys()):
            per_sentence = []
            per_sentence_norm = []
            for Z in per_layer_tokens[li]:
                res = _matrix_entropy(Z)
                if np.isfinite(res["entropy"]):
                    per_sentence.append(res["entropy"])
                if np.isfinite(res["entropy_normalized"]):
                    per_sentence_norm.append(res["entropy_normalized"])
            if per_sentence:
                layer_entropies[f"layer_{li}"] = float(np.mean(per_sentence))
            if per_sentence_norm:
                layer_entropies_norm[f"layer_{li}"] = float(np.mean(per_sentence_norm))

        finite_raw = [v for v in layer_entropies.values() if np.isfinite(v)]
        finite_norm = [v for v in layer_entropies_norm.values() if np.isfinite(v)]
        if not finite_raw:
            return {"error": "All layers produced non-finite entropy."}

        # Headline = the POST-NORM last-layer entropy, matching Wei et al.'s
        # released recipe exactly (see the note at collection). The per-layer
        # profile above is the hooked, homogeneous one; these two are different
        # tensors on purpose.
        last_key = max(layer_entropies.keys(), key=lambda k: int(k.split("_")[1]))
        _hl, _hl_norm = [], []
        for Z in (_postnorm_tokens or []):
            res = _matrix_entropy(Z)
            if np.isfinite(res["entropy"]):
                _hl.append(res["entropy"])
            if np.isfinite(res["entropy_normalized"]):
                _hl_norm.append(res["entropy_normalized"])
        if _hl:
            headline = float(np.mean(_hl))
            headline_norm = float(np.mean(_hl_norm)) if _hl_norm else float("nan")
            _headline_is_postnorm = True
        else:
            # No post-norm tensor available (a model whose blocks `get_layers`
            # cannot find, so nothing was substituted and the two are the same
            # thing anyway). Fall back to the profile's deepest entry and SAY SO.
            headline = layer_entropies[last_key]
            headline_norm = layer_entropies_norm.get(last_key, float("nan"))
            _headline_is_postnorm = False

        return {
            # Headline "Wei et al." number for the model = the NORMALISED
            # entropy, which is Definition 4.3 and the quantity their tables
            # report. Confirmed head-to-head on their protocol: pythia-70m
            # published 0.7003 vs normalised 0.6659 (-4.9%), against raw
            # H = 4.1542 (+493%). The unnormalised H is bounded by
            # log(min(N-1, d)), so shipping it as the headline also made this
            # feature a much stronger width proxy than the paper's quantity.
            # Provenance: True only when the headline really is the post-`ln_f`
            # state the reference defines. Derived, never asserted.
            "_meta_headline_is_post_final_norm": bool(_headline_is_postnorm),
            # The PROFILE's provenance, which was missing while the
            # headline had one -- and this task splits the flag, so the
            # two answers genuinely differ. Both siblings emit it.
            "_meta_last_entry_is_block_output": bool(
                per_layer_tokens is not None and _postnorm_tokens is not None),
            "matrix_entropy": headline_norm,
            # THE `_meta_` ALIAS IS GONE, because it was a NO-OP.
            #
            # I "demoted" H/log(d) to `_meta_matrix_entropy_normalized` on
            # 2026-09-04 -- but `matrix_entropy` IS H/log(d), so the identical
            # float shipped under both names (gpt2: 0.25557566658278186 for
            # each) and the quantity the comment said was "kept out of the
            # feature matrix" was the unprefixed headline all along
            # (independent Opus review, 2026-09-05).
            #
            # The demotion was also the wrong call: `matrix_entropy` must stay
            # the normalised H, because that IS Wei et al. Definition 4.3 and
            # it is what reproduces their published table (pythia-70m 0.7003 vs
            # 0.6659). The size-neutrality concern is real but belongs to
            # `matrix_entropy_raw` below, which is the column cross-model work
            # should use. Two names for one number helped nobody.
            # UNPREFIXED, because measurement inverted my stated reason for
            # demoting it.
            #
            # I demoted raw H on the argument that "H is bounded by
            # log(min(N-1, d)), so it is a stronger width proxy than the
            # paper's quantity". At the shipped protocol that is BACKWARDS. The
            # mean document is ~91 tokens, so the cap is log(N-1) -- CONSTANT
            # in width -- while the divisor log(d) is not. Measured: raw H has
            # Spearman +0.033 (d_model) / +0.049 (n_layers) on the 66-model
            # archive and +0.155 / -0.196 on a fresh 20-model panel, while the
            # DIVIDED headline has -0.483 / -0.339 and -0.247 / -0.004. So
            # dividing by log d introduces a width dependence -- the same
            # pattern as the schatten row_normalized keys, here inside Wei et
            # al.'s own Definition 4.3 (adversarial review, 2026-09-04).
            #
            # `matrix_entropy` stays the normalised H: it IS Definition 4.3 and
            # it is what reproduces the published table (pythia-70m 0.7003 vs
            # 0.6659). But cross-model work should use THIS column, which is
            # the size-neutral one at the shipped protocol.
            "matrix_entropy_raw": headline,
            # Per-layer extension (same metric applied to every block).
            "layer_matrix_entropies": layer_entropies,
            "layer_matrix_entropies_normalized": layer_entropies_norm,
            # Backwards-compatible aggregate across layers.
            # Same correction for the cross-layer aggregate.
            # Same de-duplication as the headline above: `mean_matrix_entropy`
            # IS the normalised mean, so a `_meta_..._normalized` alias carried
            # the identical float under a second name. The RAW mean keeps its
            # own name because it is a different number.
            "mean_matrix_entropy": (
                float(np.mean(finite_norm)) if finite_norm else float("nan")
            ),
            "mean_matrix_entropy_raw": float(np.mean(finite_raw)),
            "interpretation": (
                "Lower last-layer matrix entropy indicates a stronger "
                "information bottleneck (Wei et al., 2024)."
            ),
        }
