"""Layer perturbation growth — depth-wise sensitivity (chaoticity) profile.

Grounding:
  * Li et al. 2025 — "Cognitive Activation and Chaotic Dynamics in Large
    Language Models: A Quasi-Lyapunov Analysis of Reasoning Mechanisms",
    arXiv:2503.13530. Defines a quasi-Lyapunov exponent: the growth rate
    of hidden-state divergence across layers under small perturbations.
  * Fernando & Guitchounts 2025/2026 — residual-stream dynamical-systems
    line (arXiv:2502.12131, 2605.14258): layer-Jacobian spectra shape how
    perturbations amplify or contract through depth.

BLME's clean-room implementation (documented BLME formulation, cited
grounding above): perturb the input embeddings with a SEEDED Gaussian of
relative size ``rel_eps`` (scaled per token to rel_eps * ||h_token||), run
the clean and perturbed forward passes, and measure the per-layer
divergence d_l = E_t ||h_l - h'_l||. Reported:

  * ``perturbation_growth_per_layer``: log(d_l / d_0) profile.
  * ``growth_slope``: least-squares slope of log d_l against NORMALIZED
    depth in [0, 1] — a quasi-Lyapunov exponent per unit depth. Positive =
    perturbations amplify (chaotic regime); negative = contract.
  * ``growth_log_ratio``: log(d_L / d_0).

Analytic anchor: for a purely linear layer stack h_{l+1} = a * h_l the
divergence grows exactly by |a| per layer, so growth_slope = (L-1) * ln|a| /
(normalized-depth span) and growth_log_ratio = (L-1) * ln|a|.
"""

import logging

import numpy as np
import torch

from ...registry import register_task
from ..geometry.utils import block_output_count
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


def _growth_profile(clean_layers, perturbed_layers) -> dict:
    """Divergence profile from two lists of per-layer (T, D) hidden states."""
    L = min(len(clean_layers), len(perturbed_layers))
    if L < 2:
        return {}
    # DOUBLE, on both sides of the subtraction.
    #
    # Li et al. Eq. 9 is homogeneous of degree zero, so a positive rescale must
    # cancel exactly, and the perturbation is a DIFFERENCE, so a common
    # translation must cancel too. Neither did, and for two different reasons
    # that the same cast causes:
    #
    #  * `.norm()` squares before summing, so in float32 it underflows to 0 at
    #    1e-30/1e-25 and overflows at 1e25/2.406e38. `d` then contains a zero
    #    or a non-finite value, the guard below returns `{}`, and the task
    #    emits **no metric at all** rather than a wrong one. On the
    #    one-coordinate trajectory delta_l = s * 0.5^l the exact rate is
    #    -0.693147180560 for every positive s; BLME returned it at 1e-13, 1 and
    #    1e13 and nothing at the other four rungs.
    #  * casting to float32 BEFORE subtracting quantises away a common offset:
    #    a +1e6 translation moved an exact 0.139761942549 to 0.110812307057
    #    (**-20.71%**). Same defect as `matrix_entropy`, same day.
    #
    # float64 covers the whole ladder on its own here -- squaring 2.4e38 gives
    # 5.8e76 and squaring 1e-30 gives 1e-60, both comfortably inside double's
    # range -- so no rescale is needed, only the promotion.
    # (gpt-5.6-sol round 9, 2026-09-07.)
    d = []
    for l in range(L):
        diff = (clean_layers[l].double() - perturbed_layers[l].double())
        d.append(float(diff.norm(dim=-1).mean().item()))
    d = np.asarray(d, dtype=np.float64)
    if d[0] <= 0 or not np.all(np.isfinite(d)) or np.any(d <= 0):
        return {}
    log_d = np.log(d)
    x = np.arange(L, dtype=np.float64) / max(1, L - 1)
    slope = float(np.polyfit(x, log_d, 1)[0])
    # PER-BLOCK exponent (the source paper's Eq. 9 normalisation, 1/(n-m)).
    #
    # Both `growth_slope` and `growth_log_ratio` carry a depth factor:
    # log_ratio accumulates over every block traversed, so a deeper model
    # scores higher for being deeper rather than for amplifying more per block.
    # The paper's own quantity is the average log-growth PER BLOCK, i.e.
    # lambda = (log d_L - log d_0) / (L - 1), which is what a Lyapunov-style
    # exponent means and is one division away.
    n_transitions = max(1, L - 1)
    return {
        "perturbation_growth_per_layer": (log_d - log_d[0]).tolist(),
        "growth_slope": slope,
        "growth_log_ratio": float(log_d[-1] - log_d[0]),
        "growth_rate_per_block": float((log_d[-1] - log_d[0]) / n_transitions),
        "initial_divergence": float(d[0]),
        "_meta_n_transitions": int(n_transitions),
    }


@register_task("dynamics_perturbation_growth")
class PerturbationGrowthTask(DiagnosticTask):
    """Seeded input-embedding perturbation; per-layer divergence growth."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Layer Perturbation Growth (quasi-Lyapunov)...")
        num_samples = self.config.get("num_samples", 10)
        # 1e-2, not 1e-3. `rel_eps` is a RELATIVE displacement, and bf16 cannot
        # represent one of 1e-3: its eps is 7.812e-03, so adding `1e-3 * ||E||`
        # to a bf16 embedding rounds part of the displacement away. Measured
        # realized ||dE||/||E|| against the request (this file's own harness,
        # 2026-09-05):
        #
        #             requested 1e-3        1e-2          5e-2
        #   float32   1.000e-03 (100%)   1.000e-02     5.000e-02
        #   bfloat16  8.166e-04 ( 82%)   1.013e-02     5.002e-02
        #   float16   1.022e-03 (102%)   1.000e-02     5.000e-02
        #
        # So at 1e-3 a bf16 checkpoint was perturbed 18% LESS than an fp32 one
        # for the same config -- a cross-model difference produced by STORAGE
        # FORMAT. The ratio metrics divide by `d[0]` and absorb the first-order
        # part of that, but `initial_divergence` does not, and the growth is
        # not linear in the displacement. At 1e-2 all three dtypes realize the
        # request exactly.
        rel_eps = float(self.config.get("rel_eps", 1e-2))
        seed = int(self.config.get("seed", 42))

        if dataset is None:
            dataset = [{"text": f"Perturbation growth probe sentence {i}."}
                       for i in range(num_samples)]

        device = next(model.parameters()).device
        try:
            emb_layer = model.get_input_embeddings()
        except Exception:
            return {"error": "Model does not expose input embeddings"}
        if emb_layer is None:
            return {"error": "Model does not expose input embeddings"}

        gen = torch.Generator(device="cpu").manual_seed(seed)
        clean_acc, pert_acc = None, None
        _realized = []
        from ..common import get_layers as _gl
        _blocks = _gl(model)
        _sink, _used_hook, _hook = {}, [False], None
        if _blocks is not None and len(_blocks) > 0:
            def _cap_last(module, inputs, output):
                _sink["h"] = (output[0] if isinstance(output, tuple)
                              else output).detach()
            _hook = _blocks[-1].register_forward_hook(_cap_last)
        n_used = 0
        # try/FINALLY, not a straight-line remove. The first fix moved
        # `_hook.remove()` above the two early returns, which closed the
        # "no usable samples" and "degenerate profile" paths and left the
        # EXCEPTION path wide open. Measured directly: raise inside the loop
        # and `layers[-1]._forward_hooks` goes 0 -> 1 and STAYS there for the
        # life of the process, so every later task in the run pays the hook on
        # every forward and the last activation tensor is pinned alive. The
        # sibling `coe.py` already used try/finally; this one did not
        # (independent Opus review, 2026-09-05).
        try:
            with torch.no_grad():
                for i, s in enumerate(dataset):
                    if i >= num_samples:
                        break
                    text = s["text"] if isinstance(s, dict) else str(s)
                    enc = tokenizer(text, return_tensors="pt",
                                    truncation=True, max_length=128).to(device)
                    if enc["input_ids"].shape[1] < 2:
                        continue
                    E = emb_layer(enc["input_ids"])  # (1, T, D)
                    noise = torch.randn(E.shape, generator=gen).to(
                        device=E.device, dtype=E.dtype)
                    # Per-token relative scaling so every token is perturbed by
                    # rel_eps of its own norm.
                    # `safe_norm`-style rowwise scaling, not `.norm().clamp_min(1e-12)`:
                    # an ABSOLUTE floor on a scale-free normalisation, and a norm
                    # that squares before summing, are the two mechanisms behind
                    # this campaign's most repeated numeric defect.
                    from ..geometry.utils import safe_unit_rows_torch
                    _rmax = E.abs().amax(dim=-1, keepdim=True)
                    _tiny = torch.finfo(E.dtype).tiny
                    scale = (_rmax * (E / _rmax.clamp_min(_tiny)).norm(
                        dim=-1, keepdim=True)) * rel_eps
                    noise = safe_unit_rows_torch(noise)
                    E_pert = E + noise * scale
                    # The REALIZED displacement, measured rather than assumed --
                    # see the dtype note at `rel_eps`.
                    _realized.append(float(
                        ((E_pert - E).float().norm(dim=-1)
                         / E.float().norm(dim=-1).clamp_min(_tiny)).mean().item()))

                    kw = {}
                    if "attention_mask" in enc:
                        kw["attention_mask"] = enc["attention_mask"]
                    out_c = model(inputs_embeds=E, output_hidden_states=True, **kw)
                    _sink_c = dict(_sink)
                    out_p = model(inputs_embeds=E_pert, output_hidden_states=True, **kw)
                    _sink_p = dict(_sink)

                    # DROP the final entry: for most architectures HuggingFace's
                    # last hidden_states element has already passed through the
                    # final norm, so treating it as a block output makes the last
                    # "layer" a LayerNorm. Measured impact here is the largest of
                    # any task: mean |42%| change in growth_log_ratio (gpt2-large
                    # -68.6%, 0.886 -> 2.826), the SIGN of the change is
                    # architecture-determined (all GPT-2 contract, all Pythia/Qwen
                    # expand), and spearman(as-reported, corrected) = +0.571 --
                    # i.e. 43% of the cross-model ordering was a normalization
                    # (adversarial review, 2026-09-03).
                    # Recover the deepest block's RAW output instead of trimming
                    # it away. The trim kept the first L of L+1 entries, which drops
                    # the last BLOCK, so the divergence profile never reached the
                    # deepest layer -- the same defect fixed with a hook in coe.py,
                    # and the same hook applies verbatim here.
                    # post-final-norm: SUBSTITUTED. The divergence PROFILE spans
                    # every layer, and its deepest point is the one the growth
                    # ratio is read off.
                    _hc = list(out_c.hidden_states)
                    _hp = list(out_p.hidden_states)
                    if (_sink_c.get("h") is not None
                            and tuple(_sink_c["h"].shape) == tuple(_hc[-1].shape)):
                        _hc[-1] = _sink_c["h"]
                        _hp[-1] = _sink_p["h"]
                        _used_hook[0] = True
                        hs_c = [h[0].float().cpu() for h in _hc]
                        hs_p = [h[0].float().cpu() for h in _hp]
                    else:
                        _used_hook[0] = False
                        _keep = block_output_count(model)
                        hs_c = [h[0].float().cpu() for h in _hc][:_keep]
                        hs_p = [h[0].float().cpu() for h in _hp][:_keep]
                    if clean_acc is None:
                        clean_acc = [[] for _ in hs_c]
                        pert_acc = [[] for _ in hs_p]
                    for li in range(min(len(hs_c), len(clean_acc))):
                        clean_acc[li].append(hs_c[li])
                        pert_acc[li].append(hs_p[li])
                    n_used += 1

        finally:
            if _hook is not None:
                _hook.remove()
                _hook = None

        if not n_used:
            return {"error": "No usable samples for perturbation growth"}

        clean_layers = [torch.cat(c, dim=0) for c in clean_acc]
        pert_layers = [torch.cat(c, dim=0) for c in pert_acc]
        results = _growth_profile(clean_layers, pert_layers)
        if not results:
            return {"error": "Degenerate divergence profile"}
        results["rel_eps"] = rel_eps
        # MEASURED, not requested. On a bf16 checkpoint at the old rel_eps=1e-3
        # these two differed by 18%; they should now agree to <1%, and a run
        # where they do not is a run whose perturbation size is not what its
        # config says.
        _rz = float(np.mean(_realized)) if _realized else float("nan")
        results["_meta_realized_rel_eps"] = _rz
        if np.isfinite(_rz) and rel_eps > 0 and abs(_rz - rel_eps) / rel_eps > 0.05:
            logger.warning(
                f"  perturbation_growth: requested rel_eps={rel_eps:g} but the "
                f"model realized {_rz:.4g} ({_rz / rel_eps * 100:.1f}%) -- the "
                f"perturbation is below this dtype's resolution "
                f"({torch.finfo(next(model.parameters()).dtype).eps:.3g}). "
                f"Growth is not comparable against a model in a wider dtype.")
        results["n_samples_used"] = n_used
        results["_meta_last_entry_is_block_output"] = bool(_used_hook[0])
        return results
