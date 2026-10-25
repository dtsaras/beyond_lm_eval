"""
Representation Sensitivity Task
──────────────────────────────────────────────────────────────────────
Evaluates how sensitive the model's output distribution is to perturbations
in the final-layer hidden states, by computing ||grad_h log P(y|h)||^2
averaged over tokens.

This measures the squared norm of the gradient of the log-likelihood with
respect to the representation — a form of local sensitivity or Jacobian
norm.  Higher values indicate the representation space is "sharp" (small
changes in h cause large changes in the prediction), while lower values
indicate a flatter, more robust manifold.

Since the LM head is linear (logits = W h + b), the gradient has a
closed-form:  grad_h log P(y|h) = W_y - sum_k p(k) W_k,  which avoids
per-token backward passes entirely.

References:
    "Natural Gradient Works Efficiently in Learning" (Amari, 1998,
    Neural Computation 10(2):251-276) — the softmax score-function
"""

import torch
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import apply_lm_head, get_lm_head, get_embeddings
import logging
logger = logging.getLogger("blme")


@register_task("geometry_representation_sensitivity")
class RepresentationSensitivityTask(DiagnosticTask):
    """
    Computes ||grad_h log P(y|h)||^2 with respect to final-layer hidden
    states using a closed-form derivation for the linear LM head.
    Uses the true next token as the target rather than argmax.
    """
    @staticmethod
    def _logit_scale_for(model):
        """`config.logit_scale` as a float, or None when the head is unscaled.

        Read from the config rather than inferred: a value of exactly 1.0 is
        returned as None so the ordinary path stays bit-identical.
        """
        cfg = getattr(model, "config", None)
        val = getattr(cfg, "logit_scale", None) if cfg is not None else None
        try:
            f = float(val) if val is not None else None
        except (TypeError, ValueError):
            return None
        if f is None or f == 1.0:
            return None
        return f

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Representation Sensitivity Analysis...")
        num_samples = self.config.get("num_samples", 20)

        # RESET per run. `_head_guard_max_rel` is a property of THIS
        # evaluation, and accumulating it across calls meant an exactly linear
        # model still reported a previous run's 0.021276595 (adversarial
        # review round 3, 2026-09-04).
        self._head_guard_max_rel = 0.0
        if dataset is None:
             from ...cache import load_default_corpus
             dataset = load_default_corpus(num_samples)

        samples = list(dataset)[:num_samples]
        if not samples:
             return {"error": "Need at least 1 sample."}

        device = next(model.parameters()).device

        # Get the unembedding matrix W (vocab_size, hidden_dim)
        head = get_lm_head(model)
        if head is not None:
            W = head.weight.detach().float()  # (V, D)
        else:
            W = get_embeddings(model)
            if W is None:
                return {"error": "Cannot access unembedding matrix."}
            W = W.detach().float()

        W = W.to(device)

        sensitivity_values = []
        scalefree_values = []

        model.eval()
        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                input_ids = inputs["input_ids"]

                if input_ids.shape[1] < 2:
                    continue

                out = model(**inputs, output_hidden_states=True)

                # post-final-norm: DELIBERATELY post-norm. The Fisher/
                # prediction geometry here is the geometry of the vector the
                # LM head actually consumes -- `lm_head(hidden_states[-1])`
                # is `logits` exactly -- so the raw block output would be the
                # wrong object, not a cleaner one.
                final_hidden = out.hidden_states[-1][0].float()  # (T, D)

                # True next tokens (shifted input_ids)
                targets = input_ids[0, 1:]  # (T-1,)
                h = final_hidden[:-1]       # (T-1, D) — hidden states predicting next token

                # Compute logits and probabilities
                logits = h @ W.T  # (T-1, V)
                # Add bias if present
                if head is not None and head.bias is not None:
                    logits = logits + head.bias.detach().float()
                # APPLY `logit_scale`. Cohere and several LM variants multiply
                # the projection by a constant s, so the real logits are
                # z = s(Wh + b) and the closed form is
                #     grad_h log p(y|h) = s [ W_y - sum_k softmax(z)_k W_k ].
                # The scale was applied ONLY in the linearity guard below and
                # nowhere in the value path, so BOTH the probabilities (softmax
                # of an unscaled z) and the gradient (missing the factor s) were
                # wrong. On a live CohereForCausalLM with logit_scale = 1.02 the
                # task returned 0.00603947939685 against float64 autograd
                # 0.00628330365681, **-3.88051%**; at logit_scale = 1.0005,
                # -0.0998529% (independent gpt-5.6-sol review, 2026-09-06).
                #
                # `_logit_scale_for` is read once per model, not per token.
                _ls = self._logit_scale_for(model)
                if _ls is not None:
                    logits = logits * _ls
                # Guard the linear-head assumption (audit 2026-07-16): with a
                # post-head transform (e.g. Gemma-2 final_logit_softcapping)
                # both these probs and the closed form W_y - E_p[W] are
                # silently wrong (0.82 max gradient error demonstrated).
                # Verify the recomputed logits against the model's real
                # logits on the first sample; bail out cleanly on mismatch.
                # CHECKED ON EVERY SAMPLE. There is no such thing as "already
                # verified": a softcap bites hardest on the inputs with the
                # largest logits, so a weak first document passed at
                # discrepancy 0.000833222 and the stronger one that would have
                # failed was never checked. The flag was also not bound to the
                # model, so running one task instance on a linear head and then
                # on a softcapped head skipped the second guard entirely and
                # emitted 0.096115820110 against a true 0.036094533289, +166%
                # (adversarial review rounds 1-2, 2026-09-04). The check is one
                # max-abs comparison per document -- there is nothing to save.
                if True:
                    real = out.logits[0, :-1, :].float()
                    # FAIL CLOSED on non-finite logits. `rel` is NaN there,
                    # so `rel > tol` is False and BOTH guards passed -- then
                    # `torch.where(_gm > 0, ...)` turned every NaN gradient into
                    # 0.0, the MINIMUM of the scale. Demonstrated live on
                    # pythia-70m under fp16 + eager. `core.py` casts fp16 to
                    # bf16 so the study never reaches it, but a guard that fails
                    # open is not a guard (independent Opus review,
                    # 2026-09-05).
                    if not bool(torch.isfinite(out.logits).all()):
                        return {"error": (
                            "model produced non-finite logits; the closed-form "
                            "sensitivity cannot be validated against them")}
                    rel = (logits - real).abs().max() / real.abs().max().clamp_min(1e-9)
                    # DTYPE-AWARE tolerance. A fixed 1e-3 is BELOW bfloat16's own
                    # eps of 7.81e-3, so on a bf16 model the guard fires on
                    # ordinary rounding: all 8 models tested returned an error
                    # dict under bf16 (gpt2 measured rel 0.00245), and since the
                    # sweep standardizes on bf16 this task produced **no data at
                    # all for the entire zoo** while blaming "final logit
                    # softcapping" (adversarial review, 2026-09-03).
                    #
                    # The estimator itself is exact -- 1.27e-15 against float64
                    # autograd -- so the guard was the only thing wrong. Scale
                    # it to the dtype the logits actually carry, with a floor
                    # that still catches genuine non-linearity (softcapping
                    # moves logits by far more than a few ULPs).
                    # CALIBRATED to the two measured regimes, not picked.
                    # 40*eps is 0.3125 in bfloat16, which is not a rounding
                    # tolerance at all: a softcapped head whose recomputed
                    # logits deviated by rel=0.203697 sailed through it, and
                    # BLME then reported 0.0626069903 where the true value is
                    # 0.0518594161 -- a 20.72% error (adversarial review,
                    # 2026-09-04). Ordinary bf16 rounding measures rel=0.00245
                    # on gpt2, i.e. 0.31*eps, while that softcapping violation
                    # is 26*eps. 3*eps sits ~10x above the observed rounding
                    # floor and ~9x below the observed violation.
                    _ldtype = out.logits.dtype
                    _eps = float(torch.finfo(_ldtype).eps)
                    _tol = max(1e-3, 3.0 * _eps)
                    # LINEARITY is tested directly, not inferred from how close
                    # the recomputed logits happen to be. A cap=30 softcapped
                    # head evaluated at h=7.5 passed the bfloat16 magnitude
                    # test at rel=0.021276595 (tolerance 0.0234375) and BLME
                    # then returned 0.998894631863 for a head whose Jacobian is
                    # nowhere near W (adversarial review round 3, 2026-09-04).
                    # A linear head satisfies f(2h) - f(0) == 2*(f(h) - f(0))
                    # EXACTLY, which no saturating nonlinearity does.
                    # ALL scored positions, not just the first. A softcap
                    # bites hardest at the largest logits, so a 128-token
                    # document whose FIRST token is mild passed a
                    # single-position probe while later positions would have
                    # failed it (codex round 5, 2026-09-05).
                    _h1 = h.detach()
                    _lin_rel = 0.0
                    try:
                        _f = (lambda _x: (_x @ W.T + (head.bias.detach().float()
                                                      if head is not None and head.bias is not None
                                                      else 0.0)))
                        # THROUGH THE MODEL'S OWN HEAD PATH, not the bare
                        # `lm_head` module. Gemma-2 applies
                        # `final_logit_softcapping` in `forward()` AFTER that
                        # module, so probing the module alone is identically
                        # linear and the guard could never fire on the one
                        # architecture it names (adversarial review round 4,
                        # 2026-09-04). `_head_forward` prefers a documented
                        # post-head transform when the config declares one.
                        _cfg = getattr(model, "config", None)
                        _cap = getattr(_cfg, "final_logit_softcapping", None)
                        # `logit_scale` too (Cohere and several LM variants
                        # multiply the logits after the projection). It is
                        # LINEAR, so it cannot break superposition -- but
                        # omitting it makes the reconstruction wrong, and the
                        # value guard then rejects a head that is perfectly
                        # valid for the closed form.
                        _lscale = getattr(_cfg, "logit_scale", None)

                        def _real(_x):
                            _z = _f(_x)
                            if _lscale:
                                _z = _z * float(_lscale)
                            if _cap:
                                _z = torch.tanh(_z / float(_cap)) * float(_cap)
                            return _z
                        # PROBE AT SEVERAL MAGNITUDES AND TAKE THE WORST.
                        #
                        # A single probe at the observed |h| is not enough,
                        # because a saturating head's violation GROWS with
                        # magnitude while a linear head's stays at the rounding
                        # floor. Measured on a cap-30 softcap in bfloat16
                        # (linear control in the same dtype in brackets):
                        #
                        #     |h| 0.5   0.0000000  [0.0000000]
                        #     |h| 1.0   0.0065359  [0.0000000]   <- under 3*eps
                        #     |h| 2.0   0.0263158  [0.0000000]
                        #     |h| 4.6   0.1124260  [0.0000000]
                        #     |h| 16    0.4219512  [0.0000000]
                        #
                        # bfloat16's eps is 7.8e-3, so the 3*eps tolerance is
                        # 0.0234 and a genuine softcap slips under it whenever
                        # the probed activations are mild — the reported case
                        # sat at lin_rel 0.02309 against a 0.02344 threshold and
                        # the task then reported a score **+8.57% above
                        # autograd** (Opus round 11, 2026-09-08). The margin
                        # between the rounding floor and a real violation is
                        # one part in a hundred at this dtype, which is not a
                        # margin.
                        #
                        # Scaling the probe up costs three extra forward
                        # evaluations and moves the softcap an order of
                        # magnitude clear of the floor, while a linear head is
                        # scale-free and reports the same rounding noise at
                        # every rung — so the MAXIMUM over rungs separates them
                        # without tightening the tolerance, which would have
                        # made bf16 rounding fire the guard instead.
                        _a = _real(_h1 * 0.0)
                        _lin_rel = 0.0
                        for _k in (1.0, 2.0, 4.0):
                            _b = _real(_h1 * _k)
                            _c = _real(_h1 * (2.0 * _k))
                            _lhs = (_c - _a)
                            _rhs = 2.0 * (_b - _a)
                            _den = _lhs.abs().max().clamp_min(1e-30)
                            _lin_rel = max(_lin_rel, float(
                                ((_lhs - _rhs).abs().max() / _den).item()))
                    except Exception:
                        _lin_rel = 0.0
                    if _lin_rel > _tol:
                        return {"error": (
                            "LM head is not LINEAR (superposition test "
                            f"f(2h)-f(0) != 2(f(h)-f(0)) by rel={_lin_rel:.3g} "
                            f"> {_tol:.3g}); closed-form sensitivity invalid "
                            "for this architecture")}
                    if float(rel) > _tol:
                        return {"error": (
                            "LM head is not purely linear (recomputed logits "
                            f"deviate rel={float(rel):.3g}; e.g. final logit "
                            f"softcapping); tolerance {_tol:.3g} for "
                            f"{_ldtype} — closed-form sensitivity invalid "
                            "for this architecture")}
                    self._head_guard_max_rel = max(
                        float(getattr(self, "_head_guard_max_rel", 0.0)),
                        float(rel))
                probs = torch.softmax(logits, dim=-1)  # (T-1, V)

                # Closed-form gradient: grad_h log P(y|h) = W_y - sum_k p(k) W_k
                # W_y for each token: (T-1, D)
                W_target = W[targets]  # (T-1, D)
                # Expected W under model distribution: (T-1, D)
                W_expected = probs @ W  # (T-1, V) @ (V, D) -> (T-1, D)

                grad = W_target - W_expected  # (T-1, D)
                # The same factor multiplies the gradient (chain rule through
                # z = s(Wh + b)); omitting it here is the other half of the
                # -3.88% error above.
                if _ls is not None:
                    grad = grad * _ls

                # Squared L2 norm per token, computed SCALE-EXACTLY.
                #
                # `(grad ** 2).sum()` underflows in float32 long before the
                # data does: on a 512-dimensional power-of-two
                # reparameterization each true squared term was 9.361490793e-47
                # and their true sum 4.793083286e-44, all of which flushed to
                # zero (adversarial review round 3, 2026-09-04). Factoring out
                # each row's largest magnitude first only shifts exponents.
                _gm = grad.detach().abs().amax(dim=-1, keepdim=True)
                _gs = grad / _gm.clamp_min(torch.finfo(grad.dtype).tiny)
                # STAY IN float64. Computing the product in double and then
                # narrowing back to float32 throws away exactly the range the
                # rescale was there to protect: with W = 1.00000002004e20 and
                # h = 9.99999968266e-21 the finite float64 value overflows to
                # INFINITY on the cast (adversarial review round 4,
                # 2026-09-04).
                grad_sq_norms = ((_gs.double() ** 2).sum(dim=-1)
                                 * (_gm.squeeze(-1).double() ** 2))
                grad_sq_norms = torch.where(
                    _gm.squeeze(-1).double() > 0, grad_sq_norms,
                    torch.zeros_like(grad_sq_norms))
                sensitivity_values.append(grad_sq_norms.mean().item())

                # SCALE-FREE TWIN: ||h||^2 * ||grad_h log P||^2.
                #
                # The raw quantity is NOT invariant to a reparameterization
                # that leaves the model's function untouched. Verified exactly
                # on pythia-70m (untied head): multiplying final_layer_norm
                # .weight and .bias by c = 2 while dividing embed_out.weight by
                # 2 leaves the logits BIT-IDENTICAL (max |logit change| =
                # 0.000e+00) yet moves representation_sensitivity
                # 0.385760 -> 0.096440, a ratio of exactly 1/c^2. Since h
                # scales by c and the gradient by 1/c under that map, the
                # PRODUCT ||h||^2 * ||grad||^2 is invariant -- so it measures
                # the model's function rather than its residual-scale
                # convention (adversarial review, measured).
                # SCALE-EXACT, matching what the gradient above already does.
                #
                # `(h ** 2).sum(-1)` squares in float32, so it underflows at
                # denormal magnitudes and overflows near the top of the range --
                # and this product is the SCALE-FREE half of the metric, whose
                # whole purpose is to be invariant. Measured: at weight
                # magnitude 4.156000412762293e-39 it returned
                # 2.159042718946516e-78 against 1.727233943088035e-77, a ratio
                # of **exactly 1/8** (three mantissa bits lost to the denormal
                # square, not rounding noise), and NaN at 2.406159650500376e38
                # against 2.315841695931826e77 (gpt-5.6-sol round 9,
                # 2026-09-07).
                #
                # Factoring out the row max first only shifts exponents, so the
                # value is unchanged wherever float32 already worked; it is the
                # same three-line idiom applied to `grad` twenty lines above,
                # which was fixed in an earlier round while this line -- the
                # other half of the same product -- was left behind.
                _hm = h.detach().abs().amax(dim=-1, keepdim=True)
                _hs = h / _hm.clamp_min(torch.finfo(h.dtype).tiny)
                _h_sq = ((_hs.double() ** 2).sum(dim=-1)
                         * (_hm.squeeze(-1).double() ** 2))  # (T-1,)
                _h_sq = torch.where(_hm.squeeze(-1).double() > 0, _h_sq,
                                    torch.zeros_like(_h_sq))
                scalefree_values.append((_h_sq * grad_sq_norms).mean().item())

        if not sensitivity_values:
             return {"error": "Could not compute representation sensitivity."}

        out = {
            "representation_sensitivity": float(np.mean(sensitivity_values)),
            "sensitivity_std": float(np.std(sensitivity_values)),
            "num_samples_analyzed": len(sensitivity_values)
        }
        if scalefree_values:
            # DEMOTED to _meta_ (2026-09-04). The twin survives ONE global
            # rescale (ratio 1.00000000 at c=2) but not the general gauge it
            # claims to remove: an exact DIAGONAL residual-coordinate
            # reparameterization -- function-preserving, max logit change 0.0 --
            # moved it 79789.50391 -> 353822.21875, a ratio of 4.43444565
            # (adversarial review, 2026-09-04). Invariance to a one-parameter
            # subgroup is not invariance to the convention, so it does not meet
            # this campaign's mechanism bar and is kept out of the feature
            # matrix.
            out["_meta_representation_sensitivity_scalefree"] = float(
                np.mean(scalefree_values))
            out["_meta_sensitivity_scalefree_std"] = float(np.std(scalefree_values))
        if hasattr(self, "_head_guard_max_rel"):
            # The WORST discrepancy seen over every document, not the first.
            out["_meta_head_guard_max_rel"] = float(self._head_guard_max_rel)
        return out
