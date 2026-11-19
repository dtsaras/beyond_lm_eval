"""
Representation Sensitivity Task
──────────────────────────────────────────────────────────────────────
Mean squared norm of the gradient of the true-next-token log-probability with
respect to the final hidden state the unembedding reads,
E_tokens ||grad_h log P(y|h)||^2.

The LM head is linear (logits = s (W h + b)), so the gradient has the closed
form grad_h log P(y|h) = s (W_y - sum_k p(k) W_k) (the softmax score
identity), which avoids per-token backward passes. For a final tanh softcap,
each vocabulary row also receives its own derivative before the expectation.

References:
    "Natural Gradient Works Efficiently in Learning" (Amari, 1998,
    Neural Computation 10(2):251-276) — the softmax score-function
"""

import torch
import numpy as np

from .._tokens import strip_trailing_special
from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_final_norm, get_lm_head, get_embeddings
import logging
logger = logging.getLogger("blme")


def _sq_rownorm_f64(M):
    """Row-wise squared L2 norms in float64, exact at any representable
    magnitude (the row max is factored out before squaring)."""
    m = M.detach().abs().amax(dim=-1, keepdim=True)
    s = M / m.clamp_min(torch.finfo(M.dtype).tiny)
    sq = (s.double() ** 2).sum(dim=-1) * (m.squeeze(-1).double() ** 2)
    return torch.where(m.squeeze(-1).double() > 0, sq, torch.zeros_like(sq))


class _HeadNotLinear(RuntimeError):
    """The closed form does not apply (non-linear head or non-finite logits)."""


class _HeadGuardUnavailable(RuntimeError):
    """The linearity guard could not be evaluated, so the head is not certified."""


class _UnitGainNorm:
    """The final norm with its gain set to 1 and its bias to 0, plus the
    effective gain and bias the real norm applies: h = gamma * n + beta.

    The parametrisation is read off the module itself: with weight = 0 and
    bias = 0 a standard norm returns 0 and a (1 + w) norm (Gemma) returns its
    unit-gain output, so gamma is w or 1 + w respectively.
    """

    def __init__(self, norm):
        import copy
        self.norm = norm
        self.probe = copy.deepcopy(norm).float()
        w = getattr(self.probe, "weight", None)
        self.ok = isinstance(w, torch.Tensor)
        self.form = "unavailable"
        if not self.ok:
            # A norm without affine parameters (OLMo-1B's non-parametric
            # LayerNorm) already outputs the unit-gain state: gamma = 1, beta = 0.
            shape = getattr(norm, "normalized_shape", None)
            if shape:
                d = int(np.prod(tuple(shape)))
                self.ok, self.form = True, "no_affine"
                self.w = self.gamma = torch.ones(d)
                self.beta = torch.zeros(d)
            return
        self.w = norm.weight.detach().float()
        b = getattr(norm, "bias", None)
        self.beta = b.detach().float() if isinstance(b, torch.Tensor) else torch.zeros_like(self.w)
        with torch.no_grad():
            self.probe.weight.zero_()
            if isinstance(getattr(self.probe, "bias", None), torch.Tensor):
                self.probe.bias.zero_()
            x = torch.randn(2, self.w.numel(), generator=torch.Generator().manual_seed(0))
            zero_gain = bool((self.probe(x.to(self.probe.weight.device)) == 0).all())
            if zero_gain:
                self.probe.weight.fill_(1.0)
                self.form, self.gamma = "weight", self.w
            else:
                self.form, self.gamma = "one_plus_weight", 1.0 + self.w

    def unit(self, x):
        with torch.no_grad():
            w = getattr(self.probe, "weight", None)
            dev = w.device if isinstance(w, torch.Tensor) else x.device
            return self.probe(x.detach().float().to(dev))


@register_task("geometry_representation_sensitivity")
class RepresentationSensitivityTask(DiagnosticTask):
    """
    Computes ||grad_h log P(y|h)||^2 with respect to final-layer hidden
    states using a closed-form derivation for the linear LM head.
    Uses the true next token as the target rather than argmax.
    """

    TOKEN_POLICY = "not_token_based"
    SOURCES = {
        "paper": "DOI:10.1162/089976698300017746",
        "code": None,
        "llm_precedent": "none found; the softmax score identity is elementary (Amari 1998)",
    }
    ADAPTATION_NOTE = (
        "Representation sensitivity is a BLME construct: the mean squared norm of the "
        "gradient of the true-next-token log-probability with respect to the "
        "representation the unembedding reads, in closed form W_y minus the expected "
        "unembedding row (the softmax score identity), over every prediction position "
        "like perplexity. The post-norm state h makes the raw value depend on how a "
        "model splits scale between the final-norm gain and the unembedding, a free "
        "gauge, so S_unit takes the gradient with respect to the unit-gain normalised "
        "state n (gamma times n plus beta is h), which absorbs the gain into the "
        "unembedding and is exactly invariant to that gauge and to a residual rescale. "
        "S_unit is therefore the headline, and the raw value is reported only as "
        "_meta_representation_sensitivity_raw. "
        "Because ||n||^2 = d for a normalised state, S_unit equals d E||gamma * g||^2, with "
        "g the gradient with respect to h, so it carries a factor of the width. "
        "The ||h||^2 ||grad_h||^2 twin was rejected because it fails the diagonal "
        "gauge, and Fisher-trace or Hessian variants need second-order work over V. "
        "For a declared final tanh softcap, the gradient includes the derivative of "
        "each vocabulary logit. The reconstructed scores are checked against the "
        "model's actual logits on every document."
    )
    HEADLINE_KEYS = ("S_unit",)
    SIZE_INVARIANT = {
        "S_unit": (("final_norm_unembedding_gauge", "residual_scale_gauge"),
                   "tests/tasks/invariance/test_geometry_representation_sensitivity.py::"
                   "test_s_unit_invariant_under_final_norm_unembedding_gauge"),
    }
    @staticmethod
    def _logit_scale_for(model):
        """`config.logit_scale` as a float, or None when the head is unscaled
        (exactly 1.0 is None, so the ordinary path is unchanged)."""
        cfg = getattr(model, "config", None)
        val = getattr(cfg, "logit_scale", None) if cfg is not None else None
        try:
            f = float(val) if val is not None else None
        except (TypeError, ValueError):
            return None
        if f is None or f == 1.0:
            return None
        return f

    @staticmethod
    def _softcap_for(model):
        cfg = getattr(model, "config", None)
        cap = getattr(cfg, "final_logit_softcapping", None)
        if cap is None and callable(getattr(cfg, "get_text_config", None)):
            cap = getattr(cfg.get_text_config(), "final_logit_softcapping", None)
        if cap is None:
            return None
        try:
            cap = float(cap)
            if not np.isfinite(cap) or cap <= 0:
                raise ValueError("final_logit_softcapping must be finite and positive")
        except (TypeError, ValueError) as exc:
            raise _HeadGuardUnavailable(f"linearity test could not run ({type(exc).__name__}: {exc})") from exc
        return cap

    @staticmethod
    def _head_linearity(model, head, W, h):
        """Worst superposition violation |f(2kh)-f(0) - 2(f(kh)-f(0))| / |.| over
        k in {1, 2, 4} for the affine projection including logit_scale. The
        separately differentiated final softcap is checked against the actual
        model logits in the document loop.

        Returns ``(lin_rel, reason)``. A test that cannot run (an unparseable
        transform in the config, a tensor shape/device error or OOM) or gives a
        non-finite value returns ``(nan, reason)``: the guard fails closed."""
        _h1 = h.detach()
        _bias = getattr(head, "bias", None) if head is not None else None
        try:
            _f = (lambda _x: (_x @ W.T + (_bias.detach().float()
                                          if _bias is not None else 0.0)))
            _cfg = getattr(model, "config", None)
            _lscale = getattr(_cfg, "logit_scale", None)

            def _real(_x):
                _z = _f(_x)
                if _lscale:
                    _z = _z * float(_lscale)
                return _z
            _a = _real(_h1 * 0.0)
            lin_rel = 0.0
            for _k in (1.0, 2.0, 4.0):
                _b = _real(_h1 * _k)
                _c = _real(_h1 * (2.0 * _k))
                _lhs = (_c - _a)
                _rhs = 2.0 * (_b - _a)
                _num = float((_lhs - _rhs).abs().max().item())
                _den = float(_lhs.abs().max().item())
                # A head constant along this ray is linear on it (0 discrepancy);
                # any discrepancy with no step to compare it to is non-linear.
                lin_rel = max(lin_rel, 0.0 if _num == 0.0 else
                              (_num / _den if _den > 0.0 else float("inf")))
        except (TypeError, ValueError, RuntimeError) as e:
            return float("nan"), f"linearity test could not run ({type(e).__name__}: {e})"
        if not np.isfinite(lin_rel):
            return float("nan"), f"linearity test gave a non-finite discrepancy ({lin_rel})"
        return lin_rel, None

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Representation Sensitivity Analysis...")
        num_samples = self.config.get("num_samples", 20)

        # Reset per run: the worst guard discrepancy belongs to this evaluation.
        self._head_guard_max_rel = 0.0
        if dataset is None:
             from ...cache import load_default_corpus
             dataset = load_default_corpus(num_samples)

        samples = list(dataset)[:num_samples]
        if not samples:
             return {"error": "Need at least 1 sample."}

        device = next(model.parameters()).device

        # Unembedding matrix W (vocab_size, hidden_dim).
        head = get_lm_head(model)
        if head is not None:
            W = head.weight.detach().float()
        else:
            W = get_embeddings(model)
            if W is None:
                return {"error": "Cannot access unembedding matrix."}
            W = W.detach().float()
        W = W.to(device)

        sensitivity_values = []
        scalefree_values = []
        # S_unit: the gradient with respect to the unit-gain normalised state
        # n = (h - beta) / gamma, i.e. ||n||^2 ||gamma (W_y - p W)||^2. W diag(gamma)
        # is invariant to every gain/unembedding gauge and n to the residual
        # scale (tests/tasks/invariance/test_geometry_representation_sensitivity.py).
        unit_values, unit_err = [], 0.0
        unit_reason = None
        fnorm = get_final_norm(model)
        ug = _UnitGainNorm(fnorm) if fnorm is not None else None
        if ug is None or not ug.ok:
            unit_reason = "no final norm with a gain parameter was found"
        captured = {}
        hook = None
        if unit_reason is None:
            hook = fnorm.register_forward_hook(
                lambda m, i, o: captured.__setitem__("x", i[0] if isinstance(i, tuple) else i))

        model.eval()
        try:
            cap = self._softcap_for(model)
            self._loop(model, tokenizer, samples, device, head, W, sensitivity_values,
                       scalefree_values, unit_values, captured, ug if unit_reason is None else None, cap)
        except _HeadNotLinear as e:
            return {"error": str(e)}
        except _HeadGuardUnavailable as e:
            # Fail closed: an uncertified head gives NaN, never the closed form
            # (tests/tasks/test_representation_sensitivity_guard.py).
            return {"S_unit": float("nan"),
                    "_meta_representation_sensitivity_raw": float("nan"),
                    "_meta_representation_sensitivity_raw_std": float("nan"),
                    "_meta_head_guard_reason": str(e)}
        finally:
            if hook is not None:
                hook.remove()
        unit_err = max([e for _, e in unit_values], default=0.0)

        if not sensitivity_values:
             return {"error": "Could not compute representation sensitivity."}

        out = {
            # Gradient with respect to h: depends on the final-norm gain versus
            # unembedding gauge, so _meta_ only (tests/tasks/invariance/
            # test_geometry_representation_sensitivity.py::test_raw_value_is_meta_only).
            "_meta_representation_sensitivity_raw": float(np.mean(sensitivity_values)),
            "_meta_representation_sensitivity_raw_std": float(np.std(sensitivity_values)),
            # The number of documents actually scored (a config echo otherwise).
            "_meta_num_samples_analyzed": len(sensitivity_values),
        }
        if cap is not None:
            out["_meta_final_logit_softcapping"] = cap
        if unit_reason is None and unit_values:
            tol = 1e-3 if all(p.dtype == torch.float32 for p in model.parameters()) else 3e-2
            if unit_err > tol:
                unit_reason = (f"final norm output not reproduced as gamma * n + beta "
                               f"(max rel {unit_err:.3g} > {tol:g})")
        if unit_reason is None and unit_values:
            out["S_unit"] = float(np.mean([v for v, _ in unit_values]))
            out["_meta_S_unit_std"] = float(np.std([v for v, _ in unit_values]))
        else:
            out["S_unit"] = float("nan")
            out["_meta_S_unit_reason"] = unit_reason or "no document scored"
        out["_meta_final_norm_gain_form"] = ug.form if ug is not None else "unavailable"
        out["_meta_S_unit_recon_rel_err"] = float(unit_err)
        if scalefree_values:
            out["_meta_representation_sensitivity_scalefree"] = float(
                np.mean(scalefree_values))
            out["_meta_sensitivity_scalefree_std"] = float(np.std(scalefree_values))
        if hasattr(self, "_head_guard_max_rel"):
            # The worst discrepancy over every document.
            out["_meta_head_guard_max_rel"] = float(self._head_guard_max_rel)
        return out

    def _loop(self, model, tokenizer, samples, device, head, W, sensitivity_values,
              scalefree_values, unit_values, captured, ug, cap):
        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                inputs, _ = strip_trailing_special(inputs, tokenizer)
                input_ids = inputs["input_ids"]
                if input_ids.shape[1] < 2:
                    continue

                out = model(**inputs, output_hidden_states=True)

                # post-final-norm: DELIBERATELY post-norm -- the vector the LM
                # head consumes (lm_head(hidden_states[-1]) is the logits).
                final_hidden = out.hidden_states[-1][0].float()  # (T, D)
                targets = input_ids[0, 1:]  # (T-1,)
                h = final_hidden[:-1]       # (T-1, D), predicting the next token

                logits = h @ W.T
                if head is not None and head.bias is not None:
                    logits = logits + head.bias.detach().float()
                # z = s (W h + b): the scale enters the probabilities here and
                # the gradient below (Cohere-style heads).
                _ls = self._logit_scale_for(model)
                if _ls is not None:
                    logits = logits * _ls
                if cap is not None:
                    scaled = logits / cap
                    logits = cap * torch.tanh(scaled)
                    # sech²(z/c) without cancellation when tanh rounds to one.
                    tail = torch.exp(-2 * scaled.abs())
                    slope = 4 * tail / (1 + tail).square()

                # Check the complete assumed logit transform on every document.
                real = out.logits[0, :-1, :].float()
                # Fail closed on non-finite logits (a NaN `rel` passes `rel > tol`).
                if not bool(torch.isfinite(out.logits).all()):
                    raise _HeadNotLinear(
                        "model produced non-finite logits; the closed-form "
                        "sensitivity cannot be validated against them")
                rel = (logits - real).abs().max() / real.abs().max().clamp_min(1e-9)
                # 3 eps of the logits' dtype: ~10x above bf16 rounding (0.31 eps
                # on gpt2), ~9x below a measured softcap violation (26 eps).
                _ldtype = out.logits.dtype
                _eps = float(torch.finfo(_ldtype).eps)
                _tol = max(1e-3, 3.0 * _eps)
                _lin_rel, _lin_why = self._head_linearity(model, head, W, h)
                if _lin_why is not None:
                    raise _HeadGuardUnavailable(_lin_why)
                if _lin_rel > _tol:
                    raise _HeadNotLinear(
                        "LM head is not LINEAR (superposition test "
                        f"f(2h)-f(0) != 2(f(h)-f(0)) by rel={_lin_rel:.3g} "
                        f"> {_tol:.3g}); closed-form sensitivity invalid "
                        "for this architecture")
                if float(rel) > _tol:
                    raise _HeadNotLinear(
                        "LM head does not match the configured projection "
                        f"(recomputed logits deviate rel={float(rel):.3g}); tolerance {_tol:.3g} for "
                        f"{_ldtype} — closed-form sensitivity invalid "
                        "for this architecture")
                self._head_guard_max_rel = max(
                    float(getattr(self, "_head_guard_max_rel", 0.0)),
                    float(rel))

                probs = torch.softmax(logits, dim=-1)  # (T-1, V)
                # The softmax score is (one_hot(y) - p) times the logit Jacobian.
                W_target = W[targets]
                if cap is not None:
                    W_target = W_target * slope.gather(1, targets[:, None])
                    probs = probs * slope
                W_expected = probs @ W
                grad = W_target - W_expected
                if _ls is not None:
                    grad = grad * _ls

                # Squared norms in float64 with the row max factored out: a
                # float32 square underflows/overflows at extreme magnitudes.
                grad_sq_norms = _sq_rownorm_f64(grad)
                sensitivity_values.append(grad_sq_norms.mean().item())

                # ||h||^2 ||grad||^2: invariant to a global final-norm/head
                # rescale but not to a diagonal one, hence _meta_ only.
                _h_sq = _sq_rownorm_f64(h)
                scalefree_values.append((_h_sq * grad_sq_norms).mean().item())

                if ug is not None and "x" in captured:
                    n = ug.unit(captured["x"][0])[:-1].to(h.device)     # (T-1, D)
                    g = ug.gamma.to(h.device)
                    recon = g * n + ug.beta.to(h.device)
                    err = float((recon - h).abs().max() / h.abs().max().clamp_min(
                        torch.finfo(h.dtype).tiny))
                    s_unit = (_sq_rownorm_f64(n) * _sq_rownorm_f64(grad * g)).mean().item()
                    unit_values.append((s_unit, err))
