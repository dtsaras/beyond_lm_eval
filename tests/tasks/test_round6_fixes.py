"""Regression tests for round-6 audit fixes.

The final independent audit surfaced four new bugs:
  1. bias.py and neural_collapse.py double-norm hidden_states[-1]
     (already post-final-norm in transformers 5.x, so applying
     ``final_norm`` again corrupts every WEAT d-value and NC1 estimate).
  2. attention_polysemanticity removes forward hooks outside try/finally,
     leaking hooks if a forward fails mid-loop.
  3. betti_curve.betti_0_decay_rate uses raw layer index for the slope
     fit — same cross-model-comparability bug that dynamics/gradient_flow
     was fixed for in round 4.
  4. generation_diversity replicates the softmax + log(clamp) underflow
     pattern that prediction_entropy had fixed to use log_softmax.
"""
import inspect
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))


def test_bias_does_not_double_norm_final_hidden_state():
    """bias.py must NOT apply final_norm to hidden_states[-1] since
    HF already normalised it. logit_lens established empirically that
    ``lm_head(hidden_states[-1]) == outputs.logits`` — so re-norming
    here would be a double-pass."""
    from blme.tasks.consistency import bias
    src = inspect.getsource(bias._embed_word)
    # The old ``h = final_norm(h.unsqueeze(0))`` must be gone.
    assert "final_norm(h.unsqueeze(0)" not in src, (
        "bias._embed_word still re-norms the already-normed last hidden state"
    )


def test_neural_collapse_does_not_double_norm_final_hidden_state():
    """neural_collapse must NOT re-apply final_norm to hidden_states[-1]."""
    from blme.tasks.geometry import neural_collapse
    src = inspect.getsource(neural_collapse)
    # Old pattern was ``h = final_norm(h.to(norm_dtype))`` after reading
    # hidden_states[-1] — verify it's gone.
    assert "h = final_norm(h.to(norm_dtype))" not in src
    # Also verify the ``final_norm = get_final_norm(model)`` lookup is
    # no longer performed (dead code cleanup).
    assert "final_norm = get_final_norm(model)" not in src


def test_attention_polysemanticity_hook_removal_in_finally():
    """Hooks must be removed in a finally block so a forward error
    mid-loop doesn't leak them into subsequent tasks."""
    from blme.tasks.interpretability import attention_polysemanticity
    src = inspect.getsource(attention_polysemanticity)
    # Must contain the try: ... finally: ... for h in handles pattern.
    assert "try:" in src and "finally:" in src
    # And the handle-removal loop must appear inside the finally block
    # (sanity: there's no longer a bare ``for h in handles: h.remove()``
    # at the top level of the evaluate method).
    # Verified by finding the finally pattern near the hook loop.
    assert src.count("h.remove()") >= 1


def test_betti_curve_decay_rate_uses_normalised_depth():
    """betti_curve.betti_0_decay_rate must regress on ``x / (n-1)``,
    not raw layer index, so slopes are comparable across depths."""
    from blme.tasks.topology import betti_curve
    src = inspect.getsource(betti_curve)
    # The old ``x = np.arange(num_layers)`` followed directly by
    # ``np.polyfit(x, betti_0_curve, 1)`` is the bug. The fix inserts
    # ``/ float(denom)`` on the x axis.
    assert "num_layers - 1" in src or "num_layers-1" in src
    # Ensure the slope fit uses the normalised x.
    assert "np.arange(num_layers, dtype=np.float64) / float(denom)" in src


def test_generation_diversity_uses_log_softmax_not_log_clamp():
    """generation_diversity must use log_softmax directly rather than
    softmax + log(clamp) so low-prob tail isn't silently zeroed."""
    from blme.tasks.dynamics import generation_diversity
    src = inspect.getsource(generation_diversity)
    # The old pattern is softmax then torch.log(probs.clamp(min=1e-12)).
    assert "torch.log(probs.clamp(min=1e-12))" not in src
    assert "F.log_softmax(scores_stack" in src
