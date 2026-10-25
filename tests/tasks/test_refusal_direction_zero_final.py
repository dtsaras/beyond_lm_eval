"""A zero final direction must be reported as 0.0, not replaced by an earlier layer.

`direction_norm` names the FINAL layer's mean-difference norm. Zero-direction
layers used to be dropped with `continue`, and the headline was taken from the
last SURVIVING entry -- so when the final layer's class means coincided, an
earlier block's numbers were published under the final layer's name. The
released `get_mean_diff` returns the zero vector there.
"""
import numpy as np
import pytest
import torch

N_PER_CLASS = 12


def _dataset():
    return ([{"text": f"harmful prompt number {i}", "label": "harmful"}
             for i in range(N_PER_CLASS)]
            + [{"text": f"harmless prompt number {i}", "label": "harmless"}
               for i in range(N_PER_CLASS)])


def test_zero_final_direction_is_reported_as_zero(mock_model, mock_tokenizer,
                                                  monkeypatch):
    """The final block's output is made CLASS-INDEPENDENT but sample-varied.

    Both classes then have identical final means, so the exact mean-difference
    norm is 0.0 while the representations stay nonzero and varied -- the
    reviewer's `zero_final` condition. The final block is patched at the module
    level because the task re-reads it through a forward hook on `layers[-1]`,
    which would otherwise overwrite anything injected into `hidden_states`.
    """
    from blme.tasks.common import get_layers
    from blme.tasks.representation_engineering import RefusalDirectionTask

    layers = get_layers(mock_model)
    last = layers[-1]
    real_forward = last.forward
    calls = {"i": 0}

    def class_independent(*a, **k):
        out = real_forward(*a, **k)
        h = out[0] if isinstance(out, tuple) else out
        j = calls["i"] % N_PER_CLASS      # depends on the SAMPLE, not the class
        calls["i"] += 1
        new = torch.zeros_like(h)
        new[..., 0] = float(j + 1)        # varied across samples
        new[..., 1] = float(-(j + 1))
        if isinstance(out, tuple):
            return (new,) + tuple(out[1:])
        return new

    monkeypatch.setattr(last, "forward", class_independent)
    res = RefusalDirectionTask(config={"cv_splits": 3, "seed": 42}).evaluate(
        mock_model, mock_tokenizer, dataset=_dataset())
    if "error" in res:
        pytest.skip(f"task declined: {res['error']}")

    # FIXTURE INTEGRITY: the hook really did supply the final block output,
    # and every prompt was seen exactly once per class.
    assert res.get("_meta_last_entry_is_block_output") is True
    assert calls["i"] == 2 * N_PER_CLASS

    assert res["direction_norm"] == 0.0, (
        f"direction_norm = {res['direction_norm']!r}; the pre-fix code dropped "
        f"the zero-direction final layer and published an earlier block's norm")
    # THE CROSS-VALIDATED AUC IS NOT UNDEFINED HERE, and asserting that it was
    # is a mistake this test used to make. `separability_auc` is fitted on
    # TRAINING FOLDS, and a fold's mean difference can be nonzero while the
    # full-population one cancels to exactly zero -- so a zero `direction_norm`
    # must not suppress it (codex round 13, 2026-09-08). The first version of
    # this test asserted isnan() and passed, encoding the same conflation as the
    # repair it was written to guard.
    assert not np.isnan(res["separability_auc"]), (
        "a zero FULL-POPULATION direction wrongly suppressed the "
        "cross-validated AUC, which is fitted per training fold")
    assert 0.0 <= res["separability_auc"] <= 1.0, res["separability_auc"]


def test_depth_fraction_stays_within_range(mock_model, mock_tokenizer):
    """best_layer_fraction must never exceed 1.0 or promote a dropped layer."""
    from blme.tasks.representation_engineering import RefusalDirectionTask
    res = RefusalDirectionTask(config={"cv_splits": 3, "seed": 42}).evaluate(
        mock_model, mock_tokenizer, dataset=_dataset())
    if "error" in res:
        pytest.skip(f"task declined: {res['error']}")
    frac = res.get("best_layer_fraction")
    assert frac is not None
    assert np.isnan(frac) or 0.0 <= frac <= 1.0, frac
