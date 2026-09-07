"""The graceful-degradation path of `geometry_unembedding` must actually run.

`unembedding.py` wraps its two SVDs in a `try/except` written specifically for
"GPU OOM on qwen-9b/27b", with a comment recording that returning 0.0 there was
a historic bug. A later change added `unembedding_eff_rank_centered` to the
return dict but bound the variable ONLY inside the `try`, so the handler raised
`UnboundLocalError` on the way out and `core.py` recorded the ENTIRE task as
failed -- losing tying, purity, category count and embedding alignment along
with the one number that genuinely could not be computed.

The task is `verified_d`, so it is absent from all 66 archived envelopes: this
never fired in the archive, and would have fired first on exactly the large
models the comment names. This test drives the failure path directly
(red-team review, 2026-09-05).
"""
import pytest
import torch

from blme.tasks.geometry.unembedding import UnembeddingDiagnosticsTask


@pytest.mark.parametrize("exc", [RuntimeError("CUDA out of memory"),
                                 torch.linalg.LinAlgError("svd did not converge")])
def test_task_survives_an_svd_failure(monkeypatch, mock_model_tokenizer, exc):
    model, tokenizer = mock_model_tokenizer
    task = UnembeddingDiagnosticsTask({})

    def _boom(*a, **k):
        raise exc

    monkeypatch.setattr(torch.linalg, "svdvals", _boom)
    out = task.evaluate(model, tokenizer, dataset=None)

    assert "error" not in out, f"the whole task was lost: {out.get('error')!r}"
    # The two SVD-derived columns degrade to NaN...
    import math
    assert math.isnan(out["unembedding_eff_rank"])
    assert math.isnan(out["unembedding_eff_rank_centered"])
    # ...and everything that does NOT depend on the SVD still arrives.
    assert "unembedding_is_tied" in out
    assert "unembedding_purity_mean" in out
