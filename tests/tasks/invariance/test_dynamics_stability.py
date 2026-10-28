"""dynamics_stability: emittable vocabulary rows only."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamics_fixtures import inv, tiny_gpt2, tokenizer  # noqa: E402

from blme.tasks.dynamics.stability import NeighborhoodStabilityTask  # noqa: E402

CFG = {"k": 10, "num_samples": 200, "seed": 0}


def _run(model, tok, cfg=CFG):
    with inv.limit_threads(1):
        out = NeighborhoodStabilityTask(config=dict(cfg)).evaluate(model, tok, None)
    assert "error" not in out, out
    return out


@pytest.mark.parametrize("kind", ["zero", "small_random"])
def test_rows_past_the_tokenizer_do_not_enter(kind):
    """Embedding rows past len(tokenizer) are never emitted. Padding the
    vocabulary with such rows leaves the value bit-identical; pre-fix they were
    drawn as queries (zero rows are trivially stable) and as neighbours."""
    tok = tokenizer()
    model = tiny_gpt2()
    padded = inv.pad_vocabulary(model, 400, kind=kind)
    base, other = _run(model, tok), _run(padded, tok)
    inv.assert_invariant(base, other, ["stability_mean", "stability_std"], rtol=0.0)
    assert other["_meta_n_rows_excluded"] == 400
    assert other["_meta_n_rows_emittable"] == len(tok)


def test_all_rows_used_when_the_tokenizer_has_no_length():
    class _NoLen:
        pass
    out = _run(tiny_gpt2(), _NoLen())
    assert out["_meta_n_rows_excluded"] == 0
