"""Direct mechanism tests for geometry_correlation_dimension through the real
evaluate() on injected states. The task reads one point per document from the
shared cache when it holds every requested document, so the stub cache drives
it (``force=True``: it is listed as self-forwarding in the shared driver; the
driver still raises if a forward pass is attempted)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import invariance as inv  # noqa: E402

TASK = "geometry_correlation_dimension"
N_DOCS = 64


def _run(S):
    return inv.run_task_on_states(TASK, S, force=True)


@pytest.fixture(scope="module")
def tiny():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-gpt2", dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"tiny-random-gpt2 not cached: {e}")
    return model, tok


@pytest.fixture(scope="module")
def texts():
    from blme.cache import load_default_corpus
    try:
        return [d["text"] for d in load_default_corpus(N_DOCS)]
    except (RuntimeError, ValueError) as e:  # pragma: no cover - environment dependent
        pytest.skip(f"default corpus unavailable: {e}")


@pytest.fixture(scope="module")
def states(tiny, texts):
    return inv.states_from_model(*tiny, texts)


def test_headline_invariant(tiny, texts, states):
    """D2 from pair distances over a window fixed by N: zero-padding and a
    power-of-two rescale are bit-identical, an isometric embedding agrees to
    the cache's float32 rounding, and identity blocks after blocks 0 and 1
    (exact) leave the output representation and the value unchanged."""
    model, tok = tiny
    base = _run(states)
    assert np.isfinite(base["correlation_dimension"])
    assert base["_meta_n_points"] == N_DOCS
    inv.assert_invariant(base, _run(inv.zero_pad_docs(states, 2)), ["correlation_dimension"], rtol=0.0)
    inv.assert_invariant(base, _run(inv.rescale_docs(states, 2.0)), ["correlation_dimension"],
                         rtol=1e-12)
    inv.assert_invariant(base, _run(inv.embed_isometric_docs(states, 2, 3)),
                         ["correlation_dimension"], rtol=1e-4)
    ins = _run(inv.states_from_model(inv.insert_identity_blocks(model, [0, 1]), tok, texts))
    inv.assert_invariant(base, ins, ["correlation_dimension"], rtol=0.0)


def test_point_count_is_pinned_not_invariant(states):
    """Known negative: at fixed geometry D2 moves with N (the window and the
    resolution both depend on it), which is why N is pinned for every model."""
    base = _run(states)
    half = _run(states.select_docs(range(N_DOCS // 2 + 8)))
    inv.assert_moves(base, half, "correlation_dimension", min_rel=1e-3)
