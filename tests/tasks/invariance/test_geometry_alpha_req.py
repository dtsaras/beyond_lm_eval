"""Direct mechanism tests for geometry_alpha_req, through the real evaluate()
(tests/invariance driver)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import invariance as inv  # noqa: E402

TASK = "geometry_alpha_req"
KEYS = ["alpha_req", "rankme"]


@pytest.fixture(scope="module")
def states():
    """200 documents of a power-law cloud, d = 256 >= N, so every transform
    below keeps the number of eigenvalues at N."""
    rng = np.random.default_rng(0)
    d = 256
    scale = np.arange(1, d + 1) ** -0.7
    lengths = rng.integers(2, 6, size=200)
    return {l: [torch.tensor(rng.standard_normal((int(t), d)) * scale, dtype=torch.float32)
                for t in lengths] for l in range(3)}


def test_headline_invariant(states):
    """Zero-padding keeps every eigenvalue (rounding only); a power-of-two
    rescale scales them exactly; an isometric embedding agrees to the cache's
    float32 rounding; identity blocks leave the output state bit-identical."""
    base = inv.run_task_on_states(TASK, states)
    assert all(np.isfinite(base[k]) for k in KEYS)
    inv.assert_invariant(base, inv.run_task_on_states(TASK, inv.zero_pad_docs(states, 2)),
                         KEYS, rtol=1e-10)
    inv.assert_invariant(base, inv.run_task_on_states(TASK, inv.rescale_docs(states, 2.0)),
                         KEYS, rtol=0.0)
    inv.assert_invariant(base, inv.run_task_on_states(TASK, inv.embed_isometric_docs(states, 2, 3)),
                         KEYS, rtol=1e-5)
    model, tok, texts = _gpt2_and_texts()
    base_m = inv.run_task_on_model(TASK, model, tok, texts)
    ins = inv.run_task_on_model(TASK, inv.insert_identity_blocks(model, [0, 5]), tok, texts)
    assert np.isfinite(base_m["alpha_req"])
    inv.assert_invariant(base_m, ins, KEYS, rtol=0.0)


def _gpt2_and_texts():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"gpt2 not cached: {e}")
    from blme.cache import load_default_corpus
    texts = [d["text"] for d in load_default_corpus(128)]
    return model, tok, texts


def test_rank_window_needs_100_eigenvalues():
    """Fewer than 100 eigenvalues: alpha is NaN with a reason, RankMe is finite."""
    rng = np.random.default_rng(2)
    docs = {0: [torch.tensor(rng.standard_normal((3, 64)), dtype=torch.float32)
                for _ in range(80)]}
    out = inv.run_task_on_states(TASK, docs)
    assert np.isnan(out["alpha_req"]) and np.isfinite(out["rankme"])
    assert "ranks 11-100" in out["_meta_alpha_req_skipped_reason"]
