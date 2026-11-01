"""Direct mechanism tests for geometry_gride and geometry_information_imbalance,
through the real evaluate() (tests/invariance driver)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import invariance as inv  # noqa: E402

GRIDE_KEYS = ["gride_id_peak", "gride_id_last_layer"]
II_KEYS = ["ii_first_to_last", "ii_last_to_first", "ii_asymmetry_first_last"]
NO_KEYS = ["nn_overlap_first_last", "nn_overlap_adjacent_min"]


@pytest.fixture(scope="module")
def states():
    rng = np.random.default_rng(0)
    d, n = 40, 240
    lengths = rng.integers(2, 5, size=n)
    base = [rng.standard_normal((int(t), d)) for t in lengths]
    mats = [np.linalg.qr(rng.standard_normal((d, d)))[0] for _ in range(4)]
    return {l: [torch.tensor(np.tanh(x @ mats[l]) * (1 + l), dtype=torch.float32) for x in base]
            for l in range(4)}


@pytest.mark.parametrize("task,keys", [("geometry_gride", GRIDE_KEYS),
                                       ("geometry_information_imbalance", II_KEYS),
                                       ("geometry_neighborhood_overlap", NO_KEYS)],
                         ids=["geometry_gride-gride_keys", "geometry_information_imbalance-ii_keys",
                              "geometry_neighborhood_overlap-no_keys"])
def test_headline_invariant(states, task, keys):
    """Zero-padding and a power-of-two rescale leave the neighbour distances'
    order and ratios unchanged; an isometric embedding agrees to the cache's
    float32 rounding; identity blocks after blocks 0 and 1 of tiny-random-gpt2
    duplicate existing layers, so the endpoints and the peak are unchanged."""
    base = inv.run_task_on_states(task, states)
    assert all(np.isfinite(base[k]) for k in keys)
    inv.assert_invariant(base, inv.run_task_on_states(task, inv.zero_pad_docs(states, 2)),
                         keys, rtol=1e-6)
    inv.assert_invariant(base, inv.run_task_on_states(task, inv.rescale_docs(states, 2.0)),
                         keys, rtol=1e-9)
    inv.assert_invariant(base, inv.run_task_on_states(task, inv.embed_isometric_docs(states, 2, 3)),
                         keys, rtol=1e-4)
    model, tok, texts = _tiny_and_texts()
    base_m = inv.run_task_on_model(task, model, tok, texts)
    ins = inv.run_task_on_model(task, inv.insert_identity_blocks(model, [0, 1]), tok, texts)
    inv.assert_invariant(base_m, ins, keys, rtol=0.0)


def _tiny_and_texts():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-gpt2", dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"tiny-random-gpt2 not cached: {e}")
    from blme.cache import load_default_corpus
    return model, tok, [d["text"] for d in load_default_corpus(96)]


def test_middle_layer_moves_under_identity_insertion():
    """Known negative: the index-selected middle layer shifts when blocks are added."""
    model, tok, texts = _tiny_and_texts()
    base = inv.run_task_on_model("geometry_information_imbalance", model, tok, texts)
    ins = inv.run_task_on_model("geometry_information_imbalance",
                                inv.insert_identity_blocks(model, [0, 0, 0]), tok, texts)
    assert base["_meta_ii_mid_to_last"] != ins["_meta_ii_mid_to_last"]
