"""geometry_phd_dimension: precision of the private forward and the GPTID protocol."""
import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

TEXTS = [
    "The cat sat on the mat and looked out of the window at the rain.",
    "Stock markets fell sharply on Tuesday after the central bank raised rates.",
]


@pytest.fixture(scope="module")
def tiny64():
    name = "hf-internal-testing/tiny-random-gpt2"
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name)
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.float64).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"{name} not cached: {e}")
    return model, tok


def test_private_forward_keeps_float64_states(tiny64):
    """The per-text and pooled collectors take the states as the model emits
    them: a float64 model's rows reach the estimator unrounded (they were cast
    to float32 first)."""
    from blme.tasks.geometry.phd_dimension import _gather_point_cloud, _iter_per_document_clouds
    model, tok = tiny64
    data = [{"text": t} for t in TEXTS]
    with torch.no_grad():
        ref = [model(**tok(t, return_tensors="pt"), output_hidden_states=True)
               .hidden_states[-1][0].numpy() for t in TEXTS]
    assert any(not np.array_equal(r, r.astype(np.float32).astype(np.float64)) for r in ref)
    docs = list(_iter_per_document_clouds(model, tok, data, num_samples=2, max_length=64,
                                          layer_idx=-1, pooling="all_tokens"))
    assert all(np.array_equal(d, r) for d, r in zip(docs, ref))
    pooled = _gather_point_cloud(model, tok, data, num_samples=2, max_length=64, layer_idx=-1,
                                 pooling="all_tokens", cache=None, use_cache=False)
    assert np.array_equal(pooled, np.concatenate(ref))
