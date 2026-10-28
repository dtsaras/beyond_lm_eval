"""geometry_intrinsic_dim: cache gate and the one-point-per-sequence protocol."""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from blme.cache import ModelOutputCache  # noqa: E402
from blme.tasks.geometry.intrinsic_dim import IntrinsicDimensionTask  # noqa: E402

TEXTS = [
    "The cat sat on the mat and looked out of the window at the rain.",
    "Stock markets fell sharply on Tuesday after the central bank raised rates.",
    "In 1905 Einstein published four papers that changed physics forever.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "She opened the old wooden door and stepped into the dusty library.",
    "The recipe calls for two cups of flour, one egg and a pinch of salt.",
    "Volcanoes form where tectonic plates diverge or where one plate subducts.",
    "The orchestra tuned their instruments while the audience took their seats.",
]


@pytest.fixture(scope="module")
def tiny():
    name = "hf-internal-testing/tiny-random-gpt2"
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name)
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"{name} not cached: {e}")
    return model, tok


def test_cache_without_hidden_states(tiny):
    """A cache populated for a logits-only consumer returns no hidden states;
    the layerwise route self-collects instead of raising AttributeError."""
    model, tok = tiny
    data = [{"text": t} for t in TEXTS]
    cache = ModelOutputCache(model, tok, dataset=data, num_samples=len(data))
    cache.populate(need_hidden=False, need_logits=True)
    assert cache.is_populated and cache.get_hidden_states(layer_idx="all") is None
    out = IntrinsicDimensionTask({"layerwise": True, "num_samples": len(data)}).evaluate(
        model, tok, data, cache=cache)
    assert "error" not in out and "lid_first_layer" in out
    assert out["_meta_hidden_states_source"] == "self_collected"
    assert out["_meta_cache_rejected"] == "no_hidden_states"
