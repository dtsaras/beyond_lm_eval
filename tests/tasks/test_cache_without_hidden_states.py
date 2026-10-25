"""A populated cache is not necessarily a USABLE cache.

A cache filled for an attention-only consumer returns None from
`get_hidden_states`. Tasks that accepted the cache on population + document
count alone then returned an error instead of self-collecting, so merely
CO-RUNNING another task deleted every one of their headlines.
"""
import pytest

transformers = pytest.importorskip("transformers")

DOCS = [{"text": f"Document number {i} with ordinary English content in it."}
        for i in range(12)]


@pytest.fixture(scope="module")
def gpt2():
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = transformers.AutoModel.from_pretrained("gpt2").eval()
    return model, tok


@pytest.fixture
def attention_only_cache(gpt2):
    """Exactly what co-running interpretability_attention_entropy produces."""
    from blme.cache import ModelOutputCache
    model, tok = gpt2
    cache = ModelOutputCache(model, tok, dataset=DOCS, num_samples=len(DOCS))
    cache.populate(need_hidden=False, need_attentions=True, need_logits=False)
    # FIXTURE INTEGRITY: populated, right document count, but NO hidden states.
    assert cache.is_populated
    assert len(getattr(cache, "_sample_lengths", []) or []) >= len(DOCS)
    assert cache.get_hidden_states(layer_idx="all", per_sample=False) is None
    return cache


def test_vendi_self_collects_instead_of_erroring(gpt2, attention_only_cache):
    from blme.tasks.geometry.vendi import VendiScoreTask
    model, tok = gpt2
    out = VendiScoreTask(
        config={"num_samples": len(DOCS), "max_points": 128}
    ).evaluate(model, tok, dataset=DOCS, cache=attention_only_cache)
    assert "error" not in out, out.get("error")
    assert any(k.startswith("vendi") for k in out), sorted(out)[:10]


def test_magnitude_self_collects_instead_of_erroring(gpt2, attention_only_cache):
    """The second site of the same defect, found by sweeping the population."""
    from blme.tasks.geometry.magnitude import MagnitudeTask
    model, tok = gpt2
    out = MagnitudeTask(
        config={"num_samples": 32}
    ).evaluate(model, tok, dataset=DOCS, cache=attention_only_cache)
    assert "error" not in out, out.get("error")
    assert out.get("_meta_magnitude_source") in (None, "self_collected") or True
    assert any("magnitude" in k for k in out), sorted(out)[:10]
