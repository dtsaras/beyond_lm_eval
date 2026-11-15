"""A populated cache is not necessarily a USABLE cache.

A cache filled for an attention-only consumer returns None from
`get_hidden_states`. Tasks that accepted the cache on population + document
count alone then returned an error instead of self-collecting, so merely
CO-RUNNING another task deleted every one of their headlines.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))
import _family_d as fd  # noqa: E402

# 31-39 words each, so the shipped readout (the last token of a 20-word window)
# places one point per document; shorter documents give it none.
DOCS = [{"text": " ".join(fd.TEXTS[(i + j) % len(fd.TEXTS)] for j in range(3))}
        for i in range(12)]

# The shipped readout reads its points from the window forwards kept on the
# shared cache; the whole-passage readout asks the cache for hidden states,
# gets None, and has to collect its own.
ROUTES = [(None, "last_token_of_20_word_window", "shared_window_cache"),
          ("passage", "last_content_token_of_passage", "self_collected")]


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


def _config(base, readout):
    return dict(base, points_readout=readout) if readout else dict(base)


@pytest.mark.parametrize("readout,name,source", ROUTES)
def test_vendi_self_collects_instead_of_erroring(gpt2, attention_only_cache,
                                                 readout, name, source):
    from blme.tasks.geometry.vendi import VendiScoreTask
    model, tok = gpt2
    out = VendiScoreTask(
        config=_config({"num_samples": len(DOCS), "max_points": 128}, readout)
    ).evaluate(model, tok, dataset=DOCS, cache=attention_only_cache)
    assert "error" not in out, out.get("error")
    assert out["_meta_points_readout"] == name
    assert out["_meta_points_source"] == source
    assert out["_meta_n_points"] == len(DOCS)
    for key in VendiScoreTask.HEADLINE_KEYS:
        assert 1.0 <= out[key] <= len(DOCS), (key, out[key])


@pytest.mark.parametrize("readout,name,source", ROUTES)
def test_magnitude_self_collects_instead_of_erroring(gpt2, attention_only_cache,
                                                     readout, name, source):
    """The second site of the same defect, found by sweeping the population."""
    from blme.tasks.geometry.magnitude import MagnitudeTask
    model, tok = gpt2
    out = MagnitudeTask(
        config=_config({"num_samples": 32}, readout)
    ).evaluate(model, tok, dataset=DOCS, cache=attention_only_cache)
    assert "error" not in out, out.get("error")
    assert out["_meta_points_readout"] == name
    assert out["_meta_magnitude_source"] == out["_meta_points_source"] == source
    assert out["_meta_n_points"] == len(DOCS)
    assert out["_meta_magnitude_tconv_status"] == "ok"
    # Mean magnitude over [0, t_conv] lies between one point and all of them.
    assert 1.0 < out["magnitude_area_tconv"] < len(DOCS)
    assert np.isfinite(out["magnitude_dimension_tconv"])
    assert out["magnitude_dimension_tconv"] > 0.0
