"""intrinsic_dim reports which estimand ran, as geometry_lid does with _meta_lid_mode."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from blme.cache import load_default_corpus  # noqa: E402
from blme.tasks.geometry.intrinsic_dim import IntrinsicDimensionTask  # noqa: E402


def test_mode_flag_names_the_estimand():
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    static = IntrinsicDimensionTask({}).evaluate(model, tok, None)
    layerwise = IntrinsicDimensionTask({"layerwise": True, "num_samples": 30}).evaluate(
        model, tok, load_default_corpus(40))
    assert static["_meta_intrinsic_dim_mode"] == "static_embeddings"
    assert layerwise["_meta_intrinsic_dim_mode"] == "layerwise"


def test_layerwise_mode_emits_every_headline_key():
    """The study runs layerwise mode, so a headline key it never emits would be
    a column of NaN. intrinsic_dimension belongs to the static mode alone and
    stays there as a plain key."""
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    layerwise = IntrinsicDimensionTask({"layerwise": True, "num_samples": 30}).evaluate(
        model, tok, load_default_corpus(40))
    missing = [k for k in IntrinsicDimensionTask.HEADLINE_KEYS if k not in layerwise]
    assert not missing, missing
    assert "intrinsic_dimension" not in layerwise
    static = IntrinsicDimensionTask({}).evaluate(model, tok, None)
    assert static["intrinsic_dimension"] > 0
    assert "intrinsic_dimension" not in IntrinsicDimensionTask.SIZE_INVARIANT
