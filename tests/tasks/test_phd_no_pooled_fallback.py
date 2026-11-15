"""geometry_phd_dimension: when no text gives a per-text PHD the headline is
NaN with a reason; the pooled estimand never takes its place."""
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from blme.tasks.geometry.phd_dimension import PHDimensionTask  # noqa: E402


def test_short_texts_give_nan_headline_with_reason_not_pooled_value():
    """Every fixture text is under the notebook's 50-token floor, so no text
    is estimated; the 16 texts pooled still form a cloud large enough for the
    pooled estimand."""
    task = PHDimensionTask(config={"documents": "dataset", "num_samples": len(fd.TEXTS),
                                   "use_cache": False})
    r = task.evaluate(fd.tiny_gpt2(), fd.tokenizer(), fd.dataset())
    assert "error" not in r, r
    assert math.isfinite(r["_meta_phd_dimension_pooled"])
    assert math.isnan(r["phd_dimension"]), r["phd_dimension"]
    assert math.isnan(r["phd_dimension_std"])
    reason = r["_meta_phd_nan_reason"]
    assert f"{len(fd.TEXTS)} seen, {len(fd.TEXTS)} under 50 tokens" in reason, reason
    assert "_meta_phd_n_texts" not in r
