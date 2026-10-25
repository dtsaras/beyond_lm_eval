"""Disk offload must not change the statistic, only where the bytes live.

`consistency.py` builds WITHIN-document controls and treats every element of
`stats["hidden"]` as one document. The offloaded collector returned ONE tensor
for the whole corpus, so every control became corpus-wide -- a storage-induced
change to the declared algorithm.
"""
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")

DOCS = [{"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "Machine learning models learn representations of language."},
        {"text": "A third document, with its own distinct content here."},
        {"text": "Neural networks trained on text develop internal structure."}]


@pytest.fixture(scope="module")
def gpt2():
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2").eval()
    return model, tok


@pytest.fixture
def restore_offload():
    prev = os.environ.get("BLME_DISK_OFFLOAD")
    yield
    if prev is None:
        os.environ.pop("BLME_DISK_OFFLOAD", None)
    else:
        os.environ["BLME_DISK_OFFLOAD"] = prev


def _collect(model, tok, mode):
    from blme.tasks.geometry.utils import collect_prediction_stats
    os.environ["BLME_DISK_OFFLOAD"] = mode
    return collect_prediction_stats(model, tok, DOCS, num_samples=len(DOCS))[0]


def test_offload_preserves_document_segmentation(gpt2, restore_offload):
    model, tok = gpt2
    fresh, disk = _collect(model, tok, "0"), _collect(model, tok, "1")

    assert len(disk["labels"]) == len(DOCS), (
        f"offload returned {len(disk['labels'])} documents, expected {len(DOCS)}")
    for key in ("logits", "labels", "hidden"):
        assert [len(x) for x in fresh[key]] == [len(x) for x in disk[key]], (
            f"{key} document lengths differ between storage routes")
        # FIXTURE INTEGRITY: only the segmentation may differ, never the bytes.
        cat_f = torch.cat([x.reshape(len(x), -1) if x.ndim > 1 else x
                           for x in fresh[key]])
        cat_d = torch.cat([x.reshape(len(x), -1) if x.ndim > 1 else x
                           for x in disk[key]])
        assert torch.equal(cat_f, cat_d), f"{key} bytes differ, not just layout"


def test_headline_is_identical_across_storage_routes(gpt2, restore_offload):
    from blme.tasks.geometry.consistency import PredictionAlignmentTask
    model, tok = gpt2
    out = {}
    for mode in ("0", "1"):
        os.environ["BLME_DISK_OFFLOAD"] = mode
        out[mode] = PredictionAlignmentTask(
            config={"num_samples": len(DOCS), "use_cache": False}
        ).evaluate(model, tok, dataset=DOCS)

    numeric = [k for k, v in out["0"].items() if isinstance(v, (int, float))]
    assert numeric, "task emitted no numeric keys"
    for k in numeric:
        a, b = out["0"][k], out["1"][k]
        assert a == b or (a != a and b != b), (
            f"{k} differs between storage routes: {a!r} vs {b!r}")
