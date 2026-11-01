"""Parity: geometry_representation_dispersion vs the RELEASED rep_dispersion
(yanhong-lbh/rep_dispersion@dae432d), run end to end: ``extract_lm_embeddings``
on tiny-random-gpt2 over a local text file, then
``metrics.average_pairwise_cosine_distance``.

rep_dispersion is not a BLME dependency: it is installed with ``uv pip install
--no-deps --target ~/.cache/blme-sideinstalls/rep_dispersion-dae432d`` from
the clone (BLME_SIDEINSTALLS overrides the root). Skips when absent (fails
under BLME_REQUIRE_REFS=1).
"""
import os
import subprocess
import sys

import pytest
import torch

transformers = pytest.importorskip("transformers")

from blme.tasks.geometry import dispersion as D  # noqa: E402

_ROOT = os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))
_SIDE = os.path.join(os.environ.get("BLME_SIDEINSTALLS", os.path.expanduser("~/.cache/blme-sideinstalls")),
                     "rep_dispersion-dae432d")
_TINY = "hf-internal-testing/tiny-random-gpt2"


def _missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


@pytest.fixture(scope="module")
def ref():
    if not os.path.isdir(os.path.join(_SIDE, "rep_dispersion")):
        _missing(_SIDE)
    sys.path.insert(0, _SIDE)
    try:
        from rep_dispersion import data, embeddings, metrics
    finally:
        sys.path.remove(_SIDE)
    return data, embeddings, metrics


def test_sources_pin_the_cloned_commit():
    path = os.path.join(_ROOT, "rep_dispersion")
    if not os.path.isdir(path):
        _missing(path)
    head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert ("@" + head) in D.RepresentationDispersionTask.SOURCES["code"]


def test_rows_match_the_release_loader(ref):
    data, _, _ = ref
    rows = data.load_texts("Salesforce/wikitext", subset="wikitext-103-raw-v1",
                           split="validation", text_field="text")
    assert D.wikitext_validation_rows() == rows


@pytest.mark.parametrize("n_chunks", [None, 50, 10000])
def test_chunks_match_the_release(ref, n_chunks):
    data, _, _ = ref
    tok = transformers.AutoTokenizer.from_pretrained(_TINY)
    rows = D.wikitext_validation_rows()[:400]
    expected = data.make_token_chunks(tok, rows, chunk_len=128, n_chunks=n_chunks,
                                      random_sample=True, seed=42, drop_last=True)
    assert D.token_chunks(tok, rows, 128, n_chunks, 42) == expected


def test_evaluate_matches_the_release_end_to_end(ref, tmp_path):
    data, embeddings, metrics = ref
    rows = D.wikitext_validation_rows()[:300]
    text_file = tmp_path / "rows.txt"
    text_file.write_text("\n\n".join(rows), encoding="utf-8")
    cfg = embeddings.ExtractionConfig(model=_TINY, dataset=str(text_file), chunk_len=64,
                                      n_chunks=40, random_sample=True, seed=42, device="cpu",
                                      dtype="float32", trust_remote_code=False)
    paths = embeddings.extract_lm_embeddings(cfg, tmp_path / "out")
    ref_value = metrics.average_pairwise_cosine_distance(torch.load(paths["embeddings"]))

    tok = transformers.AutoTokenizer.from_pretrained(_TINY)
    model = transformers.AutoModelForCausalLM.from_pretrained(_TINY, dtype=torch.float32).eval()
    r = D.RepresentationDispersionTask({"chunk_len": 64, "n_chunks": 40, "use_dataset": True}).evaluate(
        model, tok, [{"text": t} for t in rows])
    assert r["_meta_n_chunks"] == 40
    assert r["representation_dispersion"] == ref_value


def test_metric_matches_the_release(ref):
    _, _, metrics = ref
    g = torch.Generator().manual_seed(0)
    for x in (torch.randn(50, 16, generator=g), torch.randn(7, 300, generator=g) + 3.0):
        assert D.average_pairwise_cosine_distance(x) == metrics.average_pairwise_cosine_distance(x)
    # Identical directions are at distance 0 and antipodal ones at 2.
    v = torch.randn(1, 8, generator=g)
    assert D.average_pairwise_cosine_distance(torch.cat([v, 2 * v, 0.5 * v])) == pytest.approx(0, abs=1e-6)
    assert D.average_pairwise_cosine_distance(torch.cat([v, -v])) == pytest.approx(2.0, rel=1e-6)
