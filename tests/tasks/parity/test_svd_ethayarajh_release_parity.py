"""Parity: geometry_svd's ``avg_cosine_similarity`` vs the RELEASED estimator
of Ethayarajh (2019), kawine/contextual ``analyze.py::explore_embedding_space``.

The module imports allennlp, so the function is taken with ``ast`` and run on
an in-memory stand-in for its HDF5 file (sentence id -> (layers, words, d)),
holding the states evaluate() reads (HF's post-norm last entry). The release
samples one word per sentence and averages the cosine of ``num_samples``
pairs drawn with replacement, so a pair is the same vector with probability
1/N: its expectation is 1/N + (1 - 1/N) * (mean over distinct pairs), which is
BLME's exact all-pairs value when tokens within and across sentences are
exchangeable, as here. Agreement is required within 5 standard errors of the
release's pair sampling plus its rounding to 3 decimals. On gpt2 over the
clone's own ``sts.csv`` the per-layer values reproduce the released
``gpt2/embedding_space_stats.json`` behind the paper's Fig. 1. Skips without
the clone (fails under BLME_REQUIRE_REFS=1).
"""
import ast
import csv
import json
import os
import random
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cosine

_TESTS = Path(__file__).resolve().parents[2]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import invariance as inv  # noqa: E402
from blme.tasks.geometry.isotropy import SVDIsotropyTask  # noqa: E402

TruncatedSVD = pytest.importorskip("sklearn.decomposition").TruncatedSVD

_REPO = Path(os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))) \
    / "contextual"
_ANALYZE = _REPO / "analyze.py"
N_DOCS, T, D = 3000, 3, 128     # TruncatedSVD(n_components=100) needs d > 100


def _missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


def test_sources_pin_the_cloned_commit():
    if not _REPO.is_dir():
        _missing(_REPO)
    head = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert ("kawine/contextual@" + head) in SVDIsotropyTask.SOURCES["code"]


@pytest.fixture(scope="module")
def explore_embedding_space():
    if not _ANALYZE.is_file():
        _missing(_ANALYZE)
    body = ast.parse(_ANALYZE.read_text()).body
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name == "explore_embedding_space"]
    assert len(nodes) == 1
    store = {}
    ns = {"np": np, "json": json, "random": random, "cosine": cosine,
          "TruncatedSVD": TruncatedSVD,
          "Tqdm": types.SimpleNamespace(tqdm=lambda it: it),
          "h5py": types.SimpleNamespace(File=lambda fn, mode: store[fn])}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_ANALYZE), "exec"), ns)
    return ns["explore_embedding_space"], store


def test_avg_cosine_matches_the_release_estimator(explore_embedding_space, tmp_path):
    fn, store = explore_embedding_space
    g = np.random.default_rng(11)
    mu = 2.0 * g.standard_normal(D) / np.sqrt(D)
    final = [(mu + g.standard_normal((T, D)) / np.sqrt(D)).astype(np.float32)
             for _ in range(N_DOCS)]
    blocks = {li: [g.standard_normal((T, D)).astype(np.float32) for _ in range(N_DOCS)]
              for li in range(3)}
    res = inv.run_task_on_states("geometry_svd", blocks, final_norm=final)
    assert res["_meta_svd_mode"] == "single_layer"
    ours = res["avg_cosine_similarity"]

    # float64 copies of the float32 states (exact): the release's json.dump
    # rejects the numpy float32 scalars scipy now returns for float32 input.
    store["emb"] = {str(i): x[None].astype(np.float64) for i, x in enumerate(final)}
    out = tmp_path / "stats.json"
    state = random.getstate()
    random.seed(0)
    try:
        fn("emb", str(out), num_samples=N_DOCS)
    finally:
        random.setstate(state)
    ref = json.loads(out.read_text())["mean cosine similarity across words"]["layer_0"]

    U = np.concatenate(final).astype(np.float64)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    C = U @ U.T
    sd = C[np.triu_indices(len(U), 1)].std()
    expected = 1.0 / N_DOCS + (1.0 - 1.0 / N_DOCS) * ours
    assert 0.5 < ours < 0.95 and sd > 0.01
    assert abs(ref - expected) <= 5.0 * sd / np.sqrt(N_DOCS) + 5e-4


def test_gpt2_reproduces_the_released_anisotropy_profile():
    """Ethayarajh (2019) Fig. 1, GPT-2 "mean cosine similarity across words"
    (gpt2/embedding_space_stats.json): 0.978 at layer 12, HF's post-norm last
    entry; ours 0.9705 (-0.77%) on the 196 sentences of the clone's sts.csv.
    BLME's layerwise index k is Ethayarajh's layer k + 1 (block outputs, the
    last post-norm); every layer is within 0.032 (layer 10). The released
    values are 1000-pair estimates over one word per sentence of the full STS
    2012-16 corpus rounded to 3 decimals, so the corpus sample dominates: the
    tolerance is 0.015 at layer 12 and 0.04 per layer."""
    transformers = pytest.importorskip("transformers")
    import torch
    stats, sts = _REPO / "gpt2" / "embedding_space_stats.json", _REPO / "sts.csv"
    if not (stats.is_file() and sts.is_file()):
        _missing(stats)
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"gpt2 not cached: {e}")
    pub = json.loads(stats.read_text())["mean cosine similarity across words"]
    with open(sts) as f:
        docs = [{"text": r[k]} for r in csv.DictReader(f, delimiter="\t") for k in ("Sent1", "Sent2")]
    res = SVDIsotropyTask({"num_samples": len(docs), "use_cache": False, "layerwise": True}).evaluate(
        model, tok, docs)
    prof = [res["_meta_layer_metrics"][f"layer_{k}"]["avg_cosine_similarity"] for k in range(12)]
    assert res["avg_cosine_similarity"] == prof[-1]
    assert prof[-1] == pytest.approx(pub["layer_12"], abs=0.015)
    assert prof == pytest.approx([pub[f"layer_{k + 1}"] for k in range(12)], abs=0.04)
    assert int(np.argmin(prof)) in (3, 4, 5) and prof[-1] > 0.95
