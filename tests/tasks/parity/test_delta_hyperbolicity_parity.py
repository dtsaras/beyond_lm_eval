"""Parity: geometry_delta_hyperbolicity vs the RELEASED HypLoRA code
(marlin-codes/HypLoRA@c888e16, ``utils/delta_hyperbolicity.py``):
``delta_hyp`` and ``get_delta``, compiled from the clone's source (the module
imports matplotlib and loads a model at the bottom), plus its per-prompt loop.
Skips when the clone is absent (fails under BLME_REQUIRE_REFS=1).
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "invariance"))

import _family_d as fd  # noqa: E402
from blme.cache import load_default_corpus  # noqa: E402
from blme.tasks.geometry import delta_hyperbolicity as DH  # noqa: E402

_ROOT = os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))
_REPO = os.path.join(_ROOT, "HypLoRA")


def _missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


@pytest.fixture(scope="module")
def ref():
    path = os.path.join(_REPO, "utils", "delta_hyperbolicity.py")
    if not os.path.isfile(path):
        _missing(path)
    from scipy.spatial import distance_matrix
    tree = ast.parse(open(path).read())
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"delta_hyp", "get_delta"}]
    assert len(defs) == 2
    ns = {"np": np, "distance_matrix": distance_matrix}
    exec(compile(ast.Module(body=defs, type_ignores=[]), path, "exec"), ns)
    return ns


def test_sources_pin_the_cloned_commit():
    if not os.path.isdir(_REPO):
        _missing(_REPO)
    head = subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert ("@" + head) in DH.DeltaHyperbolicityTask.SOURCES["code"]


@pytest.mark.parametrize("n,d,chunk", [(40, 8, 64), (97, 16, 7), (64, 300, 13)])
def test_estimator_matches_the_release(ref, n, d, chunk):
    X = np.random.default_rng(n).standard_normal((n, d))
    D = DH._distances(X)
    assert DH.delta_hyp(D, chunk=chunk) == ref["delta_hyp"](D)
    delta, diam = ref["get_delta"](torch.as_tensor(X), max_points=1500)
    assert DH.relative_delta(X) == pytest.approx(2 * delta / diam, rel=1e-12)


def test_anchors():
    """Points on a line (a tree metric) have delta 0; scaling and rotating the
    cloud leave the ratio unchanged; a Gaussian cloud is far from a tree."""
    line = np.c_[np.arange(30.0), np.zeros(30)]
    assert DH.relative_delta(line) == pytest.approx(0.0, abs=1e-12)
    X = np.random.default_rng(0).standard_normal((50, 12))
    q = np.linalg.qr(np.random.default_rng(1).standard_normal((12, 12)))[0]
    assert DH.relative_delta(5.0 * X @ q) == pytest.approx(DH.relative_delta(X), rel=1e-10)
    assert DH.relative_delta(X) > 0.2


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_input_embedding_value_matches_the_release_loop(ref, arch):
    """The release's loop: input_embeddings = model.get_input_embeddings()(ids),
    get_delta, 2 delta / diam, mean over prompts."""
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=2)
    tok = fd.tokenizer()
    data = load_default_corpus(40)
    cfg = {"num_samples": 6, "n_points": 24, "min_bytes": 300, "use_dataset": True}
    r = DH.DeltaHyperbolicityTask(cfg).evaluate(model, tok, data)
    texts = [d["text"] for d in data if len(d["text"].encode("utf-8")) >= 300][:6]
    ratios = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=24)["input_ids"]
            delta, diam = ref["get_delta"](model.get_input_embeddings()(ids))
            ratios.append(2 * delta / diam if diam > 0 else 0)
    assert r["_meta_n_prompts"] == 6
    assert r["delta_rel_input_embedding"] == pytest.approx(np.mean(ratios), rel=1e-6)
