"""Parity: geometry_rogue_dimensions vs the RELEASED notebook
(wtimkey/rogue-dimensions@84f4009, replication.ipynb): its ``cos_contrib``
(cell 14) and the Section 3.1 loop (cell 15), executed from the notebook's
own source. Skips when the clone is absent (fails under BLME_REQUIRE_REFS=1).
"""
import itertools
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "invariance"))

import _family_d as fd  # noqa: E402
from blme.cache import load_long_documents  # noqa: E402
from blme.tasks.geometry import rogue_dimensions as RD  # noqa: E402

_ROOT = os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))
_REPO = os.path.join(_ROOT, "rogue-dimensions")


def _missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


@pytest.fixture(scope="module")
def cos_contrib():
    path = os.path.join(_REPO, "replication.ipynb")
    if not os.path.isfile(path):
        _missing(path)
    cells = json.load(open(path))["cells"]
    src = "".join(cells[14]["source"])
    assert src.lstrip().startswith("#this is the cosine contribution function")
    ns = {"np": np}
    exec(compile(src, path + ":cell14", "exec"), ns)
    return ns["cos_contrib"]


def test_sources_pin_the_cloned_commit():
    if not os.path.isdir(_REPO):
        _missing(_REPO)
    head = subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert ("@" + head) in RD.RogueDimensionsTask.SOURCES["code"]


def _cloud(n=60, d=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X[:, 3] += 6.0          # a rogue coordinate: a shared offset
    return X


def test_exact_cc_equals_the_release_over_every_pair(cos_contrib):
    X = _cloud()
    pairs = [p for p in itertools.permutations(range(X.shape[0]), 2)]
    ref = np.mean([cos_contrib(X[a].astype(np.float64), X[b].astype(np.float64))
                   for a, b in pairs], axis=0)
    np.testing.assert_allclose(RD.cosine_contributions(X), ref, rtol=1e-12, atol=1e-15)
    shares, aniso = RD.top_shares(ref)
    assert RD.top_shares(RD.cosine_contributions(X))[0] == pytest.approx(shares, rel=1e-12)
    assert shares[0] > 0.9 and aniso == pytest.approx(ref.sum())


def test_release_monte_carlo_agrees_within_sampling_error(cos_contrib):
    """Cell 15's estimator (random.seed(2398), random distinct pairs) on a
    cloud of 2000 rows; the exact value lies within 5 standard errors."""
    X = _cloud(n=2000, d=16, seed=1)
    state = random.getstate()
    random.seed(2398)
    try:
        pairs = [random.sample(range(X.shape[0]), 2) for _ in range(20000)]
    finally:
        random.setstate(state)
    contribs = np.array([cos_contrib(X[a], X[b]) for a, b in pairs])
    se = contribs.std(axis=0, ddof=1) / np.sqrt(len(pairs))
    exact = RD.cosine_contributions(X)
    assert np.all(np.abs(contribs.mean(axis=0) - exact) <= 5 * se + 1e-12)


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_evaluate_matches_the_release_readout(cos_contrib, arch):
    """Cell 7's readout (tokenize, first context_length tokens, no special
    tokens, every hidden state stacked) with the exact all-pairs CC."""
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=3)
    tok = fd.tokenizer()
    docs = load_long_documents(4, split="validation")
    r = RD.RogueDimensionsTask({"num_samples": 4, "context_length": 32, "use_dataset": True}).evaluate(
        model, tok, docs)
    embs = []
    with torch.no_grad():
        for d in docs:
            toks = tok.tokenize(d["text"])[:32]
            out = model(torch.tensor([tok.convert_tokens_to_ids(toks)]), output_hidden_states=True)
            embs.append(torch.stack(out.hidden_states).squeeze(1).numpy())
    sample = np.concatenate(embs, axis=1)
    for layer in range(sample.shape[0]):
        cc = RD.cosine_contributions(sample[layer])
        top = np.argsort(cc)[-3:]
        aniso = cc.sum()
        assert r["rogue_top1_share_profile"][layer] == pytest.approx(cc[top[2]] / aniso, rel=1e-9)
        assert r["rogue_top3_share_profile"][layer] == pytest.approx(cc[top].sum() / aniso, rel=1e-9)
    assert r["_meta_n_tokens"] == sample.shape[1] == 128


# Timkey & van Schijndel (2021) Table 4, GPT-2 (85k tokens of Wikipedia,
# 128-token inputs, 500k pairs): top-1 share and anisotropy per layer.
_TABLE4_TOP1 = [0.054, 0.324, 0.319, 0.294, 0.297, 0.324, 0.351, 0.374, 0.376, 0.364, 0.326,
                0.275, 0.763]
_TABLE4_ANISO = [0.484, 0.626, 0.612, 0.589, 0.549, 0.517, 0.485, 0.466, 0.461, 0.466, 0.498,
                 0.640, 0.885]


def test_gpt2_reproduces_table_4():
    """Known positive: on 150 WikiText-103 articles (19k tokens) gpt2's
    per-layer top-1 shares and anisotropy match the paper's Table 4 to 0.015
    (tolerance 0.03), the output layer's single rogue dimension included."""
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"gpt2 not cached: {e}")
    r = RD.RogueDimensionsTask({}).evaluate(model, tok, None)
    assert r["rogue_top1_share_profile"] == pytest.approx(_TABLE4_TOP1, abs=0.03)
    assert r["anisotropy_profile"] == pytest.approx(_TABLE4_ANISO, abs=0.03)
    assert r["rogue_top1_share_max"] == r["rogue_top1_share_last"]
