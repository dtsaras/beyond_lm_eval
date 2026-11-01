"""Parity: geometry_dime vs the RELEASED code, loaded from the clones:

* uk-cliplab/representation-itl@e5d8a12: ``matrix_itl.vonNeumannEntropy`` and
  ``difference_of_entropies.doe``;
* OFSkean/information_flow@dde387a: ``metric_functions.normalize`` and
  ``compute_dime``, ``text_dataloader.text_augmentation`` and
  ``TextLayerwiseAutoModelWrapper._get_pooled_hidden_states`` (the last two
  taken from their files with ``ast``, since the modules import umap, llm2vec
  and matplotlib at the top).

The release draws doe's permutations from the global torch RNG, one layer
after another; BLME uses the same seeded draws at every layer, so the release
is reseeded before each layer. The release is unseeded; its augmentation is
run under BLME's per-view seeds. Skips when a clone or nlpaug is absent
(fails under BLME_REQUIRE_REFS=1).
"""
import ast
import importlib.util
import math
import os
import random
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "invariance"))

import _family_d as fd  # noqa: E402
from _nlpaug_side import missing, require_nlpaug  # noqa: E402
from blme.cache import load_default_corpus  # noqa: E402
from blme.tasks.geometry import dime as DM  # noqa: E402

_ROOT = os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))
_IF = os.path.join(_ROOT, "information_flow")
_ITL = os.path.join(_ROOT, "representation-itl")
SEED = 3


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _functions(path, names, cls=None):
    """The named top-level functions (or methods of ``cls``) of a release file."""
    body = ast.parse(Path(path).read_text()).body
    if cls is not None:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == cls).body
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(n.name for n in nodes) == sorted(names), path
    ns = {"torch": torch, "np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def repitl():
    src = os.path.join(_ITL, "src")
    if not os.path.isdir(os.path.join(src, "repitl")):
        missing(src)
    sys.path.insert(0, src)
    try:
        import repitl.difference_of_entropies as dent
        import repitl.matrix_itl as itl
    finally:
        sys.path.remove(src)
    return itl, dent


@pytest.fixture(scope="module")
def release(repitl):
    utils = os.path.join(_IF, "experiments", "utils")
    if not os.path.isdir(utils):
        missing(utils)
    # compute_intrinsic_dimension's dadapy import is unused by DiME.
    stub = types.ModuleType("dadapy")
    stub.data = types.ModuleType("dadapy.data")
    stub.data.Data = None
    saved = {k: sys.modules.get(k) for k in ("dadapy", "dadapy.data")}
    sys.modules.update({"dadapy": stub, "dadapy.data": stub.data})
    try:
        mf = _load("_information_flow_metric_functions",
                   os.path.join(utils, "metrics", "metric_functions.py"))
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k)
            else:
                sys.modules[k] = v
    aug = _functions(os.path.join(utils, "dataloaders", "text_dataloader.py"),
                     ["text_augmentation"])
    pool = _functions(os.path.join(utils, "model_definitions", "text_automodel_wrapper.py"),
                      ["_get_pooled_hidden_states"], cls="TextLayerwiseAutoModelWrapper")
    return types.SimpleNamespace(mf=mf, text_augmentation=aug["text_augmentation"],
                                 pool=pool["_get_pooled_hidden_states"])


@pytest.mark.parametrize("repo", ["information_flow", "representation-itl"])
def test_sources_pin_the_cloned_commit(repo):
    path = os.path.join(_ROOT, repo)
    if not os.path.isdir(path):
        missing(path)
    head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert ("@" + head) in DM.DiMETask.SOURCES["code"]


def _grams(n, d, rank, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rank, d, generator=g, dtype=torch.float64)
    a = torch.randn(n, rank, generator=g, dtype=torch.float64) @ w
    b = a + 0.5 * torch.randn(n, rank, generator=g, dtype=torch.float64) @ w
    a, b = DM.unit_rows(a), DM.unit_rows(b)
    return a @ a.T, b @ b.T


def _release_doe(dent, kx, ky, seed, n_iters=10):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return float(dent.doe(kx, ky, alpha=1, n_iters=n_iters))


@pytest.mark.parametrize("n,d,rank", [(300, 40, 6), (200, 16, 16), (120, 500, 3)])
def test_entropy_and_doe_match_repitl(repitl, n, d, rank):
    itl, dent = repitl
    kx, ky = _grams(n, d, rank, seed=n)
    for k in (kx, ky, kx * ky):
        assert float(DM.von_neumann_entropy(k)) == pytest.approx(
            float(itl.vonNeumannEntropy(k)), rel=1e-12, abs=1e-12)
    assert float(DM.von_neumann_entropy(kx * ky)) == pytest.approx(
        float(itl.matrixAlphaJointEntropy([kx, ky], alpha=1)), rel=1e-12, abs=1e-12)
    ours = DM.doe(kx, ky, DM.permutations(n, 10, SEED))
    assert ours == pytest.approx(_release_doe(dent, kx, ky, SEED), rel=1e-10, abs=1e-12)
    assert ours > 0


def test_augmentation_matches_the_release(release):
    """Same per-view seeds: the release's text_augmentation (a fresh chain per
    call) returns str() of nlpaug's one-element list, i.e. the same text."""
    require_nlpaug()
    aug = DM.augmenter()
    texts = [d["text"] for d in load_default_corpus(12)]
    for i, text in enumerate(texts):
        for v in range(DM.N_VIEWS):
            with DM.seeded_global_rngs(*DM.view_seeds(SEED, i, v)):
                ref = release.text_augmentation([text])[0]
            assert ref[:2] in ("['", '["') and ref[-2:] in ("']", '"]')
            ours = DM.augment_view(aug, text, SEED, i, v)
            assert [ours] == ast.literal_eval(ref)
            assert ours != text
    assert DM.augment_view(aug, texts[0], SEED, 0, 0) != DM.augment_view(aug, texts[0], SEED, 0, 1)


def test_augmentation_leaves_the_global_rngs_alone():
    require_nlpaug()
    aug = DM.augmenter()
    random.seed(11)
    np.random.seed(11)
    before = random.getstate(), np.random.get_state()
    DM.augment_view(aug, load_default_corpus(1)[0]["text"], SEED, 0, 0)
    after = random.getstate(), np.random.get_state()
    assert after[0] == before[0]
    assert all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
               for a, b in zip(after[1], before[1]))


def _release_compute_dime(mf, views_a, views_b, seed):
    """``normalize`` over all N rows of each view, then ``compute_dime`` one
    layer at a time from a reseeded global RNG: {normalisation: profile}."""
    hs = torch.stack([torch.stack([mf.normalize(a), mf.normalize(b)], dim=1)
                      for a, b in zip(views_a, views_b)])          # L x N x 2 x D
    out = {"raw": [], "maxEntropy": [], "logN": []}
    for l in range(hs.shape[0]):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            r = mf.compute_dime(hs[l:l + 1], alpha=1, normalizations=list(out))
        for k in out:
            out[k].extend(r[k])
    return out


@pytest.mark.parametrize("n,d", [(60, 96), (150, 24)], ids=["gram_branch", "covariance_branch"])
def test_compute_dime_matches_the_release(release, n, d):
    """N <= D: the release's Gram branch equals the headline profile. N > D: its
    covariance branch equals _meta_dime_release_profile, and the headline does not."""
    g = torch.Generator().manual_seed(d)
    w = torch.randn(8, d, generator=g, dtype=torch.float64)
    base = [torch.randn(n, 8, generator=g, dtype=torch.float64) @ w for _ in range(3)]
    views_a = [x + 0.3 * torch.randn(n, d, generator=g, dtype=torch.float64) for x in base]
    views_b = [x + 0.3 * torch.randn(n, d, generator=g, dtype=torch.float64) for x in base]
    for a in views_a:
        torch.testing.assert_close(DM.unit_rows(a), release.mf.normalize(a), rtol=0, atol=1e-15)
    gram, rel, branch = DM.dime_profiles(views_a, views_b, 10, SEED)
    ref = _release_compute_dime(release.mf, views_a, views_b, SEED)
    norm = DM.max_entropy(n, d)
    assert ref["maxEntropy"] == pytest.approx([x / norm for x in ref["raw"]], rel=1e-15)
    if n <= d:
        assert branch == "gram" and rel == gram
        assert gram == pytest.approx(ref["raw"], rel=1e-10, abs=1e-12)
    else:
        assert branch == "covariance"
        assert rel == pytest.approx(ref["raw"], rel=1e-10, abs=1e-12)
        assert max(abs(x - y) for x, y in zip(gram, rel)) > 1e-3


def _release_views(release, model, tok, texts, keep, max_length, seed):
    """The release's text path per view: text_augmentation (under BLME's
    seeds), tokenizer(truncation=True, max_length), a batch of one with its
    attention mask, and _get_pooled_hidden_states(method='mean') on every
    hidden state, in float64 as BLME casts them."""
    views = [None] * DM.N_VIEWS
    for i in keep:
        for v in range(DM.N_VIEWS):
            with DM.seeded_global_rngs(*DM.view_seeds(seed, i, v)):
                text = ast.literal_eval(release.text_augmentation([texts[i]])[0])[0]
            enc = tok(text, truncation=True, max_length=max_length, return_tensors="pt")
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
            if views[v] is None:
                views[v] = [[] for _ in hs]
            for l, h in enumerate(hs):
                views[v][l].append(release.pool(None, h.double(), enc["attention_mask"], method="mean")[0])
    return [[torch.stack(rows) for rows in view] for view in views]


@pytest.mark.parametrize("n_passages", [24, 48], ids=["gram_branch", "covariance_branch"])
@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_evaluate_matches_the_release(release, arch, n_passages):
    require_nlpaug()
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=3)
    tok = fd.tokenizer()
    data = load_default_corpus(n_passages)
    texts = [d["text"] for d in data]
    cfg = {"num_samples": n_passages, "max_length": 64, "augment_seed": SEED, "perm_seed": SEED}
    r = DM.DiMETask(cfg).evaluate(model, tok, data)
    keep = [i for i, t in enumerate(texts) if len(tok(t, add_special_tokens=False)["input_ids"]) >= 30]
    views_a, views_b = _release_views(release, model, tok, texts, keep, 64, SEED)
    n, d = views_a[0].shape
    assert r["_meta_n_prompts"] == n == len(keep) and r["_meta_hidden_size"] == d
    assert r["_meta_n_views_augmented"] == 2 * n
    assert len(r["dime_profile"]) == model.config.num_hidden_layers + 1
    ref = _release_compute_dime(release.mf, views_a, views_b, SEED)
    if n <= d:
        assert r["_meta_dime_release_branch"] == "gram"
        assert r["dime_profile"] == pytest.approx(ref["raw"], rel=1e-9, abs=1e-12)
    else:
        assert r["_meta_dime_release_branch"] == "covariance"
        # The release's Gram branch, which compute_dime skips when N > D.
        kx = [release.mf.normalize(a) for a in views_a]
        ky = [release.mf.normalize(b) for b in views_b]
        gram = [_release_doe(release.mf.dent, x @ x.T, y @ y.T, SEED) for x, y in zip(kx, ky)]
        assert r["dime_profile"] == pytest.approx(gram, rel=1e-9, abs=1e-12)
    assert r["_meta_dime_release_profile"] == pytest.approx(ref["maxEntropy"], rel=1e-9, abs=1e-12)
    assert r["_meta_max_entropy"] == pytest.approx(min(math.log(n), math.log(d)), rel=1e-15)
