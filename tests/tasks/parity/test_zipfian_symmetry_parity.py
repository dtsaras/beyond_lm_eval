"""Parity: geometry_zipfian_symmetry vs the RELEASED zipfian-whitening code
(cl-tohoku/zipfian-whitening@fa44d29): ``calc_zipfian_isotropy_score`` and
``calc_uniform_isotropy_score`` from ``src/calc_isotropy_score.py``, compiled
from the clone's source (the module imports sentence-transformers, mteb and
fire at top level). Skips when the clone is absent (fails under
BLME_REQUIRE_REFS=1).
"""
import ast
import math
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
from blme.tasks.common import get_embeddings, get_lm_head  # noqa: E402
from blme.tasks.geometry import zipfian_symmetry as ZS  # noqa: E402

_ROOT = os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))
_REPO = os.path.join(_ROOT, "zipfian-whitening")


def _missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


class _TT:
    """Stand-in for torchtyping's TensorType in the signatures."""
    def __class_getitem__(cls, item):
        return None


@pytest.fixture(scope="module")
def ref():
    path = os.path.join(_REPO, "src", "calc_isotropy_score.py")
    if not os.path.isfile(path):
        _missing(path)
    names = {"calc_zipfian_isotropy_score", "calc_uniform_isotropy_score"}
    tree = ast.parse(open(path).read())
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {d.name for d in defs} == names
    ns = {"torch": torch, "TT": _TT, "Tuple": tuple}
    exec(compile(ast.Module(body=defs, type_ignores=[]), path, "exec"), ns)
    return ns


def test_sources_pin_the_cloned_commit():
    if not os.path.isdir(_REPO):
        _missing(_REPO)
    head = subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert ("@" + head) in ZS.ZipfianSymmetryTask.SOURCES["code"]


def _table(n=400, d=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(n, d, generator=g, dtype=torch.float64) * torch.linspace(0.2, 2.0, d, dtype=torch.float64)
    W += 0.8                                     # an off-centre table
    p = torch.distributions.Dirichlet(torch.full((n,), 0.3, dtype=torch.float64)).sample()
    return W, p


def test_scores_match_the_release(ref):
    W, p = _table()
    s1_ref, s2_ref = ref["calc_zipfian_isotropy_score"](W, p)
    u1_ref, u2_ref = ref["calc_uniform_isotropy_score"](W)
    s1, s2, _ = ZS.symmetry(W, p)
    u1, u2, _ = ZS.symmetry(W, np.full(W.shape[0], 1.0 / W.shape[0]))
    assert s1 == pytest.approx(float(s1_ref), rel=1e-12)
    # The release's log d is float32 (torch.log of an integer tensor).
    assert s2 == pytest.approx(float(s2_ref), rel=1e-7)
    assert u1 == pytest.approx(float(u1_ref), rel=1e-12)
    assert u2 == pytest.approx(float(u2_ref), rel=1e-7)


def test_invariances_and_anchors():
    """Sym1 and Sym2 are unchanged by rescaling and rotation; zero-padding the
    width keeps Sym1 and the undivided entropy but lowers Sym2 by log d / log d'.
    A table centred under p has Sym1 = 1; an isotropic one has Sym2 near 1."""
    W, p = _table(seed=1)
    s1, s2, H = ZS.symmetry(W, p)
    q = torch.linalg.qr(torch.randn(24, 24, generator=torch.Generator().manual_seed(2),
                                    dtype=torch.float64))[0]
    r1, r2, _ = ZS.symmetry(3.0 * W @ q, p)
    assert (r1, r2) == pytest.approx((s1, s2), rel=1e-10)
    z1, z2, zH = ZS.symmetry(torch.cat([W, torch.zeros(W.shape[0], 8, dtype=torch.float64)], 1), p)
    assert z1 == pytest.approx(s1, rel=1e-12) and zH == pytest.approx(H, rel=1e-10)
    assert z2 == pytest.approx(s2 * math.log(24) / math.log(32), rel=1e-10)
    centred = W - p @ W
    assert ZS.symmetry(centred, p)[0] == pytest.approx(1.0, abs=1e-12)
    iso = torch.randn(20000, 8, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    assert ZS.symmetry(iso, np.full(20000, 1 / 20000))[1] > 0.999


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_evaluate_matches_the_release(ref, arch):
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=2)
    tok = fd.tokenizer()
    data = load_default_corpus(50)
    r = ZS.ZipfianSymmetryTask({"use_dataset": True}).evaluate(model, tok, data)
    counts = np.zeros(1000, dtype=np.int64)
    for d in data:
        for i in tok(d["text"], add_special_tokens=False)["input_ids"]:
            counts[i] += 1
    keep = np.nonzero(counts)[0]
    p = torch.as_tensor(counts[keep] / counts[keep].sum())
    for name, table in (("embedding", get_embeddings(model)), ("unembedding", get_lm_head(model).weight)):
        W = table.detach().to(torch.float64)[torch.as_tensor(keep)]
        s1, s2 = ref["calc_zipfian_isotropy_score"](W, p)
        assert r[f"zipf_sym1_{name}"] == pytest.approx(float(s1), rel=1e-10)
        assert r[f"zipf_sym2_{name}"] == pytest.approx(float(s2), rel=1e-7)
    assert r["_meta_n_used_tokens"] == keep.size


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_headline_invariant_under_identity_blocks(arch):
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=2)
    tok = fd.tokenizer()
    data = load_default_corpus(20)
    ids = [tok(t, return_tensors="pt")["input_ids"] for t in fd.TEXTS[:3]]
    deep = fd.inv.insert_identity_blocks(model, [0, 1], verify_on=ids)
    base = ZS.ZipfianSymmetryTask({"use_dataset": True}).evaluate(model, tok, data)
    ins = ZS.ZipfianSymmetryTask({"use_dataset": True}).evaluate(deep, tok, data)
    fd.inv.assert_invariant(base, ins, sorted(ZS.ZipfianSymmetryTask.SIZE_INVARIANT), rtol=0.0)
