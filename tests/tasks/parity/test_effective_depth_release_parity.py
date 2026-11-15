"""Parity: geometry_effective_depth vs the RELEASED code it cites, through evaluate().

* Block influence: ZigZagLLMs' short-transformers (RitAreaSciencePark/ZigZagLLMs,
  pythia_version) wraps every GPT-NeoX layer's forward and keeps the running
  mean over examples of ``dist.bi_score`` (ShortGPT's 1 - E_t cos). It is run
  end to end on a tiny GPT-NeoX with its own ``analyse_layers``. BLME pools the
  tokens of every sample, the release averages per-example means, so the
  windows are truncated to one length, where the two agree.
* Contribution ratios: Csordas et al.'s ``analyze_norms.py`` (NDIF / nnsight
  module, not importable) is reached through ``ast``: its own ``layer_diff``
  assignment and ``relative_contribution_layer.append(...)`` expression run on
  hooked block inputs and outputs, then pooled as it pools them (sum over
  tokens / token count). ``effective_depth_90`` is BLME's summary of that
  profile (first k with cumulative share >= 0.9, over L).

Skips without the clones (fails under BLME_REQUIRE_REFS=1).
"""
import ast
import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "invariance"))

import _family_d as fd  # noqa: E402
from dynamics_fixtures import tiny_neox  # noqa: E402
from blme.tasks.geometry import effective_depth as ED  # noqa: E402

_ROOT = Path(os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos")))
_ZIGZAG = _ROOT / "ZigZagLLMs"
_ST_DIR = _ZIGZAG / "src" / "benchmark" / "reproducibility" / "pythia_version"
_CSORDAS = _ROOT / "llm_effective_depth"
_NORMS = _CSORDAS / "analysis" / "analyze_norms.py"
N_DOCS, WINDOW = 6, 24


def _missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


def _head(repo):
    if not repo.is_dir():
        _missing(repo)
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def test_sources_pin_the_cloned_commits():
    code = ED.EffectiveDepthTask.SOURCES["code"]
    assert ("robertcsordas/llm_effective_depth@" + _head(_CSORDAS)) in code
    pinned = code.split("ZigZagLLMs@")[1].split()[0]
    assert len(pinned) >= 7 and _head(_ZIGZAG).startswith(pinned)


@pytest.fixture(scope="module")
def short_transformers():
    if not (_ST_DIR / "short_transformers" / "dist.py").is_file():
        _missing(_ST_DIR)
    sys.path.insert(0, str(_ST_DIR))
    try:
        st = importlib.import_module("short_transformers")
        dist = importlib.import_module("short_transformers.dist")
        yield types.SimpleNamespace(ShortTransformer=st.ShortTransformer, bi_score=dist.bi_score)
    finally:
        sys.path.remove(str(_ST_DIR))
        for n in [n for n in sys.modules if n == "short_transformers" or n.startswith("short_transformers.")]:
            sys.modules.pop(n, None)


@pytest.fixture(scope="module")
def csordas_ratio():
    """``layer_diff = ...`` and the ``relative_contribution_layer`` expression
    of analyze_norms.py, as a function of one block's (input, output)."""
    if not _NORMS.is_file():
        _missing(_NORMS)
    tree = ast.parse(_NORMS.read_text())
    diff = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "layer_diff" for t in n.targets)]
    expr = [n.args[0] for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "append"
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "relative_contribution_layer"]
    assert len(diff) == 1 and len(expr) == 1
    diff_code = compile(ast.Module(body=diff, type_ignores=[]), str(_NORMS), "exec")
    expr_code = compile(ast.Expression(body=expr[0]), str(_NORMS), "eval")

    def ratio(h_in, h_out):
        ns = {"layer": types.SimpleNamespace(inputs=[[h_in[None]]], output=[h_out[None]])}
        exec(diff_code, ns)
        return eval(expr_code, ns)[0]
    return ratio


def _block_io(model, enc):
    """Every block's (input, output) hidden states for one window."""
    io = []

    def pre(module, args, kwargs):
        io.append([(args[0] if args else kwargs["hidden_states"])[0].detach()])

    def post(module, args, output):
        io[-1].append((output[0] if isinstance(output, tuple) else output)[0].detach())

    blocks = model.gpt_neox.layers
    hooks = [b.register_forward_pre_hook(pre, with_kwargs=True) for b in blocks]
    hooks += [b.register_forward_hook(post) for b in blocks]
    try:
        with torch.no_grad():
            model(**enc)
    finally:
        for h in hooks:
            h.remove()
    return io


@pytest.mark.parametrize("parallel", [True, False])
def test_evaluate_matches_short_transformers_and_csordas(short_transformers, csordas_ratio, parallel):
    tok = fd.tokenizer()
    docs = fd.dataset(N_DOCS)
    assert all(len(tok(d["text"])["input_ids"]) > WINDOW for d in docs)
    model = tiny_neox(n_layers=5, parallel=parallel)
    res = ED.EffectiveDepthTask({"num_samples": N_DOCS, "max_length": WINDOW}).evaluate(
        model, tok, docs)
    L = res["_meta_n_layers_profiled"]
    assert L == 5

    # Csordas: per-token ratios of every window, pooled over tokens.
    sums, cnt = np.zeros(L), 0
    for d in docs:
        enc = tok(d["text"], return_tensors="pt", truncation=True, max_length=WINDOW)
        io = _block_io(model, enc)
        assert len(io) == L
        sums += np.array([float(csordas_ratio(i, o).sum()) for i, o in io])
        cnt += enc["input_ids"].shape[1]
    contrib = sums / cnt
    np.testing.assert_allclose(res["contribution_ratio_per_layer"], contrib, rtol=1e-5)
    cum = np.cumsum(contrib) / contrib.sum()
    assert res["effective_depth_90"] == pytest.approx((int(np.argmax(cum >= 0.9)) + 1) / L)

    # short-transformers, end to end on the same model (it wraps the layers).
    st = short_transformers.ShortTransformer.from_model(model)
    st.set_metric(short_transformers.bi_score)
    table = st.analyse_layers(docs, tokenizer=tok, key="text", limit=N_DOCS, max_length=WINDOW)
    bi = table[0, :L]
    assert np.ptp(bi) > 1e-3
    np.testing.assert_allclose(res["block_influence_per_layer"], bi, rtol=1e-5, atol=1e-7)
    assert res["max_block_influence"] == pytest.approx(bi.max(), rel=1e-5)
    assert res["total_block_influence"] == pytest.approx(bi.sum(), rel=1e-5)
