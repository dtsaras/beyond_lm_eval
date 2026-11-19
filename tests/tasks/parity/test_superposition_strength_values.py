"""geometry_superposition_strength: the six headline keys through evaluate().

The unembedding is c * S + mu: a simplex ETF S (m + 1 unit rows in R^m, every
pairwise cosine -1/m) at scale c, shifted by a common offset mu. The raw mean
squared overlap is computed by the released liuyz0/SuperpositionScaling
LLMs/overlap-0.py, whose pair loop is executed verbatim on the same matrix.
Centring removes mu exactly (the ETF rows sum to zero), so the centred
overlap is the Welch bound and its ratio is 1. The row-norm keys follow from
the row norms of c * S + mu. On pythia-70m-deduped (27 padding rows past
the tokenizer) the release's loop runs on the real head: the headline on the
emittable rows, ``_meta_mean_squared_overlap_all_rows`` on every row. Skips
without the clone (fails under BLME_REQUIRE_REFS=1).
"""
import ast
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from blme.tasks.geometry import superposition_strength as SS
from blme.tasks.geometry.superposition_strength import SuperpositionStrengthTask

_SCRIPT = Path(os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))) \
    / "SuperpositionScaling" / "LLMs" / "overlap-0.py"
M = 12


def _missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


def _official_mean_sq_overlap(W):
    """overlap-0.py's batched pair loop over the n(n-1)/2 unordered pairs."""
    if not _SCRIPT.is_file():
        _missing(_SCRIPT)
    loops = [n for n in ast.walk(ast.parse(_SCRIPT.read_text()))
             if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "i"]
    assert len(loops) == 1
    ns = {"torch": torch, "lm_head_weight": W, "batch_size": 8192,
          "result": {"n": W.size(0), "m": W.size(1)}, "overlap_sum": 0.0, "overlap_sq_sum": 0.0}
    exec(compile(ast.Module(body=loops, type_ignores=[]), str(_SCRIPT), "exec"), ns)
    return ns["overlap_sq_sum"]


def _simplex(m):
    V = m + 1
    E = np.eye(V) - 1.0 / V
    Q, _ = np.linalg.qr(E.T)
    S = E @ Q[:, :m]
    return S / np.linalg.norm(S, axis=1, keepdims=True)


class _Head(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.lm_head = nn.Linear(W.shape[1], W.shape[0], bias=False)
        self.lm_head.weight.data = W


def test_headline_keys_match_the_release_and_the_welch_bound():
    S = _simplex(M)
    assert np.allclose((S @ S.T)[~np.eye(M + 1, dtype=bool)], -1.0 / M, atol=1e-12)
    # The offset nearly cancels vertex 0, so the row norms spread across the
    # half-max cut.
    mu = -1.8 * S[0] + 0.3 * np.random.default_rng(0).standard_normal(M)
    W = torch.tensor(2.0 * S + mu, dtype=torch.float64).float()
    res = SuperpositionStrengthTask(config={}).evaluate(_Head(W), None, dataset=None)

    V = M + 1
    welch = (V - M) / (M * (V - 1))
    raw = _official_mean_sq_overlap(W.double())
    assert raw > 10 * welch
    norms = np.linalg.norm(W.double().numpy(), axis=1)
    want = {
        "mean_squared_overlap": raw,
        "superposition_ratio": raw / welch,
        "mean_squared_overlap_centered": welch,
        "superposition_ratio_centered": 1.0,
        "row_norm_cv": norms.std(ddof=1) / norms.mean(),
        "frac_norm_above_half_max": float(np.mean(norms > 0.5 * norms.max())),
    }
    assert 0.0 < want["frac_norm_above_half_max"] < 1.0
    for key, v in want.items():
        assert res[key] == pytest.approx(v, rel=1e-5), key


def test_sources_pin_the_cloned_commit():
    repo = _SCRIPT.parents[1]
    if not repo.is_dir():
        _missing(repo)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert ("@" + head) in SS.SuperpositionStrengthTask.SOURCES["code"]


def test_pythia_head_matches_the_release_on_emittable_and_all_rows():
    """The release reads every row of embed_out in float32 (CUDA there, CPU
    here); BLME accumulates the Gram in float64, hence rel 1e-5."""
    transformers = pytest.importorskip("transformers")
    name = "EleutherAI/pythia-70m-deduped"
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name)
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"{name} not cached: {e}")
    res = SuperpositionStrengthTask(config={}).evaluate(model, tok, dataset=None)
    W = model.get_output_embeddings().weight.detach().float()
    n_tok = len(tok)
    assert res["_meta_n_unused_rows"] == W.shape[0] - n_tok > 0
    with torch.no_grad():
        emittable = _official_mean_sq_overlap(W[:n_tok])
        every_row = _official_mean_sq_overlap(W)
    assert res["mean_squared_overlap"] == pytest.approx(emittable, rel=1e-5)
    assert res["_meta_mean_squared_overlap_all_rows"] == pytest.approx(every_row, rel=1e-5)
    assert abs(emittable - every_row) > 1e-4 * emittable


def test_neox_masked_vocabulary_extends_the_active_head():
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from tests.invariance.model_transforms import pad_vocabulary

    config = GPTNeoXConfig(vocab_size=48, hidden_size=32, intermediate_size=64,
                           num_attention_heads=4, num_hidden_layers=2, tie_word_embeddings=False)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(1)
        original = GPTNeoXForCausalLM(config).eval()
        padded = pad_vocabulary(original, extra_rows=3, kind="masked")
        ids = torch.tensor([[4, 8, 5]])
        expected = original(ids).logits
        obtained = padded(ids).logits
    assert expected.shape[-1] == 48
    assert obtained.shape[-1] == padded.get_output_embeddings().out_features == 51
    torch.testing.assert_close(obtained[..., :48], expected, rtol=0, atol=0)
    assert torch.isneginf(obtained[..., 48:]).all()
