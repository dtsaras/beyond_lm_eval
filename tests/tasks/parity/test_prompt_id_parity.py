"""Parity: geometry_prompt_id vs the RELEASED pipeline of Viswanathan et al.
(RitAreaSciencePark/token_geometry@0214897, ``src/extract_id.py``):
``extract_hidden_states`` (no special tokens, first ``max_length`` tokens,
HF hidden states, ``torch.cdist`` distances, the HF loss) and the GRIDE
branch of ``compute_id_across_estimators`` (np.unique over distance rows,
``Data(distances=..., maxk=100).return_id_scaling_gride(range_max=64)``),
read at ``[..., 0, 1]`` (n2 = 4) and summarised as ``plots_for_paper.ipynb``
does (mean over prompts; ``stats.pearsonr(np.log(ids[:, l]), loss)``).

The release's functions are executed from the clone's own source (module-level
code skipped: it disables autograd globally and needs skdim and datasets).
DADApy is side-installed at ~/.cache/blme-sideinstalls/dadapy-d729b88. Skips
when either is absent (fails under BLME_REQUIRE_REFS=1).
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
from blme.cache import load_long_documents  # noqa: E402
from blme.tasks.geometry import gride as G  # noqa: E402
from blme.tasks.geometry import prompt_id as P  # noqa: E402

_ROOT = os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))
_REPO = os.path.join(_ROOT, "token_geometry")
_SIDE = os.path.join(os.environ.get("BLME_SIDEINSTALLS", os.path.expanduser("~/.cache/blme-sideinstalls")),
                     "dadapy-d729b88")
_RELEASED = ("convert_to_tensor", "convert_to_distances", "extract_hidden_states",
             "get_knn_from_torch_distance_matrix", "compute_id_across_estimators")


def _missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


@pytest.fixture(scope="module")
def release():
    path = os.path.join(_REPO, "src", "extract_id.py")
    if not os.path.isfile(path):
        _missing(path)
    if not os.path.isdir(os.path.join(_SIDE, "dadapy")):
        _missing(_SIDE)
    sys.path.insert(0, _SIDE)
    try:
        from dadapy import data
    finally:
        sys.path.remove(_SIDE)
    tree = ast.parse(open(path).read())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in _RELEASED]
    assert {n.name for n in body} == set(_RELEASED)
    ns = {"np": np, "torch": torch, "data": data}
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), ns)
    return ns


def test_sources_pin_the_cloned_commit():
    if not os.path.isdir(_REPO):
        _missing(_REPO)
    head = subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert ("@" + head) in P.PromptIntrinsicDimTask.SOURCES["code"]


def test_unique_rows_fast_path_is_np_unique():
    """The first-column shortcut returns np.unique(X, axis=0) bit for bit,
    with ties, exact duplicates and signed zeros sent to np.unique."""
    rng = np.random.default_rng(0)
    for trial in range(120):
        n, d = int(rng.integers(2, 200)), int(rng.integers(1, 40))
        X = rng.standard_normal((n, d))
        if trial % 3 == 0:
            X = np.round(X, 1)
        if trial % 5 == 0:
            X[int(rng.integers(0, n))] = X[0]
        if trial % 7 == 0:
            X[:2, 0] = [0.0, -0.0]
        ref, got = np.unique(X, axis=0), G._unique_rows(X)
        assert got.shape == ref.shape and got.tobytes() == ref.tobytes(), trial


def _release_gride(release, H):
    """The release's GRIDE branch on a cloud, with float64 cdist distances."""
    Ht = torch.as_tensor(np.asarray(H, dtype=np.float64))
    return release["compute_id_across_estimators"](Ht, torch.cdist(Ht, Ht), "GRIDE")


def _manifold(n, intrinsic, ambient, seed):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, intrinsic))
    A = rng.standard_normal((intrinsic, ambient)) / np.sqrt(intrinsic)
    return np.tanh(Z @ A) + 1e-3 * rng.standard_normal((n, ambient))


@pytest.mark.parametrize("n,intrinsic,ambient", [(1024, 6, 64), (1024, 12, 768), (300, 3, 32)])
def test_per_prompt_gride_matches_the_release_on_synthetic_clouds(release, n, intrinsic, ambient):
    X = _manifold(n, intrinsic, ambient, seed=n + intrinsic)
    X[5] = X[17]                                  # a repeated token, removed by both
    ref = _release_gride(release, X)
    ids, ranks = P.prompt_scaling(X)
    assert ref.shape == (3, 6)
    assert list(ranks) == [2, 4, 8, 16, 32, 64]
    np.testing.assert_allclose(ids, ref[0], rtol=0, atol=1e-9)


def _states(model, tok, text, max_length):
    ids = tok(text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), output_hidden_states=True)
    return [h[0].float().numpy() for h in out.hidden_states[1:]]


def _gpt2():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"gpt2 not cached: {e}")
    return model, tok


@pytest.mark.parametrize("arch,max_length", [("tiny-gpt2", 256), ("tiny-llama", 256), ("gpt2", 1024)])
def test_per_prompt_gride_matches_the_release_on_real_clouds(release, arch, max_length):
    """Token clouds of one article at every hidden state: the release's GRIDE
    on float64 distances equals BLME's; with the release's own float32 cdist
    the estimates move by float32 rounding only."""
    if arch == "gpt2":
        model, tok = _gpt2()
    else:
        model = fd.tiny_gpt2() if arch == "tiny-gpt2" else fd.tiny_llama(n_layers=3)
        tok = fd.tokenizer()
    text = load_long_documents(1, split="validation")[0]["text"]
    worst32 = 0.0
    for X in _states(model, tok, text, max_length):
        assert X.shape[0] == max_length
        ids, _ = P.prompt_scaling(X)
        np.testing.assert_allclose(ids, _release_gride(release, X)[0], rtol=0, atol=1e-9)
        Xt = torch.from_numpy(X)
        ref32 = release["compute_id_across_estimators"](Xt, torch.cdist(Xt, Xt), "GRIDE")[0]
        worst32 = max(worst32, float(np.max(np.abs(ref32 - ids) / ids)))
    assert worst32 < 1e-3


def _release_pipeline(release, model, tok, texts, max_length):
    """extract_id.py's structured loop, then the notebook's summaries."""
    from scipy import stats
    gride, gride64, loss = [], [], []
    for text in texts:
        rep = release["extract_hidden_states"](text, model, tok, max_length=max_length)
        hs, hd = rep["hidden_states"][1:, 0], rep["hidden_distances"][1:]
        gride.append([release["compute_id_across_estimators"](h, d, "GRIDE") for h, d in zip(hs, hd)])
        gride64.append([_release_gride(release, h.numpy()) for h in hs])
        loss.append(float(rep["loss"]))
    out = {"loss": np.array(loss)}
    for name, arr in (("fp32", np.array(gride)), ("fp64", np.array(gride64))):
        ids = arr[:, :, 0, 1]
        out[name] = {
            "ids": ids,
            "profile": ids.mean(axis=0),
            "pearson": [stats.pearsonr(np.log(ids[:, l]), loss).statistic for l in range(ids.shape[1])],
            "scaling": arr[:, :, 0, :].mean(axis=0),
        }
    return out


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_evaluate_matches_the_release_pipeline(release, arch):
    """evaluate() on tiny-random-gpt2 / a tiny Llama (512 positions, so 128
    tokens per prompt) against the release's loop on the same articles. The
    hidden states are the release's bit for bit; the two departures are float64
    distances (vs cdist in float32) and a float64 mean of the float32 token
    losses (vs HF's float32 reduction, 1.3e-7 apart). A random model's losses
    barely vary across prompts (std 0.01 nats), which magnifies the second in
    the Pearson r to 4e-5, so the r is also checked on BLME's own surprisal."""
    from scipy import stats
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=3)
    tok = fd.tokenizer()
    cfg = {"num_samples": 6, "max_length": 128, "split": "validation"}
    r = P.PromptIntrinsicDimTask(cfg).evaluate(model, tok, None)
    texts = [d["text"] for d in load_long_documents(24, split="validation")
             if len(d["text"].encode("utf-8")) >= 8192][:6]
    ref = _release_pipeline(release, model, tok, texts, max_length=128)
    assert r["_meta_n_prompts"] == 6 and r["_meta_n_prompts_too_short"] == 0
    assert r["_meta_n_layers"] == len(ref["fp64"]["profile"])
    np.testing.assert_allclose(r["_meta_prompt_surprisal"], ref["loss"], rtol=1e-6)
    # The release's estimator on float64 distances: equal to bisection precision.
    np.testing.assert_allclose(r["prompt_id_profile"], ref["fp64"]["profile"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(r["_meta_prompt_id_scaling_mean"], ref["fp64"]["scaling"],
                               rtol=0, atol=1e-9)
    own = [stats.pearsonr(np.log(ref["fp64"]["ids"][:, l]), r["_meta_prompt_surprisal"]).statistic
           for l in range(r["_meta_n_layers"])]
    np.testing.assert_allclose(r["prompt_log_id_surprisal_pearson"], own, rtol=0, atol=1e-12)
    assert r["prompt_id_peak"] == pytest.approx(max(ref["fp64"]["profile"]), abs=1e-9)
    assert r["prompt_id_last_layer"] == pytest.approx(ref["fp64"]["profile"][-1], abs=1e-9)
    # The release as shipped (float32 cdist, float32 loss).
    np.testing.assert_allclose(r["prompt_id_profile"], ref["fp32"]["profile"], rtol=1e-4)
    np.testing.assert_allclose(r["prompt_log_id_surprisal_pearson"], ref["fp32"]["pearson"], atol=1e-3)


def test_gpt2_log_id_tracks_surprisal():
    """Known positive: within a model, prompts with a higher mean surprisal
    have a higher-dimensional token cloud at every layer (the paper's
    rho(log ID, surprisal) > 0; the release's own gpt2 arrays on Pile-10K,
    results/Pile-Structured/gpt2, give r = 0.49-0.54 at every layer). The
    profile rises from ~3.6 at hidden state 1 to an interior peak, as there
    (3.6 -> 7.1 at hidden state 9 -> 6.3). 48 prompts (111 s on 8 CPU threads)
    give r = 0.32-0.45; the default 256 give 0.48-0.56."""
    model, tok = _gpt2()
    r = P.PromptIntrinsicDimTask({"num_samples": 48}).evaluate(model, tok, None)
    assert r["_meta_n_prompts"] == 48 and r["_meta_prompt_tokens"] == 1024
    pearson = r["prompt_log_id_surprisal_pearson"]
    assert min(pearson) > 0.25 and np.mean(pearson) > 0.35
    assert max(r["_meta_prompt_log_id_surprisal_pvalue"]) < 0.05
    prof = r["prompt_id_profile"]
    assert 1 < r["_meta_prompt_id_peak_layer"] < len(prof)
    assert prof[0] == pytest.approx(3.5, abs=0.5) and r["prompt_id_peak"] > 1.8 * prof[0]
