"""Parity: geometry_predictive_information vs the RELEASED InfoRidge
(chili-lab/InfoRidge@2cde643): ``mutual_information/utils.py`` and
``analysis.analyze_batch``, loaded from the clone, fed the readout of
``embedding.compute_embeddings`` (right-padded context and context+target
batches). Skips when the clone is absent (fails under BLME_REQUIRE_REFS=1).
"""
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "invariance"))

import _family_d as fd  # noqa: E402
from blme.cache import load_default_corpus  # noqa: E402
from blme.tasks.geometry import predictive_information as PI  # noqa: E402

_ROOT = os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))
_REPO = os.path.join(_ROOT, "InfoRidge")


def _missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ref():
    mi = os.path.join(_REPO, "mutual_information")
    if not os.path.isdir(mi):
        _missing(mi)
    utils = _load("_inforidge_utils", os.path.join(mi, "utils.py"))
    # analysis.py does `from utils import ...`.
    saved = sys.modules.get("utils")
    sys.modules["utils"] = utils
    try:
        analysis = _load("_inforidge_analysis", os.path.join(mi, "analysis.py"))
    finally:
        if saved is None:
            sys.modules.pop("utils")
        else:
            sys.modules["utils"] = saved
    return utils, analysis


def test_sources_pin_the_cloned_commit():
    if not os.path.isdir(_REPO):
        _missing(_REPO)
    head = subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert ("@" + head) in PI.PredictiveInformationTask.SOURCES["code"]


def test_estimator_matches_the_release(ref):
    utils, analysis = ref
    g = torch.Generator().manual_seed(0)
    y = torch.randn(300, 24, generator=g)
    z = torch.tanh(y @ torch.randn(24, 40, generator=g)) + 0.3 * torch.randn(300, 40, generator=g)
    gz_ref, hz_ref = analysis._compute_gram_and_entropy(z.unsqueeze(1), 1.0)
    gy_ref, hy_ref = analysis._compute_gram_and_entropy(y.unsqueeze(1), 1.0)
    i_ref = float(hz_ref + hy_ref - utils.compute_joint_entropy(gz_ref, gy_ref))
    gz, hz = PI.gram_and_entropy(z)
    gy, hy = PI.gram_and_entropy(y)
    torch.testing.assert_close(gz, gz_ref, rtol=1e-5, atol=1e-6)
    assert hz == pytest.approx(float(hz_ref), rel=1e-5)
    assert PI.mutual_information(gz, hz, gy, hy) == pytest.approx(i_ref, rel=1e-4)


def test_mi_orders_dependence():
    """Known positive: a rotation of Y (the same Gram) shares more information
    with Y than an independent cloud does, and both stay within [0, log N].
    The estimator is far from 0 on independent clouds (here 1.59 against 1.94
    nats): its kernel on unit rows is flat, so only differences at a fixed N
    and sigma are meaningful."""
    g = torch.Generator().manual_seed(1)
    y = torch.randn(400, 16, generator=g)
    q = torch.linalg.qr(torch.randn(16, 16, generator=g))[0]
    gy, hy = PI.gram_and_entropy(y)
    dep = PI.mutual_information(*PI.gram_and_entropy(y @ q), gy, hy)
    ind = PI.mutual_information(*PI.gram_and_entropy(torch.randn(400, 16, generator=g)), gy, hy)
    assert 0 <= ind < 0.9 * dep and dep <= math.log(400)


def _release_profile(analysis, model, tok, texts, cfg):
    """main.run_izy_for_dataset at token level: right-padded batches of the
    contexts and of context + target, then analyze_batch."""
    encoded = [tok(t, truncation=True, max_length=cfg["max_length"])["input_ids"] for t in texts]
    targets = PI.split_points([len(e) for e in encoded], cfg["split_seed"])
    ctx = [e[:t] for e, t in zip(encoded, targets) if t is not None]
    comb = [e[:t + 1] for e, t in zip(encoded, targets) if t is not None]

    def hidden(seqs):
        width = max(len(s) for s in seqs)
        ids = torch.tensor([s + [0] * (width - len(s)) for s in seqs])
        mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs])
        with torch.no_grad():
            return model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states

    res = analysis.analyze_batch(hidden(ctx), hidden(comb), [len(c) for c in ctx], sigma=1.0)
    keys = sorted(res["layer_results"], key=lambda k: int(k.split("_")[1]))
    return [float(res["layer_results"][k]["I_ZY"]) for k in keys], float(res["H_Y"])


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_evaluate_matches_the_release(ref, arch):
    _, analysis = ref
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=3)
    tok = fd.tokenizer()
    texts = [d["text"] for d in load_default_corpus(60)]
    cfg = {"num_samples": 60, "max_length": 64, "split_seed": 0}
    r = PI.PredictiveInformationTask(cfg).evaluate(model, tok, [{"text": t} for t in texts])
    profile, h_y = _release_profile(analysis, model, tok, texts, cfg)
    assert r["_meta_n_sequences"] == 60
    assert r["_meta_H_Y"] == pytest.approx(h_y, rel=1e-5)
    assert r["predictive_info_profile"] == pytest.approx(profile, rel=1e-3, abs=1e-5)


def test_gpt2_shows_the_ridge():
    """Known positive (Chang et al.): predictive information peaks at an
    intermediate layer and falls in the final ones. gpt2 at N = 1000: peak
    1.20 nats at layer 8 of 12, last layer 0.09."""
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"gpt2 not cached: {e}")
    r = PI.PredictiveInformationTask({"num_samples": 300}).evaluate(
        model, tok, load_default_corpus(300))
    profile = r["predictive_info_profile"]
    assert 2 <= r["_meta_predictive_info_peak_layer"] <= len(profile) - 1
    assert profile[-1] < 0.5 * r["predictive_info_peak"]
    assert profile[0] < r["predictive_info_peak"]
