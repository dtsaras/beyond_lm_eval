"""Parity: interpretability_logit_lens_kl vs the RELEASED code of Hu, Lepori &
Franke (arXiv:2504.14107), jennhu/model-human-processing@69e405c.

Run from the clone:

* ``src/model.py`` ``LM.apply_lens`` (``use_tuned_lens=False``):
  ``lm_head(layer_norm(hiddens)).log_softmax(-1)``;
* ``src/utils.py`` ``get_rank``: 1 + the number of tokens scoring above a token;
* ``analysis/notebooks/0_process_lm_data.ipynb``, the metric cell, executed:
  ``get_auc`` (the entropy AUC is ``vals.sum()``) and ``layer_stay_pos``.

Transcribed, because they cannot run here:

* the nnsight trace of ``LM.logprobs_and_logit_diffs_all_layers``,
  ``hiddens_l = [layer.output[0][0, :].unsqueeze(1) for layer in self.layers]``
  (nnsight is not installed, and under transformers 5 a Llama block returns a
  bare tensor, so ``output[0][0, :]`` would take one row): forward hooks take
  each block's hidden-state output;
* the entropy line of ``conditional_score_all_layers``,
  ``(-lp.exp() * lp).sum(dim=1)``, and the time-to-decision line of
  ``compute_all_metrics``, ``(layer_stay_pos(...) + 1) / n_layers``.

``model.py`` imports nnsight and tuned_lens at module level for code not
exercised here; both are stubbed for the import. The clone is
``~/.cache/blme-refrepos/model-human-processing`` (BLME_REFREPOS overrides the
root). Skips when absent (fails under BLME_REQUIRE_REFS=1).

The free-text adaptation is applied to the released functions' outputs: the
final top-1 y* is the "correct" answer, and log p(y*) - max_{v != y*} log p(v)
plays first_logprob_diff. KL is not in the release; it is computed here from the
released log-probabilities.
"""
import importlib.util
import json
import os
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
from blme.tasks.common import get_final_norm, get_layers, get_lm_head  # noqa: E402
from blme.tasks.interpretability import logit_lens_kl as LK  # noqa: E402

inv = fd.inv
_ROOT = os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))
_REPO = os.path.join(_ROOT, "model-human-processing")
_NB = os.path.join("analysis", "notebooks", "0_process_lm_data.ipynb")


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
def release():
    if not os.path.isfile(os.path.join(_REPO, "src", "model.py")):
        _missing(_REPO)
    saved = {k: sys.modules.get(k) for k in ("utils", "nnsight", "tuned_lens")}
    try:
        utils = _load("utils", os.path.join(_REPO, "src", "utils.py"))
        sys.modules["utils"] = utils
        for name, attr in (("nnsight", "LanguageModel"), ("tuned_lens", "TunedLens")):
            stub = types.ModuleType(name)
            setattr(stub, attr, None)
            sys.modules[name] = stub
        model_py = _load("mhp_model", os.path.join(_REPO, "src", "model.py"))
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    cells = json.loads(Path(_REPO, _NB).read_text())["cells"]
    src = ["".join(c["source"]) for c in cells
           if c["cell_type"] == "code" and "def layer_stay_pos" in "".join(c["source"])]
    assert len(src) == 1
    nb = {"np": np}
    exec(compile(src[0], _NB, "exec"), nb)
    return types.SimpleNamespace(LM=model_py.LM, get_rank=utils.get_rank,
                                 get_auc=nb["get_auc"], layer_stay_pos=nb["layer_stay_pos"])


def test_sources_pin_the_cloned_commit():
    if not os.path.isdir(_REPO):
        _missing(_REPO)
    head = subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert ("@" + head) in LK.LogitLensKLTask.SOURCES["code"]


def _hiddens(model, enc):
    """The trace of LM.logprobs_and_logit_diffs_all_layers: block outputs, (T, L, D)."""
    caught = []

    def hook(module, inputs, output):
        caught.append(output[0] if isinstance(output, tuple) else output)

    handles = [layer.register_forward_hook(hook) for layer in get_layers(model)]
    try:
        with torch.no_grad():
            out = model(**enc)
    finally:
        for h in handles:
            h.remove()
    return torch.cat([h[0, :].unsqueeze(1) for h in caught], dim=1), out.logits[0]


def _released_logprobs(release, model, hiddens):
    lens = types.SimpleNamespace(use_tuned_lens=False, lm_head=get_lm_head(model),
                                 layer_norm=get_final_norm(model))
    with torch.no_grad():
        logits, logprobs = release.LM.apply_lens(lens, hiddens)
    return logits, logprobs


def _reference(release, model, tok, texts):
    """Per-position quantities from the released decode and metric functions,
    position 0 dropped, pooled over the texts in order."""
    acc = {k: [] for k in ("kl_final", "kl_rev", "kl_next", "entropy", "agree", "ttd", "first")}
    for text in texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
        hid, _ = _hiddens(model, enc)
        _, lp32 = _released_logprobs(release, model, hid)       # (T, L, V), float32
        L = lp32.shape[1]
        lp32 = lp32[1:]
        lp = lp32.double()
        p = lp.exp()
        fin = lp[:, -1:, :]
        acc["kl_final"].append((p * (lp - fin)).sum(-1))
        acc["kl_rev"].append((fin.exp() * (fin - lp)).sum(-1))
        acc["kl_next"].append((p[:, :-1] * (lp[:, :-1] - lp[:, 1:])).sum(-1))
        acc["entropy"].append((-lp32.exp() * lp32).sum(-1).double())
        top = lp[:, -1].argmax(-1)
        for t in range(lp.shape[0]):
            rank = release.get_rank(lp32[t], int(top[t]))      # one rank per layer
            acc["agree"].append((rank == 1).double())
            others = lp[t].clone()
            others[:, top[t]] = -float("inf")
            margin = (lp[t, :, top[t]] - others.max(-1).values).numpy()
            acc["ttd"].append((release.layer_stay_pos(margin) + 1) / L)
            acc["first"].append((int(np.argmax((rank == 1).numpy())) + 1) / L)
    cat = {k: torch.cat(v).numpy() for k, v in acc.items()
           if k in ("kl_final", "kl_rev", "kl_next", "entropy")}
    cat["agree"] = torch.stack(acc["agree"]).numpy()
    return cat, np.asarray(acc["ttd"]), np.asarray(acc["first"]), L


@pytest.fixture(scope="module")
def corpus():
    from blme.cache import load_default_corpus
    return [d["text"] for d in load_default_corpus(16)]


def _models():
    """Tiny models with a non-trivial final norm: at its random-init weights
    (ones, zero bias) norming twice changes the logits by eps only."""
    out = {"gpt2": fd.tiny_gpt2(), "llama": fd.tiny_llama(n_layers=4)}
    g = torch.Generator().manual_seed(0)
    for m in out.values():
        n = get_final_norm(m)
        with torch.no_grad():
            n.weight.copy_(1 + 0.5 * torch.randn(n.weight.shape, generator=g))
            if getattr(n, "bias", None) is not None:
                n.bias.copy_(0.2 * torch.randn(n.bias.shape, generator=g))
    return out


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_decode_matches_released_apply_lens(release, arch):
    """BLME's lens on each block output == the released apply_lens, and its last
    layer (the hooked pre-norm output, normed once) == the model's logits."""
    model = _models()[arch]
    tok = fd.tokenizer()
    enc = tok(fd.TEXTS[2], return_tensors="pt")
    hid, model_logits = _hiddens(model, enc)
    ref_logits, ref_lp = _released_logprobs(release, model, hid)
    norm = get_final_norm(model)
    dz = dlp = 0.0
    with torch.no_grad():
        for l in range(hid.shape[1]):
            z = LK._lens_logits(model, norm, hid[:, l])
            # The release decodes all layers in one batched matmul.
            dz = max(dz, (z - ref_logits[:, l]).abs().max().item())
            lp = torch.log_softmax(z.double(), dim=-1)
            dlp = max(dlp, (lp - ref_lp[:, l].double()).abs().max().item())
    d_out = (ref_logits[:, -1] - model_logits).abs().max().item()
    # Mutant: HF's last hidden state is already post-norm; norming it again.
    with torch.no_grad():
        post = model(**enc, output_hidden_states=True).hidden_states[-1][0]
        d_double = (LK._lens_logits(model, norm, post) - model_logits).abs().max().item()
    print(f"\n[{arch}] logits {dz:.2e}, log-probs {dlp:.2e}, release last layer vs model "
          f"{d_out:.2e}, double-norm mutant {d_double:.2e}")
    assert dz < 1e-5 and dlp < 1e-6 and d_out < 1e-5
    assert d_double > 1e-2


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_evaluate_matches_the_released_pipeline(release, corpus, arch):
    model = _models()[arch]
    tok = fd.tokenizer()
    got = inv.run_task_on_model("interpretability_logit_lens_kl", model, tok, corpus,
                                use_cache=False)
    assert got["_meta_sink_n_massive_dropped"] == 0     # position 0 is the only sink here
    ref, ttd, first, L = _reference(release, model, tok, corpus)
    assert got["_meta_n_layers"] == L and got["_meta_n_tokens"] == ref["kl_final"].shape[0]

    diffs = {}

    def close(a, b, tol, name=None):
        d = float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b))))
        diffs[name or len(diffs)] = d
        assert d <= tol, (name, d)
        return d

    kl_prof = ref["kl_final"].mean(0)
    ent_prof = ref["entropy"].mean(0)
    # BLME's log-softmax is float64, the release's float32 (row offset ~1 ulp).
    close(got["kl_to_final_profile"], kl_prof, 1e-7, "kl_to_final_profile")
    close(got["_meta_kl_final_to_layer_profile"], ref["kl_rev"].mean(0), 1e-7, "kl_rev")
    close(got["kl_next_profile"], ref["kl_next"].mean(0), 1e-7, "kl_next_profile")
    close(got["kl_churn"], ref["kl_next"].mean(0).sum(), 1e-7, "kl_churn")
    # get_auc: the released AUC is the sum over layers (its entropy branch);
    # BLME's normalized twin divides by n_layers.
    close(got["_meta_kl_to_final_auc_sum"], release.get_auc(kl_prof, "entropy"), 1e-7, "auc_sum")
    close(got["kl_to_final_auc"], release.get_auc(kl_prof, "entropy") / L, 1e-7, "kl_auc")
    close(got["entropy_profile"], ent_prof, 1e-5, "entropy_profile")
    close(got["_meta_entropy_auc_sum"], release.get_auc(ent_prof, "entropy"), 1e-4, "ent_sum")
    close(got["entropy_auc"], release.get_auc(ent_prof, "entropy") / L, 1e-5, "entropy_auc")
    print(f"\n[{arch}] max |BLME - release|:",
          ", ".join(f"{k} {v:.2e}" for k, v in diffs.items()))
    # Decisions are compared exactly: rank-1 agreement, layer_stay_pos, first agreement.
    assert got["top1_agreement_profile"] == pytest.approx(list(ref["agree"].mean(0)), abs=1e-15)
    assert got["decision_depth"] == pytest.approx(float(ttd.mean()), abs=1e-15)
    assert got["first_agreement_depth"] == pytest.approx(float(first.mean()), abs=1e-15)
    assert got["_meta_final_lens_max_abs_logit_diff"] <= 1e-5


def _tiny_gemma2(cap=0.5):
    from transformers import Gemma2Config, Gemma2ForCausalLM
    torch.manual_seed(0)
    cfg = Gemma2Config(vocab_size=1000, hidden_size=32, intermediate_size=64,
                       num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                       head_dim=8, max_position_embeddings=512, final_logit_softcapping=cap,
                       attn_implementation="eager")
    return Gemma2ForCausalLM(cfg).eval()


def test_final_softcap_is_applied_where_the_release_omits_it(release, corpus):
    """Departure, shown: with Gemma-2's final logit softcap the released
    lm_head(norm(h)) is not the model's output distribution; BLME's last-layer
    lens is, so KL(p_L || p_L) = 0 at the model's own output."""
    model = _tiny_gemma2()
    tok = fd.tokenizer()
    enc = tok(corpus[0], return_tensors="pt")
    hid, model_logits = _hiddens(model, enc)
    ref_logits, _ = _released_logprobs(release, model, hid)
    with torch.no_grad():
        ours = LK._lens_logits(model, get_final_norm(model), hid[:, -1])
    assert (ours - model_logits).abs().max().item() < 1e-6
    assert (ref_logits[:, -1] - model_logits).abs().max().item() > 0.1
    got = inv.run_task_on_model("interpretability_logit_lens_kl", model, tok, corpus[:4],
                                use_cache=False)
    assert got["_meta_final_lens_max_abs_logit_diff"] < 1e-6
    assert got["kl_to_final_profile"][-1] == 0.0
