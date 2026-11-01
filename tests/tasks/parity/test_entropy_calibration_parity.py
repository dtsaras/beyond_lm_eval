"""Parity: dynamics_entropy_calibration vs the RELEASED entropy-calibration
code (stevenxcao/entropy-calibration@d3f62e4).

``modeling.py`` imports vLLM at top level and hard-codes ``device='cuda'``,
so the test compiles the release's own ``ids_to_tensor``,
``HFModel.compute_logprobs`` and ``Tokenizer.encode_prompt`` from the clone's
source with the device set to CPU, plus ``load_datasets``'s WikiText
prompt/response split. Generation is not compared: the release samples with
vLLM. Skips when the clone is absent (fails under BLME_REQUIRE_REFS=1).
"""
import ast
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "invariance"))

import _family_d as fd  # noqa: E402
from blme.cache import load_long_documents  # noqa: E402
from blme.tasks.dynamics import entropy_calibration as EC  # noqa: E402

_ROOT = os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))
_REPO = os.path.join(_ROOT, "entropy-calibration")


def _missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


def _defs(path, names, methods=()):
    """Module-level functions ``names`` and ``(class, method)`` pairs from a file."""
    tree = ast.parse(open(path).read())
    out = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for cls, meth in methods:
        c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        out += [n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == meth]
    return out


@pytest.fixture(scope="module")
def ref():
    if not os.path.isdir(_REPO):
        _missing(_REPO)
    ns = {"torch": torch, "F": F, "np": np, "tqdm": lambda x: x,
          "GPTNeoXForCausalLM": transformers.GPTNeoXForCausalLM,
          "GemmaForCausalLM": transformers.GemmaForCausalLM}
    path = os.path.join(_REPO, "modeling.py")
    mod = ast.Module(body=_defs(path, {"ids_to_tensor"}, [("HFModel", "compute_logprobs"),
                                                           ("Tokenizer", "encode_prompt")]),
                     type_ignores=[])
    src = ast.unparse(mod).replace("device='cuda'", "device='cpu'")
    exec(compile(src, path, "exec"), ns)
    path = os.path.join(_REPO, "load_datasets.py")
    mod = ast.Module(body=_defs(path, {"get_prompt_ids_wikitext", "get_response_ids_wikitext",
                                       "get_prompt_and_response_ids"}), type_ignores=[])
    exec(compile(mod, path, "exec"), ns)
    return ns


def test_sources_pin_the_cloned_commit():
    if not os.path.isdir(_REPO):
        _missing(_REPO)
    head = subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert ("@" + head) in EC.EntropyCalibrationTask.SOURCES["code"]


def _release_tokenizer(ref, tok):
    t = types.SimpleNamespace(tokenizer=tok)
    t.encode_prompt = lambda prompt, apply_chat_template=False: ref["encode_prompt"](
        t, prompt, apply_chat_template)
    return t


def test_prompt_and_response_match_the_release(ref):
    tok = fd.tokenizer()
    rtok = _release_tokenizer(ref, tok)
    for d in load_long_documents(5, split="validation"):
        p, r = EC.prompt_and_response(tok, d["text"], 128)
        assert p == ref["get_prompt_ids_wikitext"](d["text"], rtok, prompt_len=128)
        assert r == ref["get_response_ids_wikitext"](d["text"], rtok, prompt_len=128)


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_entropies_match_and_log_losses_fix_the_release_offset(ref, arch):
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=2)
    hf = types.SimpleNamespace(model=model)
    g = torch.Generator().manual_seed(0)
    prompts = [torch.randint(1, 900, (n,), generator=g).tolist() for n in (12, 12, 12)]
    responses = [torch.randint(1, 900, (n,), generator=g).tolist() for n in (20, 14, 9)]
    ents_ref = ref["compute_logprobs"](hf, prompts, responses, temperature=1.0, compute_ent=True)
    lps_ref = ref["compute_logprobs"](hf, prompts, responses, temperature=1.0, compute_ent=False)
    longest = int(np.argmax([len(r) for r in responses]))
    for k, (p, r) in enumerate(zip(prompts, responses)):
        ours_h = EC.generation_entropies(model, p, r, "cpu")
        np.testing.assert_allclose(ours_h, ents_ref[k], rtol=1e-4, atol=1e-5)
        ours_l = EC.response_log_losses(model, p, r, "cpu")
        # The release's slice starts one position late: it holds response
        # tokens 2..n, and in padded rows one more value, the pad's logprob.
        np.testing.assert_allclose(-lps_ref[k][:len(r) - 1], ours_l[1:], rtol=1e-4, atol=1e-5)
        assert len(lps_ref[k]) == (len(r) - 1 if k == longest else len(r))


def test_self_generated_text_is_calibrated():
    """Known property: on the model's own samples the mean log loss equals the
    mean entropy in expectation, so EntCE on (sample, sample) is near 0, well
    inside the spread of the per-token terms."""
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    g = torch.Generator().manual_seed(1)
    prompts = [torch.randint(1, 900, (8,), generator=g).tolist() for _ in range(24)]
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(3)
        gens = EC._sample(model, tok, prompts, 64, 8, "cpu")
        H = np.concatenate([EC.generation_entropies(model, p, x, "cpu") for p, x in zip(prompts, gens)])
        L = np.concatenate([EC.response_log_losses(model, p, x, "cpu") for p, x in zip(prompts, gens)])
    se = np.std(L - H, ddof=1) / np.sqrt(L.size)
    assert abs(H.mean() - L.mean()) < 4 * se


def test_global_rng_is_untouched():
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    docs = load_long_documents(2, split="validation")
    torch.manual_seed(123)
    before = torch.rand(3)
    torch.manual_seed(123)
    r = EC.EntropyCalibrationTask({"num_samples": 2, "prompt_len": 16, "max_new_tokens": 16,
                                   "min_response_tokens": 16, "min_bytes": 0, "use_dataset": True,
                                   "n_bootstrap": 10}).evaluate(model, tok, docs)
    assert r["_meta_n_examples"] == 2
    assert torch.equal(torch.rand(3), before)


def test_generation_is_cut_at_every_stop_id():
    """A stop id from the generation config that is not the tokenizer's EOS
    (Qwen3: <|im_end|> and <|endoftext|>) ends the continuation; the padding
    generate() appends after it is not scored."""
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    stop = 7
    model.generation_config.eos_token_id = [stop]
    assert EC.stop_ids(model, tok) == {stop, tok.eos_token_id}
    g = torch.Generator().manual_seed(4)
    prompts = [torch.randint(1, 900, (8,), generator=g).tolist() for _ in range(32)]
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(5)
        gens = EC._sample(model, tok, prompts, 64, 8, "cpu")
    stops = EC.stop_ids(model, tok)
    assert any(x and x[-1] in stops for x in gens)
    for x in gens:
        assert all(t not in stops for t in x[:-1])      # nothing after a stop
        assert len(x) == 64 or x[-1] in stops
