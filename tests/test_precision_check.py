"""``blme.precision.check`` and ``blme precision-check`` (0.2.7).

The check reproduces the companion study's precision audit: WikiText-103
validation, the non-empty lines of the first 4,000 rows joined with newlines;
4 rows of 511 tokens, each led by the tokenizer's BOS when it prepends one;
eager attention; mean next-token NLL over finite positions with float32
logits; float32 against a 16-bit dtype, and float32 is recommended when the
16-bit NLL is higher by more than 0.02 nats.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")

from blme import precision  # noqa: E402

TINY_GPT2 = "hf-internal-testing/tiny-random-gpt2"
# Enough text for 4 x 511 tokens on any tokenizer used here.
LONG_TEXT = "\n".join(
    f"Paragraph {i}: the river carried silt past the old mill, and the town counted "
    f"{i * 7} boats before the autumn storms closed the harbour." for i in range(400))


def _tokenizer():
    return transformers.AutoTokenizer.from_pretrained(TINY_GPT2)


class _BosTokenizer:
    """The tiny GPT-2 tokenizer, but prepending a BOS by default (as Llama's does)."""

    def __init__(self):
        self.inner = _tokenizer()
        self.bos_token_id = 7

    def __call__(self, text, add_special_tokens=True, **kw):
        out = self.inner(text, add_special_tokens=False, **kw)
        if add_special_tokens:
            ids = out["input_ids"]
            out["input_ids"] = (torch.cat([torch.full((ids.shape[0], 1), 7), ids], 1)
                                if isinstance(ids, torch.Tensor) else [7] + list(ids))
        return out


def _tiny_dir(path, dtype=torch.float32):
    cfg = transformers.GPT2Config(vocab_size=1000, n_embd=32, n_layer=2, n_head=4, n_positions=1024)
    torch.manual_seed(0)
    transformers.GPT2LMHeadModel(cfg).to(dtype).save_pretrained(path)
    _tokenizer().save_pretrained(path)
    return Path(path)


def _independent_nll(path, dtype, ids):
    """The audit's arithmetic, written out: from_pretrained in ``dtype`` with eager
    attention, float32 logits, per-token cross-entropy, mean over finite values."""
    m = transformers.AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype,
                                                          attn_implementation="eager").eval()
    with torch.no_grad():
        lg = m(ids).logits.float()
    nll = []
    for r in range(ids.shape[0]):
        lp = torch.log_softmax(lg[r, :-1], dim=-1)
        nll.append(-lp.gather(1, ids[r, 1:, None])[:, 0])
    nll = torch.cat(nll)
    return float(nll[torch.isfinite(nll)].mean())


# ── layout ──────────────────────────────────────────────────────────────────

def test_rows_without_bos_are_the_first_4_x_511_tokens():
    tok = _tokenizer()
    assert not precision.prepends_bos(tok)
    ids, bos = precision.audit_rows(tok, LONG_TEXT)
    flat = tok(LONG_TEXT, add_special_tokens=False, return_tensors="pt").input_ids[0]
    assert not bos and ids.shape == (4, 511)
    assert torch.equal(ids.reshape(-1), flat[:4 * 511])


def test_rows_with_bos_lead_each_row_with_it():
    tok = _BosTokenizer()
    assert precision.prepends_bos(tok)
    ids, bos = precision.audit_rows(tok, LONG_TEXT)
    flat = tok(LONG_TEXT, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    assert bos and ids.shape == (4, 512)
    assert (ids[:, 0] == 7).all()
    assert torch.equal(ids[:, 1:].reshape(-1), flat[:4 * 511])


def test_short_text_is_an_error():
    with pytest.raises(ValueError, match="needs 2044"):
        precision.audit_rows(_tokenizer(), "too short")


# ── the rule ────────────────────────────────────────────────────────────────

def test_recommendation_rule():
    r = precision.recommend
    assert r(2.0, 2.0 + 0.03, 0, 0, "bfloat16")[0] == "float32"
    assert r(2.0, 2.0 + 0.01, 0, 0, "bfloat16")[0] == "bfloat16"
    assert r(2.0, 2.0 - 0.01, 0, 0, "bfloat16")[0] == "bfloat16"
    assert r(2.0, 2.0 + 0.5, 0, 0, "float16", threshold=1.0)[0] == "float16"
    assert r(2.0, 2.0, 0, 3, "float16")[0] == "float32"          # non-finite only in fp16
    assert r(2.0, float("nan"), 0, 0, "bfloat16")[0] == "float32"
    assert r(2.0, 2.0, 2, 2, "bfloat16")[0] == "bfloat16"         # same non-finite count


def test_arguments_are_validated():
    with pytest.raises(ValueError, match="dtype"):
        precision.check("pretrained=x", dtype="float32", text=LONG_TEXT)
    with pytest.raises(ValueError, match="threshold"):
        precision.check("pretrained=x", threshold=-1, text=LONG_TEXT)
    with pytest.raises(ValueError, match="pretrained"):
        precision.check("dtype=bfloat16", text=LONG_TEXT)


# ── end to end ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("candidate", ["bfloat16", "float16"])
def test_check_matches_the_audit_arithmetic(tmp_path, candidate):
    path = _tiny_dir(tmp_path)
    rec = precision.check(f"pretrained={path},dtype=float16,attn_implementation=sdpa",
                          dtype=candidate, text=LONG_TEXT, device="cpu")
    ids, _ = precision.audit_rows(_tokenizer(), LONG_TEXT)
    for dt in ("float32", candidate):
        assert rec[f"param_dtypes_{dt}"] == [dt]          # model_args' dtype is replaced
        assert rec[f"dtype_contract_{dt}"] == "honoured"
        assert rec[f"nonfinite_{dt}"] == 0
        want = _independent_nll(path, getattr(torch, dt), ids)
        assert math.isclose(rec[f"nll_{dt}"], want, rel_tol=0, abs_tol=1e-6), (dt, rec[f"nll_{dt}"], want)
    assert rec["delta"] == rec[f"nll_{candidate}"] - rec["nll_float32"]
    assert (rec["recommendation"], rec["reason"]) == precision.recommend(
        rec["nll_float32"], rec[f"nll_{candidate}"], 0, 0, candidate, 0.02)
    m = rec["method"]
    assert m["text"] == "custom" and m["bos_prepended"] is False
    assert (m["rows"], m["tokens_per_row"], m["predictions"]) == (4, 511, 4 * 510)
    assert m["attn_implementation"] == "eager"
    assert rec["model_class"] == "GPT2LMHeadModel" and rec["native_dtype"] == "float32"


def test_threshold_drives_the_recommendation(tmp_path):
    path = _tiny_dir(tmp_path)
    loose = precision.check(f"pretrained={path}", text=LONG_TEXT, device="cpu", threshold=1e6)
    assert loose["recommendation"] == "bfloat16"
    if loose["delta"] > 0:
        strict = precision.check(f"pretrained={path}", text=LONG_TEXT, device="cpu", threshold=0.0)
        assert strict["recommendation"] == "float32" and strict["delta"] == loose["delta"]


def test_cli(tmp_path, monkeypatch, capsys):
    from blme import cli
    path = _tiny_dir(tmp_path / "model")
    monkeypatch.setattr(precision, "audit_text", lambda: LONG_TEXT)
    out_file = tmp_path / "check.json"
    monkeypatch.setattr(sys, "argv", ["blme", "precision-check", "--model-args", f"pretrained={path}",
                                      "--device", "cpu", "--threshold", "0.05",
                                      "--output", str(out_file)])
    cli.main()
    printed = capsys.readouterr().out
    body, last = printed.rsplit("Recommendation:", 1)
    rec = json.loads(body)
    assert rec["threshold_nats"] == 0.05 and rec["candidate_dtype"] == "bfloat16"
    assert rec["method"]["text"].startswith("WikiText-103 validation")
    assert last.strip().startswith(f"run {rec['recommendation']}")
    assert json.loads(out_file.read_text()) == rec


def test_audit_text_and_gpt2():
    """The audit's own text and gpt2 (skipped when either is not cached)."""
    try:
        text = precision.audit_text()
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"WikiText-103 or gpt2 is not available offline: {e}")
    assert text.startswith(" = Homarus gammarus = \n\n Homarus gammarus , known as")
    rec = precision.check("pretrained=gpt2", text=text, device="cpu")
    assert rec["method"]["bos_prepended"] is False and rec["method"]["tokens_per_row"] == 511
    assert rec["param_dtypes_float32"] == ["float32"] and rec["param_dtypes_bfloat16"] == ["bfloat16"]
    ids, _ = precision.audit_rows(tok, text)
    assert math.isclose(rec["nll_float32"], _independent_nll("gpt2", torch.float32, ids), abs_tol=1e-6)
    # The audit measured gpt2 at 3.37809 in float32 (GPU); CPU arithmetic agrees to 1e-4.
    assert abs(rec["nll_float32"] - 3.37809) < 1e-4
    assert rec["recommendation"] in ("bfloat16", "float32")
