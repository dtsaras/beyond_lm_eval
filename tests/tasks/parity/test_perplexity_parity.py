"""Numeric-parity test: geometry_perplexity rare/freq buckets + BOS-safe BPC.

TASK: geometry_perplexity (src/blme/tasks/geometry/perplexity.py, RarePPLTask)

Campaign-4 B7 closes two gaps:
  * ppl_rare / ppl_freq were only finiteness-checked — changing the 20%
    fraction, the (count, id) lexsort tie-break, or swapping the buckets
    passed every test. Here the buckets and both PPLs are reconstructed
    from scratch (float64 log-softmax forward, python sort) on gpt2.
  * BPC on a BOS-adding tokenizer: the historic bug recounted tokens with
    add_special_tokens=False while scoring used the default, undercounting
    ~1 token/doc (~13% low). The wrapper tokenizer below emulates a
    Llama/Gemma-style BOS and pins BPC to the closed-form reduction
    nll_total / ln(2) / total_chars, which holds exactly when the recount
    matches the scored window.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")

from blme.tasks.geometry.perplexity import RarePPLTask  # noqa: E402

CORPUS = [
    {"text": "The quick brown fox jumps over the lazy dog."},
    {"text": "A journey of a thousand miles begins with a single step."},
    {"text": "The quick brown fox naps in the warm afternoon sun."},
    {"text": "Mathematics is the language of the universe we inhabit."},
]


def _load_gpt2():
    try:
        model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").eval()
        tok = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        return model, tok
    except Exception as exc:  # offline, no cache
        pytest.skip(f"gpt2 unavailable offline: {type(exc).__name__}")


def _reference_stats(model, tok, corpus, bos_id=None):
    """Independent float64 reference: per-token shifted NLLs + counts."""
    nlls, label_ids = [], []
    counts = {}
    total_chars = 0
    scored_tokens = 0
    with torch.no_grad():
        for row in corpus:
            text = row["text"]
            ids = tok(text)["input_ids"]
            if bos_id is not None:
                ids = [bos_id] + ids
            if len(ids) < 2:
                continue
            total_chars += len(text)
            scored_tokens += len(ids) - 1
            t = torch.tensor([ids])
            logits = model(t).logits[0].double()
            logp = F.log_softmax(logits, dim=-1)
            for pos in range(len(ids) - 1):
                tid = ids[pos + 1]
                nlls.append(-float(logp[pos, tid]))
                label_ids.append(tid)
                counts[tid] = counts.get(tid, 0) + 1
    return nlls, label_ids, counts, total_chars, scored_tokens


def _reference_buckets(counts):
    """Rare/freq id sets per the documented protocol: observed ids only,
    stable sort by (count, id), bottom / top round(20%)."""
    observed = sorted(counts.keys(), key=lambda tid: (counts[tid], tid))
    n = len(observed)
    k = max(1, int(round(n * 0.2)))
    return set(observed[:k]), set(observed[max(1, n - k):])


def test_rare_freq_buckets_and_ppls_match_independent_reference():
    model, tok = _load_gpt2()
    task = RarePPLTask(config={"num_samples": len(CORPUS)})
    out = task.evaluate(model, tok, dataset=list(CORPUS))
    assert "error" not in out

    nlls, label_ids, counts, total_chars, scored = _reference_stats(
        model, tok, CORPUS)
    rare_ids, freq_ids = _reference_buckets(counts)

    ref_overall = float(np.exp(np.mean(nlls)))
    rare_nll = [v for v, tid in zip(nlls, label_ids) if tid in rare_ids]
    freq_nll = [v for v, tid in zip(nlls, label_ids)
                if tid in freq_ids and tid not in rare_ids]
    ref_rare = float(np.exp(np.mean(rare_nll)))
    ref_freq = float(np.exp(np.mean(freq_nll)))
    ref_bpc = float(np.mean(nlls) / np.log(2) * (scored / total_chars))

    assert out["ppl_overall"] == pytest.approx(ref_overall, rel=1e-4)
    assert out["ppl_rare"] == pytest.approx(ref_rare, rel=1e-4)
    assert out["ppl_freq"] == pytest.approx(ref_freq, rel=1e-4)
    assert out["bits_per_char"] == pytest.approx(ref_bpc, rel=1e-4)
    # The buckets must actually differ (rare tokens are harder).
    assert out["ppl_rare"] > out["ppl_freq"]


class _BosTokenizer:
    """Wrap the gpt2 tokenizer to PREPEND a BOS token by default, emulating
    Llama/Gemma-style tokenizers (the historic BPC bug's trigger)."""

    def __init__(self, tok):
        self._tok = tok
        self.bos_token_id = tok.eos_token_id
        self.eos_token_id = tok.eos_token_id
        self.pad_token_id = tok.eos_token_id

    def __call__(self, text, add_special_tokens=True, **kw):
        enc = self._tok(text, **kw)
        if add_special_tokens and not kw.get("return_tensors"):
            enc["input_ids"] = [self.bos_token_id] + enc["input_ids"]
        elif add_special_tokens and kw.get("return_tensors") == "pt":
            import torch as _t
            bos = _t.tensor([[self.bos_token_id]])
            enc["input_ids"] = _t.cat([bos, enc["input_ids"]], dim=1)
            if "attention_mask" in enc:
                enc["attention_mask"] = _t.ones_like(enc["input_ids"])
        return enc


def test_bpc_correct_for_bos_adding_tokenizer():
    """Regression (campaign-3 BUG): with a BOS-prepending tokenizer the BPC
    recount must use the SAME tokenization as scoring; the closed-form
    reduction bpc == nll_total/ln2/chars then holds exactly."""
    model, tok = _load_gpt2()
    bos_tok = _BosTokenizer(tok)

    task = RarePPLTask(config={"num_samples": len(CORPUS)})
    out = task.evaluate(model, bos_tok, dataset=list(CORPUS))
    assert "error" not in out

    nlls, _, _, total_chars, scored = _reference_stats(
        model, tok, CORPUS, bos_id=bos_tok.bos_token_id)
    ref_bpc = float(np.mean(nlls) / np.log(2) * (scored / total_chars))
    assert out["bits_per_char"] == pytest.approx(ref_bpc, rel=1e-4), (
        "BPC recount desynced from the scored token window (BOS bug)"
    )


def test_empty_text_rows_are_skipped_not_crashing():
    """Regression (campaign-3 BUG): empty rows (present in WikiText) crashed
    the forward pass; both the scoring and BPC loops must skip them
    consistently."""
    model, tok = _load_gpt2()
    corpus = [{"text": ""}] + list(CORPUS[:2]) + [{"text": ""}]
    task = RarePPLTask(config={"num_samples": len(corpus)})
    out = task.evaluate(model, tok, dataset=corpus)
    assert "error" not in out
    assert np.isfinite(out["ppl_overall"]) and np.isfinite(out["bits_per_char"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
