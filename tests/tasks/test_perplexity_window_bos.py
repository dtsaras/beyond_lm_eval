"""Strided perplexity starts every window with BOS when the tokenizer prepends one.

The HuggingFace recipe gives only the first window a BOS. A model that always
sees its BOS first (Gemma) is off distribution in every later window: on the
study sweep gemma-2-9b's corpus perplexity read 493 against 13.2 per document.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

DOCS = [
    "The capital of France is Paris and it is a beautiful city on the Seine.",
    "Machine learning models can memorize training data in surprising ways.",
    "Photosynthesis converts sunlight into chemical energy inside plant cells.",
    "The quick brown fox jumps over the lazy dog while the cat watches quietly.",
]


@pytest.fixture(scope="module")
def gpt2():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    return tok, model


@pytest.fixture
def bos_tok(gpt2):
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.add_bos_token = True          # a Llama/Gemma-style tokenizer
    return tok


class Recorder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model, self.windows = model, []
        self.config = model.config

    @property
    def device(self):
        return self.model.device

    def forward(self, ids, labels=None):
        self.windows.append(ids[0].tolist())
        return self.model(ids, labels=labels)


def reference_nll(model, ids, window, stride):
    """Independent reference: windows [BOS] + content[b:e], each content token
    scored once, in the window that first reveals it, from raw logits."""
    bos, content = ids[:1], ids[1:]
    width, step = window - 1, min(stride, window - 1)
    total, n, prev_end = 0.0, 0, 0
    for b in range(0, len(content), step):
        e = min(b + width, len(content))
        if e <= prev_end:
            continue
        win = torch.tensor([bos + content[b:e]])
        with torch.no_grad():
            logp = F.log_softmax(model(win).logits[0].double(), dim=-1)
        for k in range(prev_end, e):              # content index k sits at window position k - b + 1
            p = k - b + 1
            total -= float(logp[p - 1, win[0, p]])
            n += 1
        prev_end = e
        if e == len(content):
            break
    return total / n, n


@pytest.mark.parametrize("window,stride", [(16, 8), (16, 15), (16, 16), (12, 5)])
def test_every_window_starts_with_bos_and_matches_the_reference(gpt2, bos_tok, window, stride):
    from blme.tasks.geometry.utils import strided_perplexity
    _, model = gpt2
    rec = Recorder(model)
    ppl, nll, n, *_ = strided_perplexity(rec, bos_tok, DOCS, num_samples=4, max_length=window, stride=stride)
    ids = bos_tok("\n\n".join(DOCS))["input_ids"]
    assert ids[0] == bos_tok.bos_token_id
    assert len(rec.windows) > 3 and all(w[0] == bos_tok.bos_token_id for w in rec.windows)
    assert all(len(w) <= window for w in rec.windows)
    assert rec.windows[0] == ids[:window]                       # the stream's own first window
    assert n == len(ids) - 1                                    # every content token once, none skipped
    ref, n_ref = reference_nll(model, ids, window, stride)
    assert n_ref == n
    assert abs(nll - ref) < 2e-5 and abs(ppl - np.exp(ref)) / np.exp(ref) < 2e-5


def test_bos_windows_change_the_value_and_never_restores_the_gpt2_recipe(gpt2, bos_tok):
    from blme.tasks.geometry.utils import strided_perplexity
    _, model = gpt2
    rec = Recorder(model)
    auto = strided_perplexity(model, bos_tok, DOCS, num_samples=4, max_length=16, stride=8)
    never = strided_perplexity(rec, bos_tok, DOCS, num_samples=4, max_length=16, stride=8, window_bos="never")
    assert auto[1] != never[1]
    assert sum(w[0] == bos_tok.bos_token_id for w in rec.windows) == 1   # BOS in the first window only


def test_tokenizers_without_bos_are_unchanged(gpt2):
    from blme.tasks.geometry.utils import strided_perplexity, tokenizer_prepends_bos
    tok, model = gpt2
    assert not tokenizer_prepends_bos(tok)
    auto = strided_perplexity(model, tok, DOCS, num_samples=4, max_length=16, stride=8)
    never = strided_perplexity(model, tok, DOCS, num_samples=4, max_length=16, stride=8, window_bos="never")
    assert auto == never


def test_unknown_mode_raises(gpt2, bos_tok):
    from blme.tasks.geometry.utils import strided_perplexity
    with pytest.raises(ValueError, match="window_bos"):
        strided_perplexity(gpt2[1], bos_tok, DOCS, num_samples=4, max_length=16, stride=8, window_bos="always")


def test_task_records_the_choice_and_exact_bytes(gpt2, bos_tok):
    from blme.tasks.geometry.perplexity import RarePPLTask
    tok, model = gpt2
    data = [{"text": d} for d in DOCS]
    cfg = {"num_samples": 4, "use_cache": False, "ppl_max_length": 16, "ppl_stride": 8}
    r_bos = RarePPLTask(config=cfg).evaluate(model, bos_tok, data)
    r_plain = RarePPLTask(config=cfg).evaluate(model, tok, data)
    r_never = RarePPLTask(config={**cfg, "ppl_window_bos": "never"}).evaluate(model, bos_tok, data)
    assert r_bos["_meta_ppl_window_bos"] is True
    assert r_plain["_meta_ppl_window_bos"] is False and r_never["_meta_ppl_window_bos"] is False
    assert r_bos["_meta_ppl_byte_attribution"] == r_never["_meta_ppl_byte_attribution"]
    assert np.isfinite(r_bos["bits_per_byte"]) and r_bos["bits_per_byte"] != r_never["bits_per_byte"]
    # The same scored span: the byte denominator is unchanged, only the NLL differs.
    ratio = r_bos["bits_per_byte"] / r_never["bits_per_byte"]
    assert abs(ratio - r_bos["mean_nll_nats"] / r_never["mean_nll_nats"]) < 1e-9
