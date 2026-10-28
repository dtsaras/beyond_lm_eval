"""Position 0 + massive-row exclusion, and trailing special-token stripping.

The policy (user decision 2026-10-28): non-LLM-origin point clouds drop
position 0 and every token whose norm exceeds 10x the median norm (rows at
position > 0) at any middle depth -- one mask per token for all layers.
Tokenizer-appended EOS/SEP tokens never reach the model.
"""
import pytest
import torch

from blme.tasks._tokens import (MASSIVE_FACTOR, middle_depth_layers, select_rows,
                                sink_keep_masks, sink_meta, strip_trailing_special)


def _docs(seed=0, lengths=(6, 5, 7), d=8):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(T, d, generator=g, dtype=torch.float64) for T in lengths]


def test_sink_policies_require_masks():
    with pytest.raises(ValueError, match="keep masks"):
        select_rows(_docs(), "exclude_sinks")


def test_a_row_massive_at_one_middle_depth_is_dropped_at_every_depth():
    depth_a, depth_b = _docs(0), _docs(1)
    depth_b[1][3] *= 1000.0                      # massive only at depth b
    masks, info = sink_keep_masks([depth_a, depth_b])
    assert info["n_massive_dropped"] == 1
    assert not bool(masks[1][3]) and all(not bool(m[0]) for m in masks)
    rows_a, meta_a = select_rows(depth_a, "exclude_sinks", keep=masks)
    rows_b, meta_b = select_rows(depth_b, "exclude_sinks", keep=masks)
    # Same rows at both depths: the cross-layer comparison stays paired.
    assert torch.equal(meta_a.position, meta_b.position) and torch.equal(meta_a.doc, meta_b.doc)
    assert rows_a.shape[0] == sum(T - 1 for T in (6, 5, 7)) - 1
    assert not ((meta_a.doc == 1) & (meta_a.position == 3)).any()


def test_exclude_sinks_with_no_massive_row_equals_exclude_pos0():
    docs = _docs(2)
    masks, info = sink_keep_masks([docs])
    assert info["n_massive_dropped"] == 0
    a, _ = select_rows(docs, "exclude_sinks", keep=masks)
    b, _ = select_rows(docs, "exclude_pos0")
    assert torch.equal(a, b)


def test_mean_excl_sinks_is_the_mean_of_the_kept_rows():
    docs = _docs(3)
    docs[0][2] *= 500.0
    masks, _ = sink_keep_masks([docs])
    pooled, _ = select_rows(docs, "mean_excl_sinks", keep=masks)
    want = docs[0][[1, 3, 4, 5]].mean(0)
    assert torch.allclose(pooled[0], want, rtol=0, atol=1e-12)


def test_position0_is_never_counted_as_massive():
    docs = _docs(4)
    for c in docs:
        c[0] *= 1e4                              # a BOS-style sink at position 0
    masks, info = sink_keep_masks([docs])
    assert info["n_massive_dropped"] == 0        # it is excluded by position, not by norm
    assert sink_meta(info)["_meta_sink_factor"] == MASSIVE_FACTOR


def test_middle_depth_layers_picks_relative_depth():
    layers = {k: None for k in range(11)}        # ranks 0..10
    assert middle_depth_layers(layers) == list(range(1, 10))
    assert middle_depth_layers({5: None, 6: None}) == [5, 6]


class _Tok:
    eos_token_id = 9
    sep_token_id = None
    unk_token_id = 3


def test_strip_trailing_special_drops_appended_eos_only():
    enc = {"input_ids": torch.tensor([[1, 5, 6, 9]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    out, n = strip_trailing_special(enc, _Tok())
    assert n == 1 and out["input_ids"].tolist() == [[1, 5, 6]] and out["attention_mask"].shape == (1, 3)
    # An UNK at the end is real text and stays; a lone special is never emptied.
    assert strip_trailing_special({"input_ids": torch.tensor([[5, 3]])}, _Tok())[1] == 0
    assert strip_trailing_special({"input_ids": torch.tensor([[9]])}, _Tok())[1] == 0


def test_stripping_a_trailing_token_leaves_every_other_row_bit_identical():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "hf-internal-testing/tiny-random-gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).eval()
    enc = tok("The cat sat on the mat.", return_tensors="pt")
    eos = torch.tensor([[tok.eos_token_id]])
    with_eos = {"input_ids": torch.cat([enc["input_ids"], eos], 1),
                "attention_mask": torch.ones(1, enc["input_ids"].shape[1] + 1, dtype=torch.long)}
    stripped, n = strip_trailing_special(with_eos, tok)
    assert n == 1
    with torch.no_grad():
        a = model(**with_eos, output_hidden_states=True).hidden_states
        b = model(**stripped, output_hidden_states=True).hidden_states
    T = stripped["input_ids"].shape[1]
    assert all(torch.equal(x[:, :T], y) for x, y in zip(a, b))


def test_collector_drops_an_appended_eos():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from blme.tasks.geometry.utils import collect_hidden_states
    name = "hf-internal-testing/tiny-random-gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).eval()

    class AppendEOS:
        """A tokenizer that appends EOS to every sequence, as OLMo-1B's does."""
        eos_token_id, sep_token_id = tok.eos_token_id, None

        def __call__(self, text, **kw):
            enc = tok(text, **kw)
            enc["input_ids"] = torch.cat([enc["input_ids"], torch.tensor([[tok.eos_token_id]])], 1)
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])
            return enc

    text = ["The cat sat on the mat."]
    plain = collect_hidden_states(model, tok, text, num_samples=1, layer_idx="all", per_sample=True)
    eos = collect_hidden_states(model, AppendEOS(), text, num_samples=1, layer_idx="all", per_sample=True)
    for li in plain:
        assert torch.equal(plain[li][0], eos[li][0])
