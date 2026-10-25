"""Byte ownership comes from the token PIECES, not from character offsets.

A byte-level BPE splits one character across several tokens and the offset map
reports the SAME span for each of them, so no arithmetic on offsets can say
which token owns which byte. Dividing the shared span evenly invents ownership.
"""
import pytest

transformers = pytest.importorskip("transformers")

from blme.tasks.geometry.utils import _exact_token_bytes, _U2B


@pytest.fixture(scope="module")
def gpt2_tok():
    return transformers.AutoTokenizer.from_pretrained("gpt2")


def test_emoji_piece_ownership_is_exact(gpt2_tok):
    """'\U0001f600b' is 5 source bytes owned [3, 1, 1], not [2, 2, 1]."""
    text = "\U0001f600b"
    ids = gpt2_tok(text)["input_ids"]
    own = _exact_token_bytes(gpt2_tok, ids, text)
    assert own == [3, 1, 1]
    assert sum(own) == len(text.encode("utf-8")) == 5
    # The first token is never scored, so 2 bytes are scored -- not 3.
    assert sum(own[1:]) == 2


def test_sentence_scored_bytes(gpt2_tok):
    """The reviewer's sentence has 49 scored bytes; the even split counted 50."""
    text = "\U0001f600 This is an ordinary sentence about the weather."
    ids = gpt2_tok(text)["input_ids"]
    own = _exact_token_bytes(gpt2_tok, ids, text)
    assert sum(own) == len(text.encode("utf-8"))
    assert sum(own[1:]) == 49


def test_reconstruction_is_verified_not_assumed(gpt2_tok):
    """Ownership is returned only when the pieces rebuild the source exactly."""
    text = "The quick brown fox.\n\nSecond document \U0001f600 here."
    ids = gpt2_tok(text)["input_ids"]
    own = _exact_token_bytes(gpt2_tok, ids, text)
    assert own is not None and sum(own) == len(text.encode("utf-8"))
    # A corpus that does NOT correspond to the ids must be refused.
    assert _exact_token_bytes(gpt2_tok, ids, text + "extra") is None


def test_byte_level_alphabet_is_a_bijection():
    """The GPT-2 byte alphabet must cover all 256 byte values exactly once."""
    assert len(_U2B) == 256
    assert sorted(_U2B.values()) == list(range(256))


def test_bits_per_byte_is_exact_end_to_end(gpt2_tok):
    """Drive the real strided_perplexity, not the helper alone."""
    import math
    from blme.tasks.geometry.utils import strided_perplexity
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2").eval()
    text = "\U0001f600b"
    _ppl, nll, n_scored, n_bytes, _n_chars, _ml, _st, mode = strided_perplexity(
        model, gpt2_tok, [{"text": text}], num_samples=1)
    assert mode == "token_pieces"
    assert n_bytes == 2, f"scored bytes {n_bytes}, expected 2"
    bpb = nll * n_scored / (math.log(2) * n_bytes)
    # Reference 12.215388353192; float32 forward noise is ~3e-7 relative.
    assert bpb == pytest.approx(12.215388353192, rel=1e-5)


def test_offsets_fallback_is_used_only_when_no_span_is_shared(gpt2_tok,
                                                              monkeypatch):
    """If piece ownership is unavailable, the offset partition is exact ONLY
    when no span was shared between tokens; otherwise BPB must be withheld.

    A normalizing tokenizer (WordPiece with lower-casing) cannot reconstruct
    the source bytes, so this path is live -- but on ASCII the offset partition
    is still exact and there is no reason to discard it.
    """
    import blme.tasks.geometry.utils as U
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2").eval()
    monkeypatch.setattr(U, "_exact_token_bytes", lambda *a, **k: None)

    ascii_only = [{"text": "The quick brown fox jumps over the lazy dog."}]
    r = U.strided_perplexity(model, gpt2_tok, ascii_only, num_samples=1)
    assert r[7] == "offsets", f"ASCII with no shared spans should stay exact: {r[7]}"
    assert r[3] == 41

    # An emoji forces a shared span; ownership is then genuinely unknown.
    shared = [{"text": "\U0001f600 hello world"}]
    r2 = U.strided_perplexity(model, gpt2_tok, shared, num_samples=1)
    assert r2[7] != "offsets", (
        "a shared span was divided evenly and reported as exact")
    assert r2[7] == "full_corpus_upper_bound"


def test_shared_span_with_pieces_available_is_exact(gpt2_tok):
    """The same emoji corpus IS exact once piece ownership is used."""
    from blme.tasks.geometry.utils import strided_perplexity
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2").eval()
    r = strided_perplexity(model, gpt2_tok,
                           [{"text": "\U0001f600 hello world"}], num_samples=1)
    assert r[7] == "token_pieces"
    # 16 source bytes; the unscored first token owns 3.
    assert r[3] == 13
