"""bits_per_byte on a tokenizer that prepends a BOS (Llama-3.x).

The byte-ownership check rebuilt the corpus from the token pieces and
compared it with the source; Llama-3's <|begin_of_text|> decoded to 17
printable bytes that are not in the corpus, so ownership was withheld and
bits_per_byte was NaN on every Llama-3.2/3.1 study model (GPU smoke run,
2026-11-16). A special token the tokenizer adds owns no bytes.
"""
import math

import pytest

transformers = pytest.importorskip("transformers")

from blme.cache import load_default_corpus  # noqa: E402
from blme.tasks.geometry.utils import _exact_token_bytes  # noqa: E402


def _llama3_tok():
    try:
        return transformers.AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"Llama-3.2 tokenizer not cached: {e}")


def test_a_prepended_bos_owns_no_bytes():
    tok = _llama3_tok()
    corpus = "\n\n".join(d["text"] for d in load_default_corpus(10))
    ids = tok(corpus)["input_ids"]
    assert ids[0] == tok.bos_token_id
    own = _exact_token_bytes(tok, ids, corpus)
    assert own is not None and own[0] == 0
    assert sum(own) == len(corpus.encode("utf-8"))


def test_bits_per_byte_is_exact_on_llama3():
    from blme.tasks.geometry.perplexity import RarePPLTask
    tok = _llama3_tok()
    cfg = transformers.LlamaConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=32,
                                   num_hidden_layers=1, num_attention_heads=2,
                                   num_key_value_heads=2, max_position_embeddings=512)
    model = transformers.LlamaForCausalLM(cfg).eval()
    r = RarePPLTask({"num_samples": 5, "use_cache": False}).evaluate(model, tok, load_default_corpus(5))
    assert r["_meta_ppl_byte_attribution"] in ("offsets", "token_pieces")
    assert math.isfinite(r["bits_per_byte"])
