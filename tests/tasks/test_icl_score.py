"""consistency_icl_score: Olsson et al.'s definition against a brute-force
per-token cross-entropy, the byte anchors, and a known positive on gpt2."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from blme.cache import load_long_documents  # noqa: E402
from blme.tasks.consistency import icl_score as IS  # noqa: E402


def _docs(n):
    return load_long_documents(n, split="validation")


# The tiny models read 512 positions and their tokenizer about 2 bytes per
# token, so their byte anchors are moved inside that window.
_TINY = {"use_long_documents": False, "early_byte": 100, "late_byte": 800}


def _brute_force(model, tok, texts, max_length, bytes_=(100, 800)):
    """Every position's cross-entropy at once, then the paper's two tokens and
    the tokens covering the two byte anchors, located by decoding prefixes."""
    tok_d, byte_d = [], []
    for text in texts:
        enc = tok(text, truncation=True, max_length=max_length, return_special_tokens_mask=True)
        ids = torch.tensor([enc["input_ids"]])
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0].float()
        ce = F.cross_entropy(logits[:-1], ids[0, 1:], reduction="none")
        content = [j for j, m in enumerate(enc["special_tokens_mask"]) if not m]
        if len(content) < 500:
            continue
        tok_d.append(float(ce[content[499] - 1] - ce[content[49] - 1]))
        # Token j covers byte b when the decoded prefix through j first passes b.
        lengths = [len(tok.decode(ids[0, content[0]:j + 1]).encode("utf-8")) for j in content]
        cover = [next(j for j, n in zip(content, lengths) if n > b) for b in bytes_]
        byte_d.append(float(ce[cover[1] - 1] - ce[cover[0] - 1]))
    return sum(tok_d) / len(tok_d), sum(byte_d) / len(byte_d), len(tok_d)


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_matches_brute_force(arch):
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=2)
    tok, docs = fd.tokenizer(), _docs(4)
    r = IS.ICLScoreTask({"num_samples": 4, **_TINY}).evaluate(model, tok, docs)
    score, score_b, n = _brute_force(model, tok, [d["text"] for d in docs], 512)
    assert r["_meta_n_contexts"] == n == 4
    assert r["icl_score"] == pytest.approx(score, rel=1e-5, abs=1e-6)
    assert r["icl_score_byte_anchored"] == pytest.approx(score_b, rel=1e-5, abs=1e-6)


def test_byte_spans_count_utf8_bytes():
    text = "aé€𝄞b"
    spans = IS._byte_spans(text, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    assert spans == [(0, 1), (1, 3), (3, 6), (6, 10), (10, 11)]
    assert IS._covering(spans, range(5), 3) == 2 and IS._covering(spans, range(5), 10) == 4


def test_short_documents_are_filtered_by_bytes():
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    docs = [{"text": "short " * 10}] + _docs(2)
    r = IS.ICLScoreTask({"num_samples": 3, **_TINY}).evaluate(model, tok, docs)
    assert r["_meta_n_contexts"] == 2


def _gpt2():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"gpt2 not cached: {e}")
    return model, tok


def test_gpt2_copies_a_repeated_sequence():
    """Known positive (Olsson et al.'s induction setting): a 200-word random
    sequence repeated four times. Token 50 and byte 200 sit in its first copy,
    token 500 and byte 2000 inside a repeat, which induction heads predict."""
    model, tok = _gpt2()
    g = torch.Generator().manual_seed(0)
    words = [w for w in (tok.decode([i]).strip() for i in range(1000, 8000))
             if w.isalpha() and w.isascii() and len(w) > 3]
    docs = []
    for _ in range(6):
        seq = " ".join(words[i] for i in torch.randint(len(words), (200,), generator=g))
        docs.append({"text": " ".join([seq] * 4)})
    r = IS.ICLScoreTask({"num_samples": 6, "use_long_documents": False}).evaluate(model, tok, docs)
    assert r["_meta_n_contexts"] == 6
    assert r["icl_score"] < -3.0 and r["icl_score_byte_anchored"] < -3.0


def test_gpt2_natural_text_sign():
    """On WikiText articles the 500th token is easier than the 50th on average;
    single-token losses are noisy (40 articles: about -0.2 +- 0.6 nats), so
    this pins the sign only."""
    model, tok = _gpt2()
    r = IS.ICLScoreTask({"num_samples": 40, "use_long_documents": False}).evaluate(
        model, tok, _docs(40))
    assert r["_meta_n_contexts"] >= 30
    assert r["icl_score"] < 0
