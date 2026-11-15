"""The 20-word windows behind the one-point-per-sequence clouds (Cheng et al. 2024).

Cheng et al.'s corpora (id-llm-abstraction@4a170a2, ``corpora/*_sane_ds.txt``)
hold 20 consecutive whitespace words per sequence, the same text for every
model; ``scripts/extract_final_representations.py`` reads the last token.
"""
import hashlib
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from blme.cache import load_default_corpus  # noqa: E402
from blme.tasks._windows import (is_punct_final, window_last_rows, window_options,  # noqa: E402
                                 window_set, word_window)
from blme.tasks.geometry._sequence_points import sequence_clouds  # noqa: E402

LONG = " ".join(f"w{i}" for i in range(57))


def test_window_is_twenty_consecutive_words_at_the_hashed_offset():
    w = word_window(LONG)
    words = LONG.split()
    off = int.from_bytes(hashlib.sha256(f"0\x00{LONG}".encode()).digest()[:8], "big") % (57 - 19)
    assert w == " ".join(words[off:off + 20])
    assert word_window(LONG) == w                       # deterministic
    assert word_window(" ".join(words[:19])) is None    # too short: skipped
    assert word_window(" ".join(words[:20])) == " ".join(words[:20])
    assert len({word_window(LONG, seed=s) for s in range(20)}) > 1


def test_the_window_of_a_passage_depends_on_its_text_only():
    data = [{"text": f"{LONG} p{i}"} for i in range(12)]
    full = window_set(data, 12)
    assert window_set(data, 5).texts == full.texts[:5]                 # N nests
    assert window_set(list(reversed(data)), 12).texts == list(reversed(full.texts))
    assert window_set(data, 12).digest == full.digest
    assert window_set(data, 12, seed=1).digest != full.digest


def test_short_passages_are_skipped_and_counted():
    data = [{"text": "too short"}, {"text": LONG}, {"text": "also short here"}, {"text": LONG + " x"}]
    ws = window_set(data, 10)
    assert len(ws.texts) == 2 and ws.n_short == 2 and ws.n_scanned == 4
    assert ws.meta()["_meta_window_short_passages_skipped"] == 2


def test_punctuation_final_windows_are_kept_and_can_be_dropped():
    assert is_punct_final("a b .") and is_punct_final("a b ,") and is_punct_final('a b ")')
    assert not is_punct_final("a b c") and not is_punct_final("a 1990") and not is_punct_final("a U.S.")
    data = [{"text": " ".join(["x"] * 19 + ["."])}, {"text": " ".join(["y"] * 20)}]
    kept = window_set(data, 2)
    assert kept.punct_final == [True, False]
    assert kept.meta()["_meta_window_punct_final_share"] == 0.5
    dropped = window_set(data, 2, drop_punct_final=True)
    assert dropped.texts == [" ".join(["y"] * 20)] and dropped.n_punct_dropped == 1


def test_options_default_to_windows_and_reject_unknown_readouts():
    assert window_options({}) == {"readout": "window", "n_words": 20, "seed": 0,
                                  "drop_punct_final": False}
    assert window_options({"points_readout": "passage"})["readout"] == "passage"
    with pytest.raises(ValueError):
        window_options({"points_readout": "mean"})


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_rows_are_the_last_token_of_each_window(arch):
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=2)
    tok = fd.tokenizer()
    texts = window_set(load_default_corpus(40), 6).texts
    rows, info = window_last_rows(model, tok, texts, select=range, replace_final_norm=False)
    L = len(rows)
    with torch.no_grad():
        for i, t in enumerate(texts):
            ids = tok(t, return_tensors="pt")["input_ids"]
            hs = model(ids, output_hidden_states=True).hidden_states
            for k in range(L):
                torch.testing.assert_close(rows[k][i], hs[k + 1][0, -1].float(), rtol=0, atol=0)
            assert info["n_tokens"][i] == ids.shape[1]
    raw, rinfo = window_last_rows(model, tok, texts, select=range, replace_final_norm=True)
    assert rinfo["last_is_block"]
    for k in range(L - 1):
        assert torch.equal(raw[k], rows[k])
    assert not torch.equal(raw[L - 1], rows[L - 1])     # raw block output, not the final norm


def test_an_appended_eos_is_not_read():
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    texts = window_set(load_default_corpus(20), 3).texts

    class AppendsEos:
        eos_token_id, sep_token_id = tok.eos_token_id, None

        def __call__(self, text, **kw):
            enc = tok(text, **kw)
            eos = torch.tensor([[tok.eos_token_id]])
            return {"input_ids": torch.cat([enc["input_ids"], eos], 1),
                    "attention_mask": torch.cat([enc["attention_mask"], torch.ones_like(eos)], 1)}

    plain, _ = window_last_rows(model, tok, texts, select=range, replace_final_norm=False)
    eos, _ = window_last_rows(model, AppendsEos(), texts, select=range, replace_final_norm=False)
    for k in plain:
        assert torch.equal(plain[k], eos[k])


def test_sequence_clouds_reads_windows_and_keeps_the_passage_token_cloud():
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    data = load_default_corpus(40)
    sc = sequence_clouds(model, tok, data, None, n=8, layer="all", replace_final_norm=True,
                         token_docs=3, points=window_options({}))
    meta = sc.meta()
    assert sc.n_points == 8
    assert meta["_meta_points_readout"] == "last_token_of_20_word_window"
    assert meta["_meta_window_set_hash"] == window_set(data, 8).digest
    assert meta["_meta_token_cloud_n_docs"] == 3                  # whole passages, sink-masked
    assert meta["_meta_window_tokens_mean"] > 20 - 1
    passage = sequence_clouds(model, tok, data, None, n=8, layer="all", replace_final_norm=True,
                              token_docs=3, points=window_options({"points_readout": "passage"}))
    assert passage.meta()["_meta_points_readout"] == "last_content_token_of_passage"
    k = max(sc.last)
    assert not torch.equal(sc.last[k], passage.last[k])
    for key in sc.tokens:                                         # same token cloud either way
        assert all(torch.equal(a, b) for a, b in zip(sc.tokens[key], passage.tokens[key]))


WINDOW_TASKS = {
    "geometry_intrinsic_dim": {"layerwise": True},
    "geometry_lid": {"k": 5},
    "geometry_vendi_score": {},
    "geometry_magnitude": {},
    "geometry_correlation_dimension": {},
    "geometry_gride": {},
    "geometry_information_imbalance": {},
    "geometry_neighborhood_overlap": {"k": 5},
    "geometry_alpha_req": {},
    "topology_betti_curve": {},
    "topology_homology": {},
    "topology_persistence_entropy": {},
    "topology_persistence_landscape": {},
}


def test_every_sequence_task_is_listed():
    from blme.core import _register_all_tasks
    from blme.registry import _TASK_REGISTRY as TASK_REGISTRY
    _register_all_tasks()
    declared = {n for n, c in TASK_REGISTRY.items()
                if getattr(c, "TOKEN_POLICY", None) == "last_per_sequence"}
    assert declared == set(WINDOW_TASKS)


@pytest.mark.parametrize("task", sorted(WINDOW_TASKS))
def test_task_reads_windows_and_the_cache_route_equals_its_own_forward(task):
    """evaluate() on the default route reads 20-word windows, reports the
    window set, and gives the same result from a shared cache (one window
    forward per run) as from its own forwards."""
    from blme.cache import ModelOutputCache
    from blme.core import _register_all_tasks
    from blme.registry import get_task
    from blme.tasks.config_loader import resolve_task_config
    _register_all_tasks()
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    data = load_default_corpus(60)
    cfg = resolve_task_config(task, dict(WINDOW_TASKS[task], num_samples=40, seed=0))
    cache = ModelOutputCache(model, tok, dataset=data, num_samples=len(data))
    cache.populate(need_hidden=True, need_attentions=False, need_logits=False)
    shared = get_task(task)(config=dict(cfg)).evaluate(model, tok, dataset=data, cache=cache)
    own = get_task(task)(config=dict(cfg, use_cache=False)).evaluate(model, tok, dataset=data,
                                                                    cache=None)
    assert "error" not in shared, shared
    assert shared["_meta_points_readout"] == "last_token_of_20_word_window"
    assert shared["_meta_window_set_hash"] == window_set(data, 40).digest
    assert shared["_meta_n_points"] == 40

    def strip(r):
        return {k: v for k, v in r.items() if not k.endswith(("_source", "_cache_rejected"))
                and k not in ("_meta_points_source",)}
    a, b = strip(shared), strip(own)
    assert a.keys() == b.keys()
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, float) and isinstance(vb, float) and va != va and vb != vb:
            continue
        assert va == vb, (k, va, vb)


def test_the_passage_route_is_capped_at_the_model_context():
    """Whole passages longer than a 64-position model's context are cut at it."""
    import transformers
    from blme.tasks.topology._points import last_token_points, role_layers
    cfg = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=fd.tokenizer().vocab_size,
                                  n_positions=64)
    model = transformers.GPT2LMHeadModel(cfg).eval()
    tok = fd.tokenizer()
    data = [{"text": " ".join(["word"] * 300)} for _ in range(4)]
    passage = window_options({"points_readout": "passage"})
    sc = sequence_clouds(model, tok, data, None, n=4, layer="all", replace_final_norm=True,
                         token_docs=2, points=passage)
    assert sc.n_points == 4
    pts, info = last_token_points(model, tok, data, None, num_samples=4, layers=role_layers,
                                  points=passage)
    assert max(info["positions"]) == 63
