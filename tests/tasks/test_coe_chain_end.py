"""dynamics_coe: the chain end is a parameter and the other end is recorded.

"final_norm" (default) ends the chain at ln_f(h_L), as the release's
inference.py does; "block_output" ends it at the raw deepest block output.
With the release's first-to-last normaliser the final norm can dominate coe_c
(smollm2-1.7b: 1.64 vs 0.074 at the block output, frozen-protocol panel).
"""
import pytest

transformers = pytest.importorskip("transformers")

NAME = "hf-internal-testing/tiny-random-gpt2"
TEXTS = [{"text": "The river carried silt to the delta, where the farmers planted rice every spring."},
         {"text": "A prime number has exactly two distinct positive divisors, one and itself."}]


def _run(cfg):
    from blme.tasks.dynamics.coe import ChainOfEmbeddingTask as CoETask
    model = transformers.AutoModelForCausalLM.from_pretrained(NAME).eval()
    tok = transformers.AutoTokenizer.from_pretrained(NAME)
    return CoETask(config={"num_samples": 2, "max_new_tokens": 8, **cfg}).evaluate(model, tok, TEXTS)


def test_chain_end_swaps_the_headline_and_the_recorded_variant():
    a = _run({})
    b = _run({"chain_end": "block_output"})
    assert a["_meta_chain_end"] == "final_norm" and b["_meta_chain_end"] == "block_output"
    assert a["_meta_last_entry_is_block_output"] is False and b["_meta_last_entry_is_block_output"] is True
    for k in ("coe_r", "coe_c"):
        assert b[k] == pytest.approx(a[f"_meta_{k}_blockout"], rel=1e-12)
        assert a[k] == pytest.approx(b[f"_meta_{k}_final_norm"], rel=1e-12)


def test_an_unknown_chain_end_is_an_error():
    assert "error" in _run({"chain_end": "middle"})
