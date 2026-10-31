"""A bundled fallback set is used once, never cycled up to ``num_samples``.

Cycling repeated items: 24 steering probes were the 8 bundled triples three
times, and 24 task-vector pairs weighted four of the five bundled pairs 5x and
the fifth 4x. A repeat adds weight, not information.
"""
import logging


def test_steering_uses_the_bundle_once(caplog):
    from blme.tasks.repe.steering import _FALLBACK_TRIPLES, _probe_samples
    with caplog.at_level(logging.WARNING, logger="blme"):
        samples, using_fallback, err = _probe_samples(None, 24)
    assert err is None and using_fallback
    keys = [(s["text_pos"], s["text_neg"]) for s in samples]
    assert len(keys) == len(set(keys)) == len(_FALLBACK_TRIPLES)
    assert f"not the 24 requested" in caplog.text


def test_steering_below_the_bundle_still_uses_all_of_it():
    from blme.tasks.repe.steering import _FALLBACK_TRIPLES, _probe_samples
    samples, _, _ = _probe_samples(None, 3)
    assert len(samples) == len(_FALLBACK_TRIPLES)


def test_task_vectors_uses_the_bundle_once(caplog):
    from blme.tasks.repe.task_vectors import _FALLBACK_PAIRS, _contrastive_samples
    with caplog.at_level(logging.WARNING, logger="blme"):
        samples = _contrastive_samples(None, 24)
    keys = [(s["text_pos"], s["text_neg"]) for s in samples]
    assert len(keys) == len(set(keys)) == len(_FALLBACK_PAIRS)
    assert "not the 24 requested" in caplog.text
    assert len(_contrastive_samples(None, 3)) == 3


def test_knowledge_capacity_uses_the_bundle_once(caplog):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
    from blme.tasks.consistency.knowledge_capacity import KnowledgeCapacityTask
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32,
                                       vocab_size=50257, n_positions=64)).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    with caplog.at_level(logging.WARNING, logger="blme"):
        out = KnowledgeCapacityTask(config={"num_samples": 5}).evaluate(
            model, tokenizer, dataset=None)
    assert "error" not in out, out
    assert out["_meta_n_samples"] == 3 and out["_meta_n_requested"] == 5
    assert "not the 5 requested" in caplog.text
