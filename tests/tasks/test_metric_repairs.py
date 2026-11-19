"""Regression controls for sampling policy, document-held-out linearity, and attribution labels."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from blme.tasks._tokens import DocList, sink_keep_masks
from blme.tasks.geometry.procrustes_linearity import _document_heldout_profile, _procrustes_similarity

KEY = "procrustes_linearity_heldout_r2_mean"


def _profile(xs, ys, ids, order=None):
    order = list(range(len(xs))) if order is None else order
    layers = {k: DocList([torch.as_tensor(seq[i]) for i in order], doc_ids=[ids[i] for i in order])
              for k, seq in enumerate([xs, ys])}
    return _document_heldout_profile(layers, [0, 1], [0])


def test_procrustes_matches_numpy_on_unseen_documents_and_permuted_chunks():
    rng = np.random.default_rng(41)
    ids = [4, 0, 1, 0, 1, 2, 3, 5, 1]
    x = [rng.normal(size=(4 + i % 3, 3)) for i in range(len(ids))]
    y = [a @ rng.normal(size=(3, 4)) + i for i, a in enumerate(x)]
    result = _profile(x, y, ids)
    def rows(seq, train):
        return np.concatenate([a[1:] for a, i in zip(seq, ids) if (i not in (0, 5)) == train])
    a, b, c, d = rows(x, True), rows(y, True), rows(x, False), rows(y, False)
    pred = (c - a.mean(0)) @ np.linalg.pinv(a - a.mean(0)) @ (b - b.mean(0)) + b.mean(0)
    expected = 1 - np.square(pred - d).sum() / np.square(d - b.mean(0)).sum()
    assert result[KEY] == pytest.approx(expected, abs=1e-12)
    assert _profile(x, y, ids, list(reversed(range(len(ids)))))[KEY] == pytest.approx(expected)
    meta = result["_meta_procrustes_heldout"]
    assert meta["test_documents"] == ["record:0", "record:5"]
    assert not set(meta["train_documents"]) & set(meta["test_documents"])
    assert (meta["train_tokens"], meta["test_tokens"]) == (len(a), len(d))


def test_procrustes_affine_recovery_and_independent_wide_noise():
    rng = np.random.default_rng(4)
    x = [rng.normal(size=(5, 4)) for _ in range(20)]
    weight = rng.normal(size=(4, 4))
    assert _profile(x, [a @ weight + 7 for a in x], list(range(20)))[KEY] == pytest.approx(1.)
    x = [rng.normal(size=(3, 64)) for _ in range(12)]
    y = [rng.normal(size=(3, 64)) for _ in x]
    raw = _procrustes_similarity(np.concatenate([a[1:] for a in x]),
                                 np.concatenate([a[1:] for a in y]), rank_aware=True)
    assert raw == pytest.approx(1.)
    assert _profile(x, y, list(range(len(x))))[KEY] < 0


def test_procrustes_degenerate_and_training_only_sink_thresholds():
    a = torch.tensor([[0.], [1.], [2.], [250.]], dtype=torch.float64)
    b = torch.full((24, 1), 200., dtype=torch.float64)
    masks, _ = sink_keep_masks([[a, b]], reference_chunks=[True, False])
    assert masks[0].tolist() == [False, True, True, False]
    assert not masks[1].any()
    assert _profile([a], [a], [0])["_meta_procrustes_heldout"]["status"] == "fewer than two eligible documents"
    assert np.isnan(_profile([a, a], [a * 0, a * 0], [0, 1])[KEY])
    result = _profile([a, a, a[:0], a[:1]], [a, a, a[:0], a[:1]], [0, 1, 2, 3])
    assert result[KEY] == pytest.approx(1.)
    assert result["_meta_procrustes_heldout"]["train_documents"] == ["record:1"]
    # Row provenance accepts sequence-like IDs, including NumPy arrays and tensors.
    first = DocList([a, a], doc_ids=np.array([7, 8]))
    second = DocList([a, a], doc_ids=torch.tensor([7, 8]))
    assert _document_heldout_profile({0: first, 1: second}, [0, 1], [0])[KEY] == pytest.approx(1.)
    second.doc_ids = [8, 7]
    with pytest.raises(ValueError, match="document order"):
        _document_heldout_profile({0: first, 1: second}, [0, 1], [0])


def test_procrustes_wikitext_articles_are_grouped_before_passage_selection(monkeypatch):
    from blme import cache
    long = "A passage of sufficient length. " * 3
    rows = [" = First = ", long + "a", " = = Sub = = ", long + "b", " = Second = ", long + "c"]
    monkeypatch.setattr(cache, "_load_split", lambda *args: [{"text": s} for s in rows])
    plain = cache.load_default_corpus(3, diverse=False)
    annotated = cache.load_default_corpus(3, diverse=False, include_document_ids=True)
    assert [d["text"] for d in annotated] == [d["text"] for d in plain]
    assert [d["article_id"] for d in annotated] == [0, 0, 1]


def test_sampling_accepts_structured_return_and_restores_rng_and_checkpoint():
    from transformers import GenerationConfig
    from blme.tasks.consistency.self_consistency import SelfConsistencyTask
    class Batch(dict):
        def to(self, device):
            return self
    class Tokenizer:
        eos_token_id = 5
        pad_token_id = 0
        def __call__(self, *args, **kwargs):
            return Batch(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.ones(1, 2))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.generation_config = GenerationConfig(top_k=1, suppress_tokens=[3])
        def generate(self, input_ids, attention_mask, generation_config):
            assert generation_config.top_k == 0 and generation_config.top_p == 1.0
            assert generation_config.num_beams == 1 and generation_config.pad_token_id == 0
            assert self.generation_config.suppress_tokens is None
            assert input_ids.shape == attention_mask.shape
            return SimpleNamespace(sequences=torch.cat([input_ids, torch.tensor([[3], [3], [4]])], 1))
    model = Model()
    checkpoint, rng = model.generation_config, torch.get_rng_state()
    result = SelfConsistencyTask({"seed": 31, "n_samples_per_prompt": 3}).evaluate(model, Tokenizer(), ["p"])
    assert result["mean_first_token_agreement"] == 2 / 3
    assert result["mean_first_token_entropy"] == pytest.approx(-sum(p * np.log(p) for p in [1/3, 2/3]))
    assert model.generation_config is checkpoint and checkpoint.suppress_tokens == [3]
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.fixture
def tiny_checkpoint(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=40, n_embd=12, n_head=2, n_layer=2,
                                           n_positions=32, bos_token_id=1, eos_token_id=39, pad_token_id=0))
    words = Tokenizer(WordLevel({f"t{i}": i for i in range(40)}, unk_token="t0"))
    words.pre_tokenizer = WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=words, unk_token="t0", eos_token="t39", pad_token="t0")
    model.save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    return tmp_path


def test_public_evaluator_grouped_documents_and_attribution_labels(tiny_checkpoint, capsys):
    from blme.cli import _cmd_list_tasks
    from blme.core import evaluate
    import json

    data = [{"text": "", "document_id": "empty", "prompt": "t1 t2", "target": "t3"}]
    for i in range(12):
        prompt = f"t1 t{i+2} t{i+3} t{i+4}"
        data.append({"text": prompt, "prompt": prompt, "target": f"t{i+5}", "document_id": f"d{i%6}"})
    labels = {"causality_knowledge_neurons": "Neuron saliency", "causality_edge_attribution": "Layer attribution"}
    task = "geometry_procrustes_linearity"
    names = [task, *labels]
    _cmd_list_tasks(SimpleNamespace(group="causality", json=True))
    listing = json.loads(capsys.readouterr().out)["tasks"]
    assert {d["name"]: d["display_name"] for d in listing if d["name"] in labels} == labels
    for cache_enabled in (True, False):
        out = evaluate(model_args=f"pretrained={tiny_checkpoint},dtype=float32", device="cpu",
                       tasks=names if cache_enabled else [task], seed=37, dataset=data,
                       cache_num_samples=len(data) if cache_enabled else 0,
                       task_configs={task: {"num_samples": len(data), "use_cache": cache_enabled}})
        assert not out["errors"], out["errors"]
        result = out["results"][task]
        assert np.isfinite(result[KEY]) and np.isfinite(result["procrustes_linearity_mean"])
        meta = result["_meta_procrustes_heldout"]
        assert meta["test_documents"] == ['document:"d0"', 'document:"d5"']
        assert (meta["train_tokens"], meta["test_tokens"]) == (24, 12)
        if cache_enabled:
            saved = result[KEY]
            assert out["config"]["task_labels"] == labels
            assert out["results"]["causality_knowledge_neurons"]["mean_top1_saliency_share"] > 0
            assert len(out["results"]["causality_edge_attribution"]["mean_layer_attribution_profile"]) == 2
            printed = capsys.readouterr().out
            assert all(label in printed for label in labels.values())
        else:
            assert result[KEY] == pytest.approx(saved)


@pytest.mark.parametrize("use_cache", [True, False], ids=["shared", "private"])
@pytest.mark.parametrize("field,dtype", [("document_id", np.int64), ("doc_id", np.int32),
                                         ("article_id", np.uint64)])
def test_procrustes_numpy_parent_ids_never_leak(tiny_checkpoint, field, dtype, use_cache):
    from blme.core import evaluate

    # Two passages per true parent. Use IDs outside the signed range for uint64
    # to catch casts that lose precision or change the identity of a document.
    first = 2**63 + 7 if dtype == np.uint64 else -3
    parents = [first + i // 2 for i in range(12)]
    name = "geometry_procrustes_linearity"
    source_parent = {f"document:{parent}": parent for parent in parents}
    source_parent.update({f"record:{i}": parent for i, parent in enumerate(parents)})
    baseline = None
    for typed_ids in (parents, [dtype(p) for p in parents],
                      [dtype(p) if i % 2 else p for i, p in enumerate(parents)]):
        data = [{"text": f"t1 t{i+2} t{i+3} t{i+4}", field: identifier}
                for i, identifier in enumerate(typed_ids)]
        out = evaluate(model_args=f"pretrained={tiny_checkpoint},dtype=float32", tasks=[name],
                       dataset=data, device="cpu", seed=37, cache_num_samples=len(data) if use_cache else 0,
                       task_configs={name: {"num_samples": len(data), "use_cache": use_cache}})
        assert not out["errors"], out["errors"]
        result = out["results"][name]
        meta = result["_meta_procrustes_heldout"]
        train = {source_parent[g] for g in meta["train_documents"]}
        test = {source_parent[g] for g in meta["test_documents"]}
        assert train.isdisjoint(test) and train | test == set(parents)
        assert (len(train), len(test), meta["train_tokens"], meta["test_tokens"]) == (4, 2, 24, 12)
        observed = (meta, result[KEY])
        if baseline is None:
            baseline = observed
        else:
            assert observed == baseline


@pytest.mark.parametrize("prompt,content", [("t5 t6", [5, 6]), ("t5 t6 t39", [5, 6, 39])],
                         ids=["plain-text", "literal-eos"])
def test_sampling_eos_postprocessor_preserves_prompt(tiny_checkpoint, monkeypatch, prompt, content):
    from tokenizers.processors import TemplateProcessing
    from transformers import AutoTokenizer, GPT2LMHeadModel
    from blme.core import evaluate

    generate = GPT2LMHeadModel.generate
    calls = []

    def capture(model, *args, **kwargs):
        ids, mask = kwargs["input_ids"], kwargs["attention_mask"]
        assert ids.shape == mask.shape and torch.all(mask == 1)
        calls.append(ids.detach().cpu().tolist())
        return generate(model, *args, **kwargs)

    monkeypatch.setattr(GPT2LMHeadModel, "generate", capture)

    name = "consistency_self_consistency"
    config = {"seed": 77, "num_prompts": 1, "n_samples_per_prompt": 32, "max_new_tokens": 1}
    scores = []
    stripped = []
    for suffix in ("", " t39 t39"):
        tok = AutoTokenizer.from_pretrained(tiny_checkpoint)
        tok.backend_tokenizer.post_processor = TemplateProcessing(
            single="t1 $A" + suffix, special_tokens=[("t1", 1), ("t39", 39)])
        tok.save_pretrained(tiny_checkpoint)
        out = evaluate(model_args=f"pretrained={tiny_checkpoint},dtype=float32", tasks=[name], device="cpu",
                       dataset=[{"prompt": prompt}], seed=77, task_configs={name: config})
        assert not out["errors"], out["errors"]
        result = out["results"][name]
        assert calls[-1] == [[1, *content]] * config["n_samples_per_prompt"]
        stripped.append(result.get("_meta_n_trailing_special_stripped"))
        scores.append({k: v for k, v in result.items() if k.startswith("sampling_stability_")})
    assert scores[0] == scores[1] and len(scores[0]) == 4
    assert stripped == [0, 2]


def test_sampling_eos_does_not_create_a_scored_prompt(tiny_checkpoint):
    from tokenizers.processors import TemplateProcessing
    from transformers import AutoTokenizer
    from blme.core import evaluate

    name = "consistency_self_consistency"
    config = {"seed": 77, "num_prompts": 3, "n_samples_per_prompt": 32, "max_new_tokens": 1}
    values = []
    for template in ("$A", "$A t39"):
        tokenizer = AutoTokenizer.from_pretrained(tiny_checkpoint)
        tokenizer.backend_tokenizer.post_processor = TemplateProcessing(single=template,
                                                                       special_tokens=[("t39", 39)])
        tokenizer.save_pretrained(tiny_checkpoint)
        out = evaluate(model_args=f"pretrained={tiny_checkpoint},dtype=float32", tasks=[name], device="cpu",
                       dataset=[{"prompt": ""}, {"prompt": "   "}, {"prompt": "t5 t6"}],
                       task_configs={name: config})
        assert not out["errors"], out["errors"]
        result = out["results"][name]
        assert result["n_prompts"] == 1
        values.append((result["mean_first_token_agreement"], result["mean_first_token_entropy"]))
    assert values[0] == values[1]
