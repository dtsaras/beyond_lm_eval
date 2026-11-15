"""An MoE model whose grouped-matmul experts cannot run on CPU is switched to
the eager experts by the loader and by every task's ``evaluate``; any other
model is left alone, and the envelope names the switch."""
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent / "tasks" / "invariance"))

import _family_d as fd  # noqa: E402
from blme.experts import FALLBACK_ATTR, ensure_runnable_experts, grouped_mm_runs_on_cpu  # noqa: E402
from blme.results import build_results_envelope  # noqa: E402
from blme.tasks.base import DiagnosticTask  # noqa: E402
from blme.tasks.geometry.schatten import SchattenNormTask  # noqa: E402


def _tiny_moe():
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.for_model(
        "qwen2_moe", vocab_size=1000, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, pad_token_id=0, num_experts=4,
        num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=64)
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").eval()


def _forward(model):
    with torch.no_grad():
        return model(input_ids=torch.arange(1, 9).unsqueeze(0)).logits


def _default_experts_run_on_cpu(model):
    return model.config._experts_implementation != "grouped_mm" or grouped_mm_runs_on_cpu()


def test_cpu_moe_forward_runs_after_the_switch_and_only_then_is_recorded():
    model = _tiny_moe()
    needs_switch = not _default_experts_run_on_cpu(model)
    if needs_switch:
        with pytest.raises(NotImplementedError, match="_grouped_mm"):
            _forward(model)
    record = ensure_runnable_experts(model)
    assert torch.isfinite(_forward(model)).all()
    if needs_switch:
        assert record == getattr(model, FALLBACK_ATTR)
        assert (record["from"], record["to"]) == ("grouped_mm", "eager")
        assert model.config._experts_implementation == "eager"
        assert ensure_runnable_experts(model) == record  # idempotent, record kept
    else:
        assert record is None and not hasattr(model, FALLBACK_ATTR)


def test_a_task_called_directly_on_a_cpu_moe_model_runs():
    model = _tiny_moe()
    needs_switch = not _default_experts_run_on_cpu(model)
    r = SchattenNormTask(config={"num_samples": 2, "documents": "dataset"}).evaluate(
        model, fd.tokenizer(), fd.dataset(2))
    assert "error" not in r and r["_meta_schatten_mlp_n_routed_blocks"] == 2
    assert hasattr(model, FALLBACK_ATTR) == needs_switch


def test_every_registered_task_prepares_the_model():
    from blme.core import _register_all_tasks
    from blme.registry import get_task, list_tasks
    _register_all_tasks()
    unprepared = [n for n in list_tasks()
                  if not getattr(get_task(n).evaluate, "_blme_prepares_model", False)]
    assert not unprepared, unprepared
    assert not getattr(DiagnosticTask.evaluate, "_blme_prepares_model", False)


def test_dense_models_and_non_models_are_untouched():
    model = fd.tiny_llama()
    before = model.config._experts_implementation
    assert ensure_runnable_experts(model) is None
    assert model.config._experts_implementation == before and not hasattr(model, FALLBACK_ATTR)
    assert ensure_runnable_experts(object()) is None
    assert ensure_runnable_experts(None) is None


def test_envelope_names_the_switch_only_when_one_happened():
    kw = dict(model_args="pretrained=x", tasks_requested=[], task_results={}, task_errors={}, device="cpu")
    assert "experts_implementation_fallback" not in build_results_envelope(**kw)["config"]
    record = {"from": "grouped_mm", "to": "eager", "reason": "r"}
    env = build_results_envelope(**kw, experts_fallback=record)
    assert env["config"]["experts_implementation_fallback"] == record


def test_blme_evaluate_loads_a_cpu_moe_checkpoint_and_records_the_switch(tmp_path):
    import blme
    model = _tiny_moe()
    needs_switch = not _default_experts_run_on_cpu(model)
    model.save_pretrained(tmp_path)
    fd.tokenizer().save_pretrained(tmp_path)
    env = blme.evaluate(
        model_args=f"pretrained={tmp_path},dtype=float32", tasks=["geometry_schatten"], device="cpu",
        task_configs={"geometry_schatten": {"num_samples": 2, "documents": "dataset"}},
        dataset=fd.dataset(2), fail_on_task_error=True)
    assert env["errors"] is None, env["errors"]
    assert env["results"]["geometry_schatten"]["_meta_schatten_mlp_n_routed_blocks"] == 2
    fallback = env["config"].get("experts_implementation_fallback")
    if needs_switch:
        assert (fallback["from"], fallback["to"]) == ("grouped_mm", "eager")
        assert "CPU" in fallback["reason"]
    else:
        assert fallback is None
