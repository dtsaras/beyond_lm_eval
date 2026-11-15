"""A task that fails while parameters are promoted must not leave them promoted.

GPU smoke (gemma-4-E2B and Yi-1.5-34B in bf16, 24 GB cards): the fp32-promoting
dynamics tasks ran out of memory inside fp32_compute, a cast back in its
`finally` ran out of memory too (the traceback still held the block's tensors),
the loop stopped, and every later task failed with "expected mat1 and mat2 to
have the same dtype, but got: float != c10::BFloat16" (29 tasks on gemma-4-E2B).
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from blme.registry import _TASK_REGISTRY, register_task  # noqa: E402
from blme.tasks._precision import fp32_compute  # noqa: E402
from blme.tasks.base import DiagnosticTask  # noqa: E402


def _tiny_llama(dtype=torch.bfloat16):
    cfg = transformers.LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(cfg).eval().to(dtype)


def test_a_failed_cast_back_does_not_stop_the_restore(monkeypatch):
    model = _tiny_llama()
    n_params = sum(1 for _ in model.parameters())
    real_to = torch.Tensor.to
    calls = {"n": 0}

    def flaky_to(self, *a, **k):
        target = a[0] if a and isinstance(a[0], torch.dtype) else k.get("dtype")
        if target == torch.bfloat16 and self.dtype == torch.float32:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("CUDA out of memory (simulated)")
        return real_to(self, *a, **k)

    with pytest.raises(RuntimeError):
        with fp32_compute(model):
            monkeypatch.setattr(torch.Tensor, "to", flaky_to)
            raise torch.cuda.OutOfMemoryError("block ran out of memory (simulated)")
    monkeypatch.setattr(torch.Tensor, "to", real_to)
    left = [n for n, p in model.named_parameters() if p.dtype != torch.bfloat16]
    # The one failed cast stays fp32 on CPU; every other parameter is restored.
    assert len(left) <= 1 and n_params > 2, left


def test_evaluate_restores_dtypes_a_failed_task_left_behind(monkeypatch):
    import blme.core as core
    model = _tiny_llama()
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=__import__("tokenizers").Tokenizer(
            __import__("tokenizers").models.WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]")))
    monkeypatch.setattr(core, "load_model_and_tokenizer", lambda *a, **k: (model, tok))
    seen = {}

    @register_task("zz_test_leaves_fp32")
    class _Leaves(DiagnosticTask):
        def evaluate(self, model, tokenizer, dataset, cache=None):
            p = next(model.parameters())
            p.data = p.data.float()     # promoted, then the task dies
            raise torch.cuda.OutOfMemoryError("simulated")

    @register_task("zz_test_reads_dtypes")
    class _Reads(DiagnosticTask):
        def evaluate(self, model, tokenizer, dataset, cache=None):
            seen["dtypes"] = {p.dtype for p in model.parameters()}
            return {"ok": 1.0}

    try:
        out = core.evaluate(model_args="pretrained=unused",
                            tasks=["zz_test_leaves_fp32", "zz_test_reads_dtypes"],
                            dataset=[{"text": "a a a"}], output_dir=None)
    finally:
        _TASK_REGISTRY.pop("zz_test_leaves_fp32", None)
        _TASK_REGISTRY.pop("zz_test_reads_dtypes", None)
    assert seen["dtypes"] == {torch.bfloat16}
    errors = out.get("errors") or {}
    assert "restored before the next task" in str(errors.get("zz_test_leaves_fp32")), errors
