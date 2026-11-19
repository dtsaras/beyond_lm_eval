"""The envelope records the attention kernel the tasks ran with.

bf16 eager and sdpa logits differ (gemma-4-E2B: mean NLL 2.2434 vs 2.2407 on
one passage), and core.evaluate switches the whole model to eager when a
co-run task needs attention weights, so the same task gives different values
depending on what ran with it and on the transformers version's default.
"""
import pytest

pytest.importorskip("transformers")

TINY = "pretrained=hf-internal-testing/tiny-random-gpt2,dtype=float32"


def test_the_kernel_is_recorded_and_follows_the_co_run_tasks():
    from blme.core import evaluate
    alone = evaluate(model_args=TINY, tasks=["geometry_svd"], device="cpu", seed=1)
    assert alone["config"]["attn_implementation"]["model"] in ("sdpa", "eager")
    with_sharpness = evaluate(
        model_args=TINY, tasks=["geometry_svd", "dynamics_sharpness"], device="cpu", seed=1,
        task_configs={"dynamics_sharpness": {"num_samples": 2, "n_hutchinson": 2}})
    assert with_sharpness["config"]["attn_implementation"]["model"] == "eager"
