"""
Tests for core.evaluate() dispatcher.
"""

import pytest
from unittest.mock import patch, MagicMock
from blme.core import evaluate


@patch("blme.core.load_model_and_tokenizer")
@patch("blme.core.get_task")
def test_evaluate_dispatch(mock_get_task, mock_load, mock_model, mock_tokenizer):
    """Test that evaluate() dispatches to diagnostic tasks correctly."""
    # Setup model loader mock
    mock_load.return_value = (mock_model, mock_tokenizer)

    # Setup a mock diagnostic task
    mock_task_instance = MagicMock()
    mock_task_instance.evaluate.return_value = {"diag_metric": 1.0}
    mock_task_cls = MagicMock(return_value=mock_task_instance)

    # get_task returns our mock task class for "my_diag", None otherwise
    def side_effect(name):
        if name == "my_diag":
            return mock_task_cls
        return None
    mock_get_task.side_effect = side_effect

    # Execute (new signature: no `model=` kwarg)
    results = evaluate(
        model_args="pretrained=test",
        tasks=["my_diag"],
    )

    # Verify envelope structure
    assert results["config"]["model_args"] == "pretrained=test"
    assert results["summary"]["total_tasks"] == 1
    assert results["summary"]["completed_tasks"] == 1
    assert results["summary"]["failed_tasks"] == 0

    # Verify task results are under "results" key
    assert "my_diag" in results["results"]
    assert results["results"]["my_diag"] == {"diag_metric": 1.0}

    # Verify diagnostic task was instantiated and called
    mock_task_cls.assert_called_once()
    mock_task_instance.evaluate.assert_called_once()


@patch("blme.core.load_model_and_tokenizer")
@patch("blme.core.get_task")
def test_evaluate_error_isolation(mock_get_task, mock_load, mock_model, mock_tokenizer):
    """Test that a failing task doesn't crash the entire run."""
    mock_load.return_value = (mock_model, mock_tokenizer)

    # Task that raises
    mock_task_instance = MagicMock()
    mock_task_instance.evaluate.side_effect = RuntimeError("boom")
    mock_task_cls = MagicMock(return_value=mock_task_instance)

    mock_get_task.side_effect = lambda name: mock_task_cls if name == "bad_task" else None

    results = evaluate(
        model_args="pretrained=test",
        tasks=["bad_task"],
    )

    # Task should be in errors, not results
    assert results["summary"]["failed_tasks"] == 1
    assert "bad_task" in results["errors"]
    assert "boom" in results["errors"]["bad_task"]


@patch("blme.core.load_model_and_tokenizer")
@patch("blme.core.get_task")
def test_evaluate_off_main_thread_does_not_crash(mock_get_task, mock_load, mock_model, mock_tokenizer):
    """Audit-V2 infra:7 regression: signal.signal() raises off the main thread.

    The finally block must not raise UnboundLocalError on ``old_handler``; the
    per-task timeout simply degrades to disabled and the run completes.
    """
    import threading

    mock_load.return_value = (mock_model, mock_tokenizer)
    mock_task_instance = MagicMock()
    mock_task_instance.evaluate.return_value = {"diag_metric": 1.0}
    mock_task_cls = MagicMock(return_value=mock_task_instance)
    mock_get_task.side_effect = lambda name: mock_task_cls if name == "my_diag" else None

    box = {}

    def run():
        box["results"] = evaluate(model_args="pretrained=test", tasks=["my_diag"])

    t = threading.Thread(target=run)
    t.start()
    t.join()

    # The run completed and the task succeeded despite signal binding failing.
    assert "results" in box, "evaluate() crashed off the main thread"
    assert box["results"]["summary"]["completed_tasks"] == 1


@patch("blme.core.load_model_and_tokenizer")
@patch("blme.core.get_task")
def test_evaluate_accepts_float_timeout(mock_get_task, mock_load, mock_model, mock_tokenizer):
    """Audit-V2 infra:7 regression: a float per-task timeout must not raise.

    signal.alarm() requires an int; the dispatcher now coerces it.
    """
    mock_load.return_value = (mock_model, mock_tokenizer)
    mock_task_instance = MagicMock()
    mock_task_instance.evaluate.return_value = {"diag_metric": 1.0}
    mock_task_cls = MagicMock(return_value=mock_task_instance)
    mock_get_task.side_effect = lambda name: mock_task_cls if name == "my_diag" else None

    results = evaluate(
        model_args="pretrained=test",
        tasks=["my_diag"],
        task_configs={"my_diag": {"timeout": 1.5}},
    )
    assert results["summary"]["completed_tasks"] == 1


@patch("blme.core.load_model_and_tokenizer")
@patch("blme.core.get_task")
def test_pure_benchmark_run_does_not_load_diagnostic_model(mock_get_task, mock_load):
    """Campaign-3 release blocker: a pure lm_eval run loaded the diagnostic model
    for nothing (run_lm_eval loads its own via model_args). No diagnostic tasks
    -> no diagnostic model load."""
    mock_get_task.return_value = None  # nothing is a diagnostic task
    with patch("blme.tasks.benchmarks.is_lm_eval_task", return_value=True), \
         patch("blme.tasks.benchmarks.run_lm_eval", return_value={"acc": 0.5}):
        results = evaluate(model_args="pretrained=test", tasks=["hellaswag"])
    mock_load.assert_not_called()
    assert results["results"]["_lm_eval"] == {"acc": 0.5}


@patch("blme.core.load_model_and_tokenizer")
@patch("blme.core.get_task")
def test_corpus_sized_to_largest_task_num_samples(mock_get_task, mock_load, mock_model, mock_tokenizer):
    """Campaign-3 release blocker: the default corpus was loaded (cap 200)
    before per-task configs were resolved, so a task requesting num_samples>200
    was silently starved. The corpus must be sized to the largest request."""
    mock_load.return_value = (mock_model, mock_tokenizer)
    inst = MagicMock()
    inst.evaluate.return_value = {"m": 1.0}
    cls = MagicMock(return_value=inst)
    mock_get_task.side_effect = lambda n: cls if n == "my_diag" else None

    with patch("blme.cache.default_corpus_size", return_value=1691), \
         patch("blme.cache.load_default_corpus", return_value=[{"text": "x"}]) as mock_corpus:
        evaluate(
            model_args="pretrained=test",
            tasks=["my_diag"],
            task_configs={"my_diag": {"num_samples": 500}},
        )
    requested = mock_corpus.call_args[0][0]
    assert requested >= 500, f"corpus sized {requested}, expected >= 500"


@patch("blme.core.load_model_and_tokenizer")
@patch("blme.core.get_task")
def test_corpus_request_is_capped_at_the_corpus_size(mock_get_task, mock_load, mock_model, mock_tokenizer):
    """`num_samples` of 5000 (dynamics_stability counts vocabulary rows) must
    not make the strict loader raise: the request is capped at what exists."""
    mock_load.return_value = (mock_model, mock_tokenizer)
    inst = MagicMock()
    inst.evaluate.return_value = {"m": 1.0}
    cls = MagicMock(return_value=inst)
    mock_get_task.side_effect = lambda n: cls if n == "my_diag" else None

    with patch("blme.cache.default_corpus_size", return_value=1691), \
         patch("blme.cache.load_default_corpus", return_value=[{"text": "x"}]) as mock_corpus:
        evaluate(
            model_args="pretrained=test",
            tasks=["my_diag"],
            task_configs={"my_diag": {"num_samples": 5000}},
        )
    assert mock_corpus.call_args[0][0] == 1691


@patch("blme.core.load_model_and_tokenizer")
@patch("blme.core.get_task")
def test_mixed_run_frees_model_before_lm_eval(mock_get_task, mock_load, mock_model, mock_tokenizer):
    """Campaign-3 release blocker: a mixed diagnostic+lm_eval run held two full
    model copies. The diagnostic model is freed before run_lm_eval; the cleanup
    path must run without error and both result sets survive."""
    mock_load.return_value = (mock_model, mock_tokenizer)
    inst = MagicMock()
    inst.evaluate.return_value = {"m": 1.0}
    cls = MagicMock(return_value=inst)
    mock_get_task.side_effect = lambda n: cls if n == "my_diag" else None

    with patch("blme.tasks.benchmarks.is_lm_eval_task", side_effect=lambda n: n == "hellaswag"), \
         patch("blme.tasks.benchmarks.run_lm_eval", return_value={"acc": 0.5}) as mock_run:
        results = evaluate(model_args="pretrained=test", tasks=["my_diag", "hellaswag"])
    mock_run.assert_called_once()
    assert results["results"]["my_diag"] == {"m": 1.0}
    assert results["results"]["_lm_eval"] == {"acc": 0.5}


@patch("blme.core.load_model_and_tokenizer")
@patch("blme.core.get_task")
def test_evaluate_saves_json(mock_get_task, mock_load, mock_model, mock_tokenizer, tmp_path):
    """Test that results.json is saved when output_dir is provided."""
    import json
    import os

    mock_load.return_value = (mock_model, mock_tokenizer)

    mock_task_instance = MagicMock()
    mock_task_instance.evaluate.return_value = {"val": 42}
    mock_task_cls = MagicMock(return_value=mock_task_instance)
    mock_get_task.side_effect = lambda name: mock_task_cls if name == "my_task" else None

    output_dir = str(tmp_path / "results")
    evaluate(
        model_args="pretrained=test",
        tasks=["my_task"],
        output_dir=output_dir,
    )

    result_path = os.path.join(output_dir, "results.json")
    assert os.path.exists(result_path)

    with open(result_path) as f:
        data = json.load(f)
    assert "blme_version" in data
    assert data["results"]["my_task"]["val"] == 42
