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
