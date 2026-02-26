import pytest
from unittest.mock import patch, MagicMock
from blme.core import evaluate

@patch("blme.core.load_model_and_tokenizer")
@patch("blme.tasks.benchmarks.run_lm_eval")
def test_evaluate_dispatch(mock_lm_eval, mock_load, mock_model, mock_tokenizer):
    # Setup mocks
    mock_load.return_value = (mock_model, mock_tokenizer)
    mock_lm_eval.return_value = {"lm_eval_metric": 0.5}
    
    # Mock a diagnostic task
    with patch("blme.core.get_task") as mock_get_task:
        mock_task_instance = MagicMock()
        mock_task_instance.evaluate.return_value = {"diag_metric": 1.0}
        mock_task_cls = MagicMock(return_value=mock_task_instance)
        
        # Configure get_task to return our mock for "my_diag" and None for "my_bench"
        def side_effect(name):
            if name == "my_diag":
                return mock_task_cls
            return None
        mock_get_task.side_effect = side_effect
        
        # Also need to mock is_lm_eval_task to return True for "my_bench"
        with patch("blme.tasks.benchmarks.is_lm_eval_task") as mock_is_eval:
             mock_is_eval.side_effect = lambda x: x == "my_bench"
             
             # Execute
             results = evaluate(
                 model="hf",
                 model_args="pretrained=test",
                 tasks=["my_diag", "my_bench"]
             )
             
             # Verify Results
             assert results["config"]["model"] == "hf"
             assert results["my_diag"] == {"diag_metric": 1.0}
             assert results["lm_eval"] == {"lm_eval_metric": 0.5}
             
             # Verify Calls
             mock_task_cls.assert_called_once() # Diagnostic task instantiated
             mock_task_instance.evaluate.assert_called_once() # Evaluated
             mock_lm_eval.assert_called_once() # lm_eval called
