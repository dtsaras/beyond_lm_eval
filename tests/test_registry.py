import pytest
from blme.registry import register_task, get_task, list_tasks, _TASK_REGISTRY
from blme.tasks.base import DiagnosticTask

def test_task_registration():
    # Setup
    initial_tasks = list_tasks()
    
    @register_task("test_dummy_task")
    class DummyTask(DiagnosticTask):
        def evaluate(self, model, tokenizer, dataset):
            return {"status": "ok"}
            
    # Check registration
    assert "test_dummy_task" in list_tasks()
    assert get_task("test_dummy_task") == DummyTask
    
    # Check execution
    task = get_task("test_dummy_task")()
    assert task.evaluate(None, None, None) == {"status": "ok"}
    
    # Cleanup (optional, but good practice if sharing state)
    if "test_dummy_task" in _TASK_REGISTRY:
        del _TASK_REGISTRY["test_dummy_task"]
