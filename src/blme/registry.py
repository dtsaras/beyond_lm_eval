from typing import Type, Dict, Optional
from .tasks.base import DiagnosticTask

_TASK_REGISTRY: Dict[str, Type[DiagnosticTask]] = {}

def register_task(name: str):
    """Decorator to register a task class."""
    def decorator(cls):
        _TASK_REGISTRY[name] = cls
        return cls
    return decorator

def get_task(name: str) -> Optional[Type[DiagnosticTask]]:
    return _TASK_REGISTRY.get(name)

def list_tasks() -> list[str]:
    return list(_TASK_REGISTRY.keys())
