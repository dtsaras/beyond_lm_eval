import functools
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List, Optional, TYPE_CHECKING

from blme.experts import ensure_runnable_experts

if TYPE_CHECKING:
    from blme.cache import ModelOutputCache

class DiagnosticTask(ABC):
    """
    Abstract base class for all diagnostics tasks in blme.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def __init_subclass__(cls, **kwargs):
        """Every task's ``evaluate`` first makes the model's MoE experts
        runnable on its device (tests/test_experts_implementation.py), so a
        task called directly on a user's model behaves as one run by
        ``blme.evaluate``."""
        super().__init_subclass__(**kwargs)
        fn = cls.__dict__.get("evaluate")
        if fn is None or getattr(fn, "_blme_prepares_model", False):
            return

        @functools.wraps(fn)
        def evaluate(self, model, *args, **kw):
            ensure_runnable_experts(model)
            return fn(self, model, *args, **kw)

        evaluate._blme_prepares_model = True
        cls.evaluate = evaluate

    @abstractmethod
    def evaluate(self, model, tokenizer, dataset, cache: Optional["ModelOutputCache"] = None) -> Dict[str, Any]:
        """
        Run the evaluation task.
        
        Args:
            model: The LLM to evaluate.
            tokenizer: The tokenizer.
            dataset: Optional dataset (or None if task loads its own).
            cache: Optional shared ModelOutputCache for avoiding redundant
                   forward passes across tasks.
            
        Returns:
            Dictionary of metrics.
        """
        pass
