from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from blme.cache import ModelOutputCache

class DiagnosticTask(ABC):
    """
    Abstract base class for all diagnostics tasks in blme.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

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
