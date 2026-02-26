from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List

class DiagnosticTask(ABC):
    """
    Abstract base class for all diagnostics tasks in blme.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    @abstractmethod
    def evaluate(self, model, tokenizer, dataset) -> Dict[str, Any]:
        """
        Run the evaluation task.
        
        Args:
            model: The LLM to evaluate.
            tokenizer: The tokenizer.
            dataset: Optional dataset (or None if task loads its own).
            
        Returns:
            Dictionary of metrics.
        """
        pass
