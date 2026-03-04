"""
Shared model output cache for BLME.

Runs a single forward pass over a dataset and caches hidden states,
attention weights, and logits so that multiple tasks can re-use them
without redundant computation.

Usage (automatic via core.evaluate):
    cache = ModelOutputCache(model, tokenizer, dataset, num_samples=100)
    # Tasks call:
    hidden = cache.get_hidden_states(layer_idx=-1)
    all_layers = cache.get_hidden_states(layer_idx="all")
    attns = cache.get_attentions()

Usage (standalone):
    cache = ModelOutputCache(model, tokenizer, dataset)
    cache.populate(need_hidden=True, need_attentions=True)
    X = cache.get_hidden_states(layer_idx="all")
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger("blme")


class ModelOutputCache:
    """
    Cache model outputs from a single dataset pass.

    The cache is populated lazily on the first call to any ``get_*`` method,
    or eagerly via :meth:`populate`.  Subsequent calls return cached tensors.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        dataset: Optional[Any] = None,
        num_samples: int = 100,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.num_samples = num_samples

        # Cached data (populated on first access)
        self._hidden_states: Optional[Dict[int, torch.Tensor]] = None
        self._attentions: Optional[Dict[int, List[torch.Tensor]]] = None
        self._logits: Optional[List[torch.Tensor]] = None
        self._labels: Optional[List[torch.Tensor]] = None

        # Feature flags — set by populate() or by first get_* call
        self._need_hidden: bool = False
        self._need_attentions: bool = False
        self._populated: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def populate(
        self,
        need_hidden: bool = True,
        need_attentions: bool = False,
    ) -> None:
        """
        Run the forward pass and populate the cache.

        If already populated, this is a no-op (call :meth:`clear` first
        to force re-computation).
        """
        if self._populated:
            return

        self._need_hidden = need_hidden or self._need_hidden
        self._need_attentions = need_attentions or self._need_attentions
        self._run_forward_pass()

    def get_hidden_states(
        self,
        layer_idx: Union[int, str] = "all",
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Return cached hidden states.

        Args:
            layer_idx: ``"all"`` returns ``{layer: Tensor}``.
                       An int returns a single ``Tensor (N, D)``.
                       Negative indexing is supported (``-1`` = last layer).

        Returns:
            Hidden states tensor(s).
        """
        if not self._populated:
            self._need_hidden = True
            self.populate(need_hidden=True)

        if self._hidden_states is None:
            return None

        if layer_idx == "all":
            return self._hidden_states

        n_layers = len(self._hidden_states)
        actual = layer_idx if layer_idx >= 0 else n_layers + layer_idx
        actual = max(0, min(actual, n_layers - 1))

        return self._hidden_states.get(actual, None)

    def get_attentions(self) -> Optional[Dict[int, List[torch.Tensor]]]:
        """Return cached attention weights ``{layer: [batch_attn, ...]}``."""
        if not self._populated:
            self._need_attentions = True
            self.populate(need_attentions=True)
        return self._attentions

    def get_logits(self) -> Optional[List[torch.Tensor]]:
        """Return cached logits (always collected)."""
        if not self._populated:
            self.populate()
        return self._logits

    def get_labels(self) -> Optional[List[torch.Tensor]]:
        """Return cached label token IDs."""
        if not self._populated:
            self.populate()
        return self._labels

    def get_prediction_stats(self):
        """
        Return (stats, embeddings) matching the signature of
        ``collect_prediction_stats`` for backward-compatible tasks.
        """
        if not self._populated:
            self._need_hidden = True
            self.populate(need_hidden=True)

        # Build stats dict
        logits = self.get_logits() or []
        labels = self.get_labels() or []

        # Compute token_counts from labels
        vocab_size = self.model.config.vocab_size if hasattr(self.model, "config") else 50257
        token_counts = np.zeros(vocab_size, dtype=np.float64)
        for lbl in labels:
            for t in lbl.view(-1).tolist():
                if 0 <= t < vocab_size:
                    token_counts[t] += 1

        stats = {
            "logits": logits,
            "labels": labels,
            "token_counts": token_counts,
        }

        # Embeddings = (V, D) from embedding layer
        embeddings = None
        try:
            from blme.tasks.common import get_embeddings
            embeddings = get_embeddings(self.model)
        except Exception:
            pass

        return stats, embeddings

    def clear(self) -> None:
        """Release all cached data."""
        self._hidden_states = None
        self._attentions = None
        self._logits = None
        self._labels = None
        self._populated = False

    @property
    def is_populated(self) -> bool:
        return self._populated

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_dataset(self) -> list:
        """Get a list of text samples from the dataset."""
        dataset = self.dataset
        if dataset is None:
            # Fall back to a simple default dataset
            dataset = [
                {"text": "The quick brown fox jumps over the lazy dog."},
                {"text": "In machine learning, a neural network is a computational model."},
                {"text": "Large language models have transformed natural language processing."},
            ] * max(1, self.num_samples // 3)

        samples = []
        for i, sample in enumerate(dataset):
            if i >= self.num_samples:
                break
            if isinstance(sample, str):
                samples.append(sample)
            elif isinstance(sample, dict):
                samples.append(sample.get("text", ""))
            else:
                samples.append(str(sample))
        return samples

    def _run_forward_pass(self) -> None:
        """Execute a single dataset pass and store results."""
        samples = self._resolve_dataset()
        if not samples:
            self._populated = True
            return

        device = next(self.model.parameters()).device

        # Accumulators
        hidden_accum: Dict[int, List[torch.Tensor]] = {}  # layer -> list of (T, D)
        attn_accum: Dict[int, List[torch.Tensor]] = {}    # layer -> list of (H, T, T)
        logits_accum: List[torch.Tensor] = []
        labels_accum: List[torch.Tensor] = []

        forward_kwargs: Dict[str, bool] = {}
        if self._need_hidden:
            forward_kwargs["output_hidden_states"] = True
        if self._need_attentions:
            forward_kwargs["output_attentions"] = True

        logger.info(
            f"Cache: running forward pass over {len(samples)} samples "
            f"(hidden={self._need_hidden}, attn={self._need_attentions})"
        )

        with torch.no_grad():
            for text in tqdm(samples, desc="Caching model outputs", unit="sample"):
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512,
                ).to(device)

                outputs = self.model(**inputs, **forward_kwargs)

                # --- Hidden states ---
                if self._need_hidden and hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    hs = outputs.hidden_states  # tuple[Tensor], len = layers + 1
                    n_layers = len(hs) - 1  # skip embedding layer
                    for li in range(n_layers):
                        h = hs[li + 1]  # skip embedding output
                        h_flat = h.reshape(-1, h.shape[-1]).float().detach().cpu()
                        hidden_accum.setdefault(li, []).append(h_flat)

                # --- Attentions ---
                if self._need_attentions and hasattr(outputs, "attentions") and outputs.attentions:
                    for li, attn in enumerate(outputs.attentions):
                        if attn is not None:
                            attn_accum.setdefault(li, []).append(
                                attn.squeeze(0).float().detach().cpu()
                            )

                # --- Logits (always collected) ---
                if hasattr(outputs, "logits") and outputs.logits is not None:
                    logits_accum.append(outputs.logits.detach().cpu())

                # --- Labels ---
                if "input_ids" in inputs:
                    labels_accum.append(inputs["input_ids"].detach().cpu())

        # Concatenate hidden states
        if hidden_accum:
            self._hidden_states = {
                li: torch.cat(tensors, dim=0) for li, tensors in hidden_accum.items()
            }

        # Store attentions as lists (not concatenated — shapes may differ)
        if attn_accum:
            self._attentions = attn_accum

        self._logits = logits_accum if logits_accum else None
        self._labels = labels_accum if labels_accum else None
        self._populated = True

        # Log cache stats
        if self._hidden_states:
            n_layers = len(self._hidden_states)
            n_tokens = self._hidden_states[0].shape[0] if 0 in self._hidden_states else 0
            logger.info(f"Cache populated: {n_layers} layers, {n_tokens} tokens cached")
