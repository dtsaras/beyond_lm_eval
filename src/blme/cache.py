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
        self._sample_lengths: List[int] = []

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
        num_samples: Optional[int] = None,
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Return cached hidden states.

        Args:
            layer_idx: ``"all"`` returns ``{layer: Tensor}``.
                       An int returns a single ``Tensor (N, D)``.
                       Negative indexing is supported (``-1`` = last layer).
            num_samples: Optional cap on the number of samples to include.
                         Uses cached sample lengths to slice tokens.

        Returns:
            Hidden states tensor(s).
        """
        if not self._populated:
            self._need_hidden = True
            self.populate(need_hidden=True)

        if self._hidden_states is None:
            return None

        if layer_idx == "all":
            return self._slice_hidden_states(self._hidden_states, num_samples)

        n_layers = len(self._hidden_states)
        actual = layer_idx if layer_idx >= 0 else n_layers + layer_idx
        actual = max(0, min(actual, n_layers - 1))

        return self._slice_hidden_states({actual: self._hidden_states.get(actual)}, num_samples).get(actual, None)

    def get_attentions(self, num_samples: Optional[int] = None) -> Optional[Dict[int, List[torch.Tensor]]]:
        """Return cached attention weights ``{layer: [batch_attn, ...]}``."""
        if not self._populated:
            self._need_attentions = True
            self.populate(need_attentions=True)
        if self._attentions is None or num_samples is None:
            return self._attentions
        if not self._attentions:
            return self._attentions
        max_samples = min(num_samples, len(next(iter(self._attentions.values()))))
        return {li: attn_list[:max_samples] for li, attn_list in self._attentions.items()}

    def get_logits(self, num_samples: Optional[int] = None) -> Optional[List[torch.Tensor]]:
        """Return cached logits (always collected)."""
        if not self._populated:
            self.populate()
        if self._logits is None or num_samples is None:
            return self._logits
        return self._logits[: min(num_samples, len(self._logits))]

    def get_labels(self, num_samples: Optional[int] = None) -> Optional[List[torch.Tensor]]:
        """Return cached label token IDs."""
        if not self._populated:
            self.populate()
        if self._labels is None or num_samples is None:
            return self._labels
        return self._labels[: min(num_samples, len(self._labels))]

    def get_prediction_stats(self, num_samples: Optional[int] = None):
        """
        Return (stats, embeddings) matching the signature of
        ``collect_prediction_stats`` for backward-compatible tasks.
        """
        if not self._populated:
            self._need_hidden = True
            self.populate(need_hidden=True)

        # Build stats dict
        logits = self.get_logits(num_samples=num_samples) or []
        labels = self.get_labels(num_samples=num_samples) or []

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

        # Add last-layer hidden states if available (per-sample list)
        if self._hidden_states:
            last_layer = max(self._hidden_states.keys())
            last_hidden = self._hidden_states.get(last_layer)
            if last_hidden is not None:
                stats["hidden"] = self._split_by_samples(last_hidden, num_samples)

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
        self._sample_lengths = []
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
            dataset = _load_default_corpus(self.num_samples)

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

        self._sample_lengths = []
        with torch.no_grad():
            for text in tqdm(samples, desc="Caching model outputs", unit="sample"):
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512,
                ).to(device)
                if "input_ids" in inputs:
                    seq_len = inputs["input_ids"].shape[1]
                    self._sample_lengths.append(int(seq_len))

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

    def _slice_hidden_states(
        self,
        hidden_states: Optional[Dict[int, torch.Tensor]],
        num_samples: Optional[int],
    ) -> Dict[int, torch.Tensor]:
        """Slice cached hidden states to the first num_samples (by token count)."""
        if hidden_states is None:
            return {}
        if num_samples is None or not self._sample_lengths:
            return hidden_states
        max_samples = min(num_samples, len(self._sample_lengths))
        token_limit = int(sum(self._sample_lengths[:max_samples]))
        return {
            li: tensor[:token_limit] for li, tensor in hidden_states.items() if tensor is not None
        }

    def _split_by_samples(
        self,
        tensor: torch.Tensor,
        num_samples: Optional[int],
    ) -> List[torch.Tensor]:
        """Split a flat (N, D) tensor into per-sample chunks."""
        if tensor is None or not self._sample_lengths:
            return []
        max_samples = min(num_samples, len(self._sample_lengths)) if num_samples is not None else len(self._sample_lengths)
        lengths = self._sample_lengths[:max_samples]
        chunks = []
        start = 0
        for length in lengths:
            end = start + int(length)
            chunks.append(tensor[start:end])
            start = end
        return chunks


def _load_default_corpus(num_samples: int) -> list:
    """
    Load diverse text passages from WikiText-103 validation split.

    Falls back to hardcoded sentences if WikiText loading fails.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split="validation", trust_remote_code=False,
        )
        passages = [
            {"text": row["text"]}
            for row in ds
            if isinstance(row.get("text"), str) and len(row["text"]) >= 50
        ]
        if passages:
            result = passages[:num_samples]
            logger.info(f"Loaded {len(result)} passages from WikiText-103 validation")
            return result
        logger.warning("WikiText-103 returned no usable passages; using fallback corpus")
    except Exception as e:
        logger.warning(f"Could not load WikiText-103 ({e}); using fallback corpus")

    return [
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "In machine learning, a neural network is a computational model."},
        {"text": "Large language models have transformed natural language processing."},
    ] * max(1, num_samples // 3)
