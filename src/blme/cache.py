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


# The window the shared cache tokenises at. Tasks whose value depends on the
# window MUST compare their own `max_length` against this and self-collect when
# it differs -- `geometry_schatten` declared 128 while every archived run
# silently used this 512 (red-team review, 2026-09-05).
CACHE_TOKENIZER_MAX_LENGTH = 512


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
        # INITIALISED HERE, not created lazily inside populate(). All three
        # were absent on a fresh instance, so a partially populated cache
        # raised AttributeError from get_hidden_states(include_embedding=True)
        # and the substitution flag could not be read at all (adversarial
        # review round 2, 2026-09-04).
        self._embedding_states: Optional[torch.Tensor] = None
        self._last_block_raw: Optional[torch.Tensor] = None
        self._last_block_substituted: bool = False

        # Feature flags — set by populate() or by first get_* call
        self._need_hidden: bool = False
        self._need_attentions: bool = False
        self._need_logits: bool = False
        self._populated: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def populate(
        self,
        need_hidden: bool = True,
        need_attentions: bool = False,
        need_logits: bool = True,
    ) -> None:
        """
        Run the forward pass and populate the cache.

        If already populated, this is a no-op (call :meth:`clear` first
        to force re-computation).

        ``need_logits`` defaults to True so direct/standalone use keeps the
        historic "logits always collected" behaviour; the evaluate() loop passes
        need_logits=False when no logits-consuming task is in the run, avoiding
        the memory cost of caching full-vocab (T x V) logits for every sample.
        """
        if self._populated:
            return

        self._need_hidden = need_hidden or self._need_hidden
        self._need_attentions = need_attentions or self._need_attentions
        self._need_logits = need_logits or self._need_logits
        self._run_forward_pass()

    def get_hidden_states(
        self,
        layer_idx: Union[int, str] = "all",
        num_samples: Optional[int] = None,
        per_sample: bool = False,
        replace_final_norm: bool = False,
        include_embedding: bool = False,
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor], List[torch.Tensor], Dict[int, List[torch.Tensor]]]:
        """
        Return cached hidden states.

        Args:
            layer_idx: ``"all"`` returns one entry per layer.
                       An int returns a single layer's value.
                       Negative indexing is supported (``-1`` = last layer).
            num_samples: Optional cap on the number of samples to include.
                         Uses cached sample lengths to slice tokens.
            per_sample: When True, the flat ``(Σ T_i, D)`` tensor is split
                        back into a ``List[Tensor]`` of per-sample
                        ``(T_i, D)`` chunks using the stored sample lengths.
                        This is what CKA, RSA, LID and matrix-entropy need;
                        the default flat layout silently mixes tokens from
                        different documents.

        Returns:
            Hidden states tensor(s), dict, list, or dict-of-list depending
            on ``layer_idx`` / ``per_sample``.
        """
        if not self._populated:
            self._need_hidden = True
            self.populate(need_hidden=True)

        if self._hidden_states is None:
            return None

        # Opt-in: substitute the deepest block's RAW output for HF's
        # ln_f(block L-1) entry. Off by default so existing consumers are
        # untouched; a task that profiles BLOCKS (rather than the raw tuple)
        # asks for it explicitly.
        _hs = self._hidden_states
        if include_embedding and self._embedding_states is not None:
            # Shift by one so the embedding is index 0 and block i is i+1.
            _hs = {0: self._embedding_states}
            for _k, _v in self._hidden_states.items():
                _hs[_k + 1] = _v
        # `_last_block_substituted` records whether the swap ACTUALLY HAPPENED,
        # not merely whether a hooked tensor exists. Callers derived
        # `_last_is_block` from `_last_block_raw is not None`, which is a
        # weaker condition: a (30, 9) raw tensor against a (31, 9) cached final
        # tensor is correctly REJECTED here, yet the task still reported
        # `_meta_last_entry_is_block_output = true` (adversarial review round 2,
        # 2026-09-04).
        self._last_block_substituted = False
        if replace_final_norm and self._last_block_raw is not None:
            _keys = sorted(_hs.keys())
            if _keys:
                _lastk = _keys[-1]
                if tuple(self._last_block_raw.shape) == tuple(_hs[_lastk].shape):
                    _hs = dict(_hs)
                    _hs[_lastk] = self._last_block_raw
                    self._last_block_substituted = True

        if layer_idx == "all":
            sliced = self._slice_hidden_states(_hs, num_samples)
            if per_sample:
                return {
                    li: self._split_by_samples(tensor, num_samples)
                    for li, tensor in sliced.items()
                    if tensor is not None
                }
            return sliced

        # Resolve negative / out-of-range index against the highest
        # available layer key (not len()), because populate() may leave
        # gaps if some forward passes failed mid-run.
        keys = sorted(_hs.keys())
        if not keys:
            return [] if per_sample else None
        max_layer = keys[-1]
        actual = layer_idx if layer_idx >= 0 else max_layer + 1 + layer_idx
        actual = max(keys[0], min(actual, max_layer))

        sliced = self._slice_hidden_states(
            {actual: _hs.get(actual)}, num_samples
        )
        tensor = sliced.get(actual, None)
        if per_sample:
            return self._split_by_samples(tensor, num_samples)
        return tensor

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
        """Return cached logits, or None if they were not collected
        (populate(need_logits=False))."""
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
        """Return ``(stats, embeddings)`` aligned for next-token prediction.

        This mirrors :func:`collect_prediction_stats` — the non-cache
        helper — which returns, per sample:

            logits : (T-1, V)   logits at positions 0..T-2
            labels : (T-1,)     target token ids at positions 1..T-1
            hidden : (T-1, D)   last-layer hidden states at positions 0..T-2

        Historic bug: this method used to return ``(1, T, V)``/``(1, T)``
        tensors without the shift. Callers that did
        ``torch.cat(stats["logits"], dim=0)`` would get a 3-D tensor and
        comparisons would pair the distribution at position ``t`` with the
        token at position ``t`` (the identity), silently inflating
        calibration and deflating cross-entropy.
        """
        if not self._populated:
            self._need_hidden = True
            self.populate(need_hidden=True)

        from .tasks.common import get_vocab_size, get_embeddings

        raw_logits = self.get_logits(num_samples=num_samples) or []
        raw_labels = self.get_labels(num_samples=num_samples) or []

        # Per-sample last-layer hidden states before any shift.
        hidden_per_sample: List[torch.Tensor] = []
        if self._hidden_states:
            last_layer = max(self._hidden_states.keys())
            last_hidden = self._hidden_states.get(last_layer)
            if last_hidden is not None:
                hidden_per_sample = self._split_by_samples(last_hidden, num_samples)

        shifted_logits: List[torch.Tensor] = []
        shifted_labels: List[torch.Tensor] = []
        shifted_hidden: List[torch.Tensor] = []
        have_hidden = bool(hidden_per_sample)

        for i, (lg, lb) in enumerate(zip(raw_logits, raw_labels)):
            # Logits arrive as (1, T, V); labels as (1, T).
            if lg.ndim != 3 or lb.ndim != 2:
                continue
            T = lg.shape[1]
            if T < 2:
                # Not enough positions to form a shifted pair.
                continue

            # Logits at t predict token at t+1.
            lg_shift = lg[:, :-1, :].reshape(-1, lg.shape[-1]).float()
            lb_shift = lb[:, 1:].reshape(-1).long()

            # Hidden state for the same sample — if missing or the
            # wrong shape, we skip the whole sample so the three lists
            # stay in lock-step. Downstream consumers ``zip`` the three
            # lists together (``consistency.py`` etc.) and silently
            # dropping one would misalign samples.
            h_shift = None
            if have_hidden:
                if i >= len(hidden_per_sample):
                    continue
                h = hidden_per_sample[i]
                if h.ndim != 2 or h.shape[0] < T:
                    continue
                h_shift = h[:T - 1].float()

            shifted_logits.append(lg_shift)
            shifted_labels.append(lb_shift)
            if h_shift is not None:
                shifted_hidden.append(h_shift)

        # Token counts: use unshifted input ids so frequency statistics
        # reflect actual token exposure, matching
        # ``collect_prediction_stats`` (``np.add.at(counts, ids, 1)``).
        # Same rationale as in ``geometry/utils.py``: the *tokenizer* can
        # emit ids larger than the model's ``config.vocab_size`` when
        # special tokens were added post-hoc (e.g. Llama 3 with
        # ``len(tokenizer) = 128 256`` vs ``vocab_size = 128 000``). We
        # size the count array by the maximum id actually observed so
        # downstream consumers never index out of bounds.
        base_vocab = get_vocab_size(self.model) or 50257
        tokenizer_len = int(
            getattr(self.tokenizer, "__len__", lambda: 0)()
        ) if hasattr(self.tokenizer, "__len__") else 0
        vocab_size = max(base_vocab, tokenizer_len)
        # Extend beyond even that if an observed id still exceeds.
        for lbl in raw_labels:
            m = int(lbl.max().item()) + 1 if lbl.numel() else 0
            if m > vocab_size:
                vocab_size = m

        token_counts = np.zeros(vocab_size, dtype=np.float64)
        for lbl in raw_labels:
            for t in lbl.view(-1).tolist():
                if 0 <= t < vocab_size:
                    token_counts[t] += 1

        stats = {
            "logits": shifted_logits,
            "labels": shifted_labels,
            "token_counts": token_counts,
        }
        if shifted_hidden:
            stats["hidden"] = shifted_hidden

        embeddings = None
        try:
            embeddings = get_embeddings(self.model)
        except Exception:
            pass

        return stats, embeddings

    def clear(self) -> None:
        """Release all cached data."""
        self._hidden_states = None
        self._last_block_raw = None
        self._embedding_states = None
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

        # Capture the deepest block's RAW (pre-final-norm) output alongside the
        # HF tuple. HF's last hidden_states entry is ln_f(block L-1), so the
        # deepest block is not otherwise observable. Stored as an ADDITIONAL
        # accumulator -- never substituted into _hidden_states -- so every
        # existing cache consumer reads exactly what it read before, and a task
        # opts in via get_hidden_states(replace_final_norm=True).
        _last_raw_accum: List[torch.Tensor] = []
        _emb_accum: List[torch.Tensor] = []
        _sink: Dict[str, torch.Tensor] = {}
        _hook = None
        if self._need_hidden:
            try:
                from .tasks.common import get_layers as _gl
                _blocks = _gl(self.model)
                if _blocks is not None and len(_blocks) > 0:
                    def _cap_last(module, inputs, output):
                        _sink["h"] = (output[0] if isinstance(output, tuple)
                                      else output).detach()
                    _hook = _blocks[-1].register_forward_hook(_cap_last)
            except Exception as e:  # pragma: no cover - defensive
                logger.info(f"cache: could not hook last block ({e})")

        # try/FINALLY, and this is the WORST-PLACED instance of the leak.
        #
        # `core.py` populates ONE shared cache for a whole run, so a hook
        # left on `layers[-1]` here is paid by EVERY later task on every
        # forward, and the closure pins the last activation tensor alive.
        # Measured: raise on the second forward during `populate()` and
        # `layers[-1]._forward_hooks` goes 0 -> 1 permanently, exposed via
        # geometry_procrustes_linearity, geometry_trajectory_curvature and
        # geometry_vendi_score (independent Opus review, 2026-09-06).
        #
        # `tests/tasks/test_forward_hook_leaks.py` STRUCTURALLY CANNOT see
        # it: its discovery maps `inspect.getfile(cls)` over the task
        # registry, and `cache.py` defines no task class. The check written
        # to replace a hand-maintained list had a hand-list-shaped hole one
        # indirection further out.
        try:
            self._sample_lengths = []
            with torch.no_grad():
                for text in tqdm(samples, desc="Caching model outputs", unit="sample"):
                    inputs = self.tokenizer(
                        text, return_tensors="pt", truncation=True,
                        max_length=CACHE_TOKENIZER_MAX_LENGTH,
                    ).to(device)
                    # Skip empty/whitespace docs: a 0-token forward pass crashes
                    # ("cannot reshape tensor of 0 elements"); mirrors the
                    # non-cache path's skip (audit 2026-07-16).
                    if "input_ids" in inputs and inputs["input_ids"].shape[1] < 1:
                        continue
                    if "input_ids" in inputs:
                        seq_len = inputs["input_ids"].shape[1]
                        self._sample_lengths.append(int(seq_len))

                    outputs = self.model(**inputs, **forward_kwargs)

                    if _hook is not None and "h" in _sink:
                        _h = _sink.pop("h")
                        _last_raw_accum.append(
                            _h.reshape(-1, _h.shape[-1]).float().cpu())

                    # --- Hidden states ---
                    if self._need_hidden and hasattr(outputs, "hidden_states") and outputs.hidden_states:
                        hs = outputs.hidden_states  # tuple[Tensor], len = layers + 1
                        n_layers = len(hs) - 1  # skip embedding layer
                        # The EMBEDDING output, kept separately. Some transition
                        # metrics need the embedding->block0 step, which is a real
                        # layer transformation but is dropped by the `li + 1`
                        # indexing everywhere else. Stored apart so no existing
                        # consumer's layer numbering shifts.
                        _e = hs[0]
                        _emb_accum.append(
                            _e.reshape(-1, _e.shape[-1]).float().detach().cpu())
                        # post-final-norm: THE CACHE'S OWN STORE. `hs[-1]` was
                        # already substituted above when the caller asked for it,
                        # so what is written here is whichever representation the
                        # caller declared.
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

                    # --- Logits (only when a logits-consuming task needs them) ---
                    if self._need_logits and hasattr(outputs, "logits") and outputs.logits is not None:
                        logits_accum.append(outputs.logits.detach().cpu())

                    # --- Labels ---
                    if "input_ids" in inputs:
                        labels_accum.append(inputs["input_ids"].detach().cpu())

        finally:
            if _hook is not None:
                _hook.remove()

        # Concatenate hidden states
        if hidden_accum:
            self._hidden_states = {
                li: torch.cat(tensors, dim=0) for li, tensors in hidden_accum.items()
            }

        # Deepest block's RAW output, kept SEPARATE from _hidden_states.
        self._last_block_raw = (torch.cat(_last_raw_accum, dim=0)
                                if _last_raw_accum else None)
        self._embedding_states = (torch.cat(_emb_accum, dim=0)
                                  if _emb_accum else None)

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


# The default corpus: WikiText-103 validation paragraphs (one dataset row per
# paragraph) and, for tasks that need long texts, whole validation articles.
WIKITEXT_DATASET = ("Salesforce/wikitext", "wikitext-103-raw-v1", "validation")
WIKITEXT_DOCUMENTS = ("EleutherAI/wikitext_document_level", "wikitext-103-raw-v1")
MIN_PASSAGE_CHARS = 64


def nested_order(n: int) -> List[int]:
    """A fixed order of ``range(n)`` whose every prefix is evenly spread.

    Base-2 van der Corput order scaled to ``n``: the k-th draw is
    ``floor(rev(k) * n / M)`` with ``M`` the smallest power of two >= n and
    ``rev`` the bit reversal over ``log2 M`` bits; repeats are skipped. Every
    index appears once, prefixes nest (``order[:a]`` is a prefix of
    ``order[:b]``), a prefix of length ``2**j`` is exactly the stride sample
    ``{floor(i * n / 2**j)}``, and any prefix of length ``m`` leaves no gap
    wider than ``ceil(2 n / m)`` indices.
    """
    n = int(n)
    if n <= 0:
        return []
    bits = max(1, (n - 1).bit_length())
    M = 1 << bits
    seen = bytearray(n)
    order: List[int] = []
    for k in range(M):
        r = int(format(k, f"0{bits}b")[::-1], 2)
        j = (r * n) // M
        if not seen[j]:
            seen[j] = 1
            order.append(j)
    return order


def _load_split(path: str, name: str, split: str):
    """``datasets.load_dataset`` (no remote code), with a clear error."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "The default corpus needs the `datasets` package; install it or pass "
            "a dataset explicitly.") from e
    try:
        return load_dataset(path, name, split=split)
    except Exception as e:
        raise RuntimeError(
            f"Could not load {path} ({name}, split={split}): {e}. The default corpus "
            f"is not replaced by a fallback; make the dataset available (network or "
            f"HF cache) or pass a dataset explicitly.") from e


def _stripped_texts(rows, field: str) -> List[str]:
    out = []
    for row in rows:
        text = row.get(field) if isinstance(row, dict) else None
        if isinstance(text, str):
            text = text.strip()
            if len(text) >= MIN_PASSAGE_CHARS:
                out.append(text)
    return out


def _wikitext_passages() -> List[str]:
    """Stripped WikiText-103 validation passages of >= MIN_PASSAGE_CHARS characters."""
    path, name, split = WIKITEXT_DATASET
    return _stripped_texts(_load_split(path, name, split), "text")


def default_corpus_size() -> int:
    """How many passages :func:`load_default_corpus` can return."""
    return len(_wikitext_passages())


def load_default_corpus(num_samples: int, diverse: bool = True) -> list:
    """WikiText-103 validation passages as ``[{"text": ...}, ...]``.

    The passages are the rows of ``Salesforce/wikitext`` /
    ``wikitext-103-raw-v1`` / validation with at least ``MIN_PASSAGE_CHARS`` (64)
    characters after stripping, returned STRIPPED: every WikiText row ends in a
    newline token, and on pythia and TinyLlama that token is a massive-activation
    sink (100% of its occurrences exceed 10x the median norm) that stripping
    removes.

    ``diverse=True`` (default) draws in :func:`nested_order`: every draw is
    evenly spread across the split, so a small draw spans many articles, and
    draws nest -- ``load_default_corpus(n)`` is a prefix of
    ``load_default_corpus(m)`` for ``n < m``. ``diverse=False`` returns the
    first ``num_samples`` passages in split order (same filter and stripping).
    WikiText stores articles consecutively, so a short prefix is a handful of
    articles: 20 passages are one article.

    Raises ``ValueError`` if more passages are requested than exist and
    ``RuntimeError`` if the dataset cannot be loaded; there is no fallback text.
    """
    n = int(num_samples)
    if n < 0:
        raise ValueError(f"num_samples must be >= 0, got {num_samples}")
    passages = _wikitext_passages()
    if n > len(passages):
        raise ValueError(
            f"load_default_corpus: {n} passages requested, but WikiText-103 validation "
            f"has {len(passages)} of >= {MIN_PASSAGE_CHARS} characters.")
    idx = nested_order(len(passages))[:n] if diverse else range(n)
    result = [{"text": passages[i]} for i in idx]
    logger.info(f"Loaded {len(result)} passages from WikiText-103 validation "
                f"({'nested evenly spread order' if diverse else 'split-order prefix'})")
    return result


def load_long_documents(
    n: int,
    min_tokens: Optional[int] = None,
    tokenizer: Any = None,
    split: str = "validation",
) -> list:
    """Whole WikiText-103 articles as ``[{"text": ...}, ...]``.

    From ``EleutherAI/wikitext_document_level`` (``wikitext-103-raw-v1``),
    stripped, in :func:`nested_order` over the articles of ``split``, so draws
    nest and spread evenly. With ``min_tokens`` (which needs ``tokenizer``)
    articles shorter than ``min_tokens`` tokens -- counted as the model sees
    them, special tokens included -- are skipped in that order; the result is
    still nested for a fixed tokenizer. Raises ``ValueError`` if fewer than
    ``n`` articles qualify. The validation split holds 60 articles.
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if min_tokens is not None and tokenizer is None:
        raise ValueError("load_long_documents: min_tokens needs a tokenizer")
    path, name = WIKITEXT_DOCUMENTS
    docs = _stripped_texts(_load_split(path, name, split), "page")
    out = []
    for i in nested_order(len(docs)):
        if len(out) >= n:
            break
        text = docs[i]
        if min_tokens is not None:
            n_tok = len(tokenizer(text, verbose=False)["input_ids"])
            if n_tok < int(min_tokens):
                continue
        out.append({"text": text})
    if len(out) < n:
        need = f" of >= {min_tokens} tokens" if min_tokens is not None else ""
        raise ValueError(
            f"load_long_documents: {n} articles requested, but {path} {split} has "
            f"{len(out)}{need}.")
    return out


# Keep backward-compatible private alias
_load_default_corpus = load_default_corpus
