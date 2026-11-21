"""Opt-in precision check: does running a checkpoint in 16-bit arithmetic cost loss?

``check(model_args)`` (CLI: ``blme precision-check``) measures the mean
next-token NLL of the checkpoint loaded in float32 and in a candidate dtype
(bfloat16 by default), on the same fixed text, and recommends float32 when the
candidate's NLL exceeds float32's by more than ``threshold`` nats (0.02).

The measurement is the companion study's precision audit:

* Text: WikiText-103 validation (``Salesforce/wikitext``,
  ``wikitext-103-raw-v1``), the non-empty lines among the first 4,000 rows,
  joined with newlines.
* Rows: the first 4 x 511 tokens of that text (tokenized without special
  tokens), as 4 rows of 511. When the tokenizer prepends a BOS (``tok("a")``
  starts with ``bos_token_id``), each row is that BOS followed by its 511
  tokens (512); otherwise the row is the 511 tokens.
* Model: loaded with ``attn_implementation=eager`` in each dtype, through
  ``load_model_and_tokenizer`` and so through its dtype contract (which
  reloads a checkpoint whose loader ignored the requested dtype). For every
  loader that honours ``dtype`` this is the audit's own
  ``from_pretrained(path, dtype=..., attn_implementation="eager")``.
* Loss: one forward per dtype on the 4 rows, logits cast to float32,
  per-token cross-entropy of every next token, averaged over the finite
  values; the non-finite count is reported beside it.

The rule is: candidate - float32 above the threshold, or any non-finite NLL in
the candidate that float32 does not have, means run float32. A checkpoint's
native dtype is only a hint (``config.precision.warnings`` in a results
envelope): a bfloat16-native checkpoint can still lose loss in bfloat16, which
only this measurement shows.
"""
from __future__ import annotations

import gc
import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger("blme")

DEFAULT_THRESHOLD = 0.02
CANDIDATE_DTYPES = ("bfloat16", "float16")
AUDIT_DATASET = ("Salesforce/wikitext", "wikitext-103-raw-v1", "validation")
AUDIT_TEXT_ROWS = 4000
N_ROWS = 4
ROW_TOKENS = 511


def audit_text() -> str:
    """The audit's text: the non-empty lines of the first 4,000 rows of
    WikiText-103 validation, joined with newlines (item set 0 always)."""
    from .cache import _load_split
    path, name, split = AUDIT_DATASET
    rows = _load_split(path, name, split)["text"][:AUDIT_TEXT_ROWS]
    return "\n".join(t for t in rows if t.strip())


def prepends_bos(tokenizer) -> bool:
    """The audit's BOS rule: a BOS id exists and ``tokenizer("a")`` starts with it."""
    bos = tokenizer.bos_token_id
    return bos is not None and tokenizer("a").input_ids[:1] == [bos]


def audit_rows(tokenizer, text: str) -> Tuple[torch.Tensor, bool]:
    """``(ids, bos_prepended)``: ids of shape (4, 512) with the BOS first, or
    (4, 511) for a tokenizer that does not prepend one."""
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    need = N_ROWS * ROW_TOKENS
    if enc.shape[1] < need:
        raise ValueError(f"the text has {enc.shape[1]} tokens; the check needs {need}")
    enc = enc[:, :need].view(N_ROWS, ROW_TOKENS)
    if prepends_bos(tokenizer):
        return torch.cat([torch.full((N_ROWS, 1), tokenizer.bos_token_id), enc], 1), True
    # The audit writes this case as torch.cat([enc[:, :1], enc], 1)[:, 1:], which is enc.
    return enc, False


def mean_nll(model, ids: torch.Tensor) -> Tuple[float, int, int]:
    """``(mean NLL over finite positions, non-finite count, predictions)``."""
    device = getattr(model, "device", None) or next(model.parameters()).device
    with torch.no_grad():
        logits = model(ids.to(device)).logits.float()
    nll = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        ids[:, 1:].to(logits.device).reshape(-1), reduction="none")
    finite = torch.isfinite(nll)
    return float(nll[finite].mean()), int((~finite).sum()), int(nll.numel())


def recommend(nll_fp32: float, nll_cand: float, nonfinite_fp32: int, nonfinite_cand: int,
              candidate: str, threshold: float = DEFAULT_THRESHOLD) -> Tuple[str, str]:
    """``(dtype to run, reason)`` under the rule."""
    delta = nll_cand - nll_fp32
    if nonfinite_cand > nonfinite_fp32:
        return "float32", (f"{nonfinite_cand} non-finite next-token NLL value(s) in {candidate} "
                           f"against {nonfinite_fp32} in float32")
    if not math.isfinite(delta):
        return "float32", f"the {candidate} - float32 NLL difference is not finite"
    if delta > threshold:
        return "float32", f"{candidate} - float32 = {delta:.4g} nats > {threshold:g}"
    return candidate, f"{candidate} - float32 = {delta:.4g} nats <= {threshold:g}"


def _model_args(args: Dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in args.items())


class _NoSelfReference(logging.Filter):
    """Drops the loader's native-dtype hint, which recommends this very check."""

    def filter(self, record):
        return "blme precision-check" not in record.getMessage()


def check(model_args: str, dtype: str = "bfloat16", threshold: float = DEFAULT_THRESHOLD,
          device: Optional[str] = None, text: Optional[str] = None) -> Dict[str, Any]:
    """Measure the checkpoint's next-token NLL in float32 and in ``dtype``.

    Args:
        model_args: as for ``blme.evaluate`` (``pretrained=...,device_map=...``);
            its ``dtype`` and ``attn_implementation`` are replaced (float32 and
            ``dtype`` in turn, eager attention).
        dtype: the candidate, ``bfloat16`` or ``float16``.
        threshold: nats; ``dtype`` - float32 above it recommends float32.
        device: as for ``blme.evaluate`` (default: cuda if available).
        text: replaces the audit text (for tests); the record says which was used.

    Returns:
        A record with ``nll_float32``, ``nll_<dtype>``, ``delta``, the
        non-finite counts, each load's parameter dtypes and dtype contract,
        ``recommendation`` (the dtype to run) and its ``reason``.
    """
    from .models.dtype_contract import PRECISION_ATTR, dtype_name, floating_dtypes
    from .models.wrapper import load_model_and_tokenizer, parse_model_args

    candidate = dtype_name(dtype)
    if candidate not in CANDIDATE_DTYPES:
        raise ValueError(f"dtype must be one of {CANDIDATE_DTYPES}, got {dtype!r}")
    threshold = float(threshold)
    if not (math.isfinite(threshold) and threshold >= 0):
        raise ValueError(f"threshold must be finite and >= 0, got {threshold}")
    args = parse_model_args(model_args)
    if "pretrained" not in args:
        raise ValueError(f"model_args={model_args!r} does not specify 'pretrained='")
    from ._version import __version__

    rec: Dict[str, Any] = {
        "model_args": model_args,
        "candidate_dtype": candidate,
        "threshold_nats": threshold,
        "method": {
            "text": ("custom" if text is not None else
                     "WikiText-103 validation (Salesforce/wikitext, wikitext-103-raw-v1): the "
                     "non-empty lines of the first 4000 rows, joined with newlines"),
            "rows": N_ROWS,
            "content_tokens_per_row": ROW_TOKENS,
            "attn_implementation": "eager",
            "loss": "mean next-token cross-entropy over finite positions (logits in float32)",
        },
        "blme_version": __version__,
    }
    if text is None:
        text = audit_text()
    ids = None
    for dt in ("float32", candidate):
        load_args = {**args, "dtype": dt, "attn_implementation": "eager"}
        quiet = _NoSelfReference()
        logger.addFilter(quiet)
        try:
            model, tokenizer = load_model_and_tokenizer(_model_args(load_args), device=device)
        finally:
            logger.removeFilter(quiet)
        try:
            if ids is None:
                ids, bos = audit_rows(tokenizer, text)
                rec["method"].update({"bos_prepended": bos, "tokens_per_row": int(ids.shape[1]),
                                      "predictions": int(ids.shape[0] * (ids.shape[1] - 1))})
            nll, nonfinite, _ = mean_nll(model, ids)
            load = getattr(model, PRECISION_ATTR, {}) or {}
            rec[f"nll_{dt}"] = nll
            rec[f"nonfinite_{dt}"] = nonfinite
            rec[f"param_dtypes_{dt}"] = floating_dtypes(model, "parameter")
            rec[f"dtype_contract_{dt}"] = load.get("dtype_contract")
            rec.setdefault("model_class", type(model).__name__)
            rec.setdefault("native_dtype", load.get("native_dtype"))
        finally:
            model = tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.info(f"precision check: {dt} mean NLL {nll:.6f} ({nonfinite} non-finite)")

    rec["delta"] = rec[f"nll_{candidate}"] - rec["nll_float32"]
    rec["recommendation"], rec["reason"] = recommend(
        rec["nll_float32"], rec[f"nll_{candidate}"], rec["nonfinite_float32"],
        rec[f"nonfinite_{candidate}"], candidate, threshold)
    return rec
