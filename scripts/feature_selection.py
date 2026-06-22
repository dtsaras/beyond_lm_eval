"""Shared analysis-column selection helpers.

These rules define the primary predictor set for downstream statistical
analysis. They intentionally exclude target-like, benchmark-derived, size, and
obvious proxy columns so correlation and LASSO scripts use the same guardrails.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence, Set

import pandas as pd


METADATA_COLUMNS: Set[str] = {
    "model",
    "family",
    "hf_id",
    "dtype",
    "n_gpus",
    "purpose",
    "model_path",
    "size_family",
    "is_instruct",
    "num_params",
}

SIZE_ARCHITECTURE_COLUMNS: Set[str] = {
    "d_model",
    "n_layers",
    "n_heads",
    "vocab_size",
    "n_params_est",
    "n_params_M",
    "log_n_params",
}

TARGET_COLUMNS: Set[str] = {
    "composite_benchmark",
}

PERFORMANCE_LIKE_EXACT: Set[str] = {
    "baseline_loss",
    "mean_nll",
    "nll",
    "ppl",
    "bpc",
    "perplexity",
    "accuracy",
    "acc",
    "auroc",
    "auc",
    "exact_match",
    "em",
    "f1",
    "precision",
    "recall",
    "prob",
    "probability",
    "likelihood",
    "logprob",
    "log_prob",
    "loglikelihood",
    "log_likelihood",
    "conditional_likelihood",
    "loss",
    "calibration",
    "ece",
    "bytes_per_token",
    "tokenization_efficiency",
    "tokenizer_efficiency",
}

# Task-local architecture / configuration leaves that confound model depth.
ARCHITECTURE_PROXY_LEAVES: Set[str] = {
    "max_layer",
    "min_layer",
    "best_layer_idx",
    "layers_reported",
    "sample_size",
    "n_words_tracked",
    "num_samples",
    "n_samples",
    "tokens_evaluated",
    "samples_evaluated",
    "num_layers",
    "num_heads",
    "hidden_size",
    "intermediate_size",
}

LEAKY_PREFIXES = (
    "benchmark_",
    "geometry_perplexity.",
    "consistency_calibration.",
)

TOKENIZER_PROXY_TERMS = (
    "tokenizer",
    "tokenisation",
    "tokenization",
)

PERFORMANCE_LIKE_SUBSTRINGS: Sequence[str] = (
    "loss_ablate",
    "sam_perturbed_loss",
    "mean_nll",
    "baseline_loss",
    "perturbed_loss",
    "conditional_likelihood",
    "logprob",
    "log_prob",
    "loglikelihood",
    "log_likelihood",
    "perplexity",
    "calibration",
    "exact_match",
    "linear_probe_accuracy",
    "probe_accuracy",
    "prediction_entropy",
    "tokenizer_efficiency",
    "bytes_per_token",
)

PERFORMANCE_LIKE_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\.acc(?:\.|$|_)"),
    re.compile(r"\.accuracy(?:\.|$|_)"),
    re.compile(r"\.auroc(?:\.|$|_)"),
    re.compile(r"\.auc(?:\.|$|_)"),
    re.compile(r"\.nll(?:\.|$|_)"),
    re.compile(r"\.ppl(?:\.|$|_)"),
    re.compile(r"\.loss(?:\.|$|_)"),
    re.compile(r"\.prob(?:\.|$|_)"),
    re.compile(r"\.probability(?:\.|$|_)"),
)

LOWER_IS_BETTER_TERMS = (
    "perplexity",
    "ppl",
    "nll",
    "negative_log_likelihood",
    "loss",
    "bpc",
    "bits_per_char",
    "bits_per_byte",
)

LAYER_SLOPE_SUFFIX = ".slope"


def _leaf(column: str) -> str:
    return column.rsplit(".", 1)[-1].lower()


def _path_segments(column: str) -> list[str]:
    return column.lower().split(".")


def _has_architecture_proxy_segment(column: str) -> bool:
    """Catch namespaced config/size fields like ``geometry_cka.n_layers``."""
    proxies = SIZE_ARCHITECTURE_COLUMNS | ARCHITECTURE_PROXY_LEAVES
    return any(seg in proxies for seg in _path_segments(column))


def is_layer_slope_proxy(column: str) -> bool:
    """Slopes over per-layer profiles use raw ordinal indices (depth confound)."""
    return column.endswith(LAYER_SLOPE_SUFFIX)


def _as_metadata_exclusions(feature_metadata: Optional[pd.DataFrame]) -> Set[str]:
    if feature_metadata is None or feature_metadata.empty:
        return set()
    if "feature" not in feature_metadata.columns:
        return set()

    exclusions: Set[str] = set()
    lower_cols = {str(c).lower(): c for c in feature_metadata.columns}
    for _, row in feature_metadata.iterrows():
        feature = row.get("feature")
        if not isinstance(feature, str):
            continue

        if "primary_eligible" in feature_metadata.columns:
            eligible = row.get("primary_eligible")
            if eligible is False or eligible == 0:
                exclusions.add(feature)
                continue

        category_col = lower_cols.get("category")
        task_col = lower_cols.get("task")
        category = str(row.get(category_col, "")).lower() if category_col else ""
        task = str(row.get(task_col, "")).lower() if task_col else ""
        if category == "y_variable" or task in {"geometry_perplexity", "consistency_calibration"}:
            exclusions.add(feature)
    return exclusions


def find_benchmark_columns(agg: pd.DataFrame) -> list[str]:
    """Return target benchmark columns in stable table order."""
    columns = [c for c in agg.columns if c.startswith("benchmark_")]
    if "composite_benchmark" in agg.columns:
        columns.append("composite_benchmark")
    return columns


def is_lower_is_better_benchmark(column: str) -> bool:
    """Return True for benchmark metrics where lower raw values are better."""
    name = column.lower()
    return any(term in name for term in LOWER_IS_BETTER_TERMS)


def is_excluded_predictor(
    column: str,
    *,
    benchmark_columns: Iterable[str] = (),
    target: Optional[str] = None,
    metadata_exclusions: Iterable[str] = (),
) -> bool:
    """Return True when a numeric column should not be a primary predictor."""
    lower = column.lower()
    leaf = _leaf(column)
    excluded_exact = (
        METADATA_COLUMNS
        | SIZE_ARCHITECTURE_COLUMNS
        | TARGET_COLUMNS
        | PERFORMANCE_LIKE_EXACT
        | ARCHITECTURE_PROXY_LEAVES
    )

    if column in excluded_exact or lower in excluded_exact:
        return True
    if target is not None and column == target:
        return True
    if column in set(benchmark_columns) or column in set(metadata_exclusions):
        return True
    if any(column.startswith(prefix) for prefix in LEAKY_PREFIXES):
        return True
    if any(term in lower for term in TOKENIZER_PROXY_TERMS):
        return True
    if is_layer_slope_proxy(column):
        return True
    if _has_architecture_proxy_segment(column):
        return True
    if any(sub in lower for sub in PERFORMANCE_LIKE_SUBSTRINGS):
        return True
    if any(pat.search(column) for pat in PERFORMANCE_LIKE_PATTERNS):
        return True
    if "prediction_entropy" in lower:
        return True
    if leaf in PERFORMANCE_LIKE_EXACT or leaf in ARCHITECTURE_PROXY_LEAVES:
        return True
    if leaf in SIZE_ARCHITECTURE_COLUMNS:
        return True
    if "logit_lens" in lower and any(term in lower for term in ("acc", "accuracy", "exact_match", "f1")):
        return True
    if ("probing" in lower or "probe" in lower) and any(
        term in lower for term in ("acc", "accuracy", "exact_match", "f1", "auroc", "auc")
    ):
        return True
    if lower.startswith("task_category_") or lower.startswith("y_variable"):
        return True
    return False


def find_predictor_columns(
    agg: pd.DataFrame,
    feature_metadata: Optional[pd.DataFrame] = None,
    *,
    benchmark_columns: Optional[Iterable[str]] = None,
    target: Optional[str] = None,
) -> list[str]:
    """Return conservative numeric predictor columns for primary analyses."""
    benchmark_set = set(find_benchmark_columns(agg) if benchmark_columns is None else benchmark_columns)
    metadata_exclusions = _as_metadata_exclusions(feature_metadata)

    columns: list[str] = []
    for column in agg.columns:
        if is_excluded_predictor(
            column,
            benchmark_columns=benchmark_set,
            target=target,
            metadata_exclusions=metadata_exclusions,
        ):
            continue
        if pd.api.types.is_numeric_dtype(agg[column]):
            columns.append(column)
    return columns


def _exclusion_reason(column: str) -> str:
    if column.startswith("benchmark_"):
        return "benchmark"
    leaf = _leaf(column)
    if leaf in SIZE_ARCHITECTURE_COLUMNS | ARCHITECTURE_PROXY_LEAVES:
        return f"architecture_proxy:{leaf}"
    if is_layer_slope_proxy(column):
        return "layer_slope_proxy"
    lower = column.lower()
    for sub in PERFORMANCE_LIKE_SUBSTRINGS:
        if sub in lower:
            return f"performance_like:{sub}"
    for pat in PERFORMANCE_LIKE_PATTERNS:
        if pat.search(column):
            return f"performance_pattern:{pat.pattern}"
    return "unknown"


def tag_primary_eligibility(features: Iterable[str]) -> pd.DataFrame:
    """Build a feature -> primary_eligible mapping for metadata export."""
    rows = []
    for feat in features:
        if feat == "model":
            continue
        rows.append({
            "feature": feat,
            "primary_eligible": not is_excluded_predictor(feat),
            "exclusion_reason": _exclusion_reason(feat) if is_excluded_predictor(feat) else "",
        })
    return pd.DataFrame(rows)
