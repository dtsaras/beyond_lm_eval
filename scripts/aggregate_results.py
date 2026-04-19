"""
Aggregate BLME + lm_eval results from the study into a single analysis-ready
pandas DataFrame.

Output:
    results/study_v1/aggregated.csv      # (models x features) flat table
    results/study_v1/metadata.csv        # model metadata (family, n_params, etc.)
    results/study_v1/feature_metadata.csv # per-feature metadata (category, tier)

Usage:
    python scripts/aggregate_results.py --input-dir results/study_v1
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_zoo import MODELS

# ── Feature category mapping ────────────────────────────────────────────
TASK_CATEGORY = {
    # Geometry (Tier 1)
    "geometry_spectral": ("geometry", 1),
    "geometry_hubness": ("geometry", 1),
    "geometry_unembedding": ("geometry", 1),
    "geometry_weight_norms": ("geometry", 1),
    "geometry_tokenizer_efficiency": ("geometry", 1),
    # Geometry (Tier 2)
    "geometry_svd": ("geometry", 2),
    "geometry_isoscore": ("geometry", 2),
    "geometry_lid": ("geometry", 2),
    "geometry_collapse": ("geometry", 2),
    "geometry_lipschitz": ("geometry", 2),
    "geometry_intrinsic_dim": ("geometry", 2),
    "geometry_matrix_entropy": ("geometry", 2),
    "geometry_hsic": ("geometry", 2),
    "geometry_rsa": ("geometry", 2),
    "geometry_cka": ("geometry", 2),
    "geometry_correlation_dimension": ("geometry", 2),
    "geometry_positional_decay": ("geometry", 2),
    "geometry_prediction_alignment": ("geometry", 2),
    "geometry_contextualization": ("geometry", 2),
    "geometry_neural_collapse": ("geometry", 2),
    "geometry_schatten": ("geometry", 2),
    # Interpretability
    "interpretability_logit_lens": ("interpretability", 2),
    "interpretability_attention_entropy": ("interpretability", 2),
    "interpretability_attention_rank": ("interpretability", 2),
    "interpretability_induction_heads": ("interpretability", 2),
    "interpretability_head_roles": ("interpretability", 2),
    "interpretability_prediction_entropy": ("interpretability", 2),
    "interpretability_sparsity": ("interpretability", 2),
    "interpretability_superposition": ("interpretability", 2),
    "interpretability_waa": ("interpretability", 2),
    "interpretability_attention_graph": ("interpretability", 2),
    # Causality
    "causality_tracing": ("causality", 2),
    "causality_attention_knockout": ("causality", 2),
    "causality_circuit_quality": ("causality", 2),
    "causality_knowledge_neurons": ("causality", 2),
    "causality_edge_attribution": ("causality", 2),
    # Dynamics
    "dynamics_gradient_flow": ("dynamics", 2),
    "dynamics_sharpness": ("dynamics", 2),
    # Consistency (Tier 3)
    "consistency_position_sensitivity": ("consistency", 3),
    "consistency_format_robustness": ("consistency", 3),
    "consistency_icl_slope": ("consistency", 3),
    # RepE
    "repe_task_vectors": ("repe", 2),
    "repe_concept_separability": ("repe", 2),
    "repe_refusal_direction": ("repe", 2),
    # Y-variables
    "geometry_perplexity": ("y_variable", 2),
    "consistency_calibration": ("y_variable", 3),
}

# ── Feature-specific normalization (divide by d_model, log, etc.) ───────
# (task_key, value_key, normalizer_function_name)
NORMALIZATIONS = {
    # divide by d_model
    ("geometry_svd", "effective_rank"): "div_dmodel",
    ("geometry_svd", "participation_ratio"): "div_dmodel",
    ("geometry_lid", "lid_mean"): "div_dmodel",
    ("geometry_lid", "lid_median"): "div_dmodel",
    ("geometry_intrinsic_dim", "intrinsic_dimension"): "div_dmodel",
    ("geometry_correlation_dimension", "correlation_dimension"): "div_dmodel",
    ("geometry_collapse", "max_erank"): "div_dmodel",
    ("geometry_unembedding", "unembedding_eff_rank"): "div_dmodel",
    # divide by log(d_model)
    ("geometry_matrix_entropy", "mean_matrix_entropy"): "div_log_dmodel",
}


# Matches: layer0, layer_0, layer0_acc, layer_0_aie, layer_27_entropy, ...
# and — only when the whole dict's keys share this shape — bare
# integer indices like "0", "1", "2" (used by
# ``interpretability_waa.layer_waa_alignments``).
_LAYER_RE = re.compile(r"^layer_?(\d+)(?:_(.+))?$")
_BARE_INT_RE = re.compile(r"^\d+$")


def _summarise_list(vals: List[Any], key: str) -> Dict[str, float]:
    """Return mean / std / min / max / slope / q25 / q50 / q75 for a
    list of scalars, keyed under ``key``. Filters non-finite entries.
    Returns an empty dict if no finite values remain."""
    arr = np.asarray([float(v) for v in vals if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    out: Dict[str, float] = {}
    if arr.size == 0:
        return out
    out[f"{key}.mean"] = float(arr.mean())
    out[f"{key}.std"] = float(arr.std())
    out[f"{key}.min"] = float(arr.min())
    out[f"{key}.max"] = float(arr.max())
    if arr.size >= 3:
        xs = np.arange(arr.size)
        slope, _ = np.polyfit(xs, arr, 1)
        out[f"{key}.slope"] = float(slope)
    n = arr.size
    out[f"{key}.q25"] = float(arr[min(n - 1, max(0, int(0.25 * (n - 1))))])
    out[f"{key}.q50"] = float(arr[min(n - 1, max(0, int(0.50 * (n - 1))))])
    out[f"{key}.q75"] = float(arr[min(n - 1, max(0, int(0.75 * (n - 1))))])
    return out


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Flatten a nested dict into {prefix.subkey: scalar} for all scalar
    values.

    Layer-indexed keys (``layer_0``, ``layer_0_aie``, ``layer0_acc``, …)
    at this level are regrouped by suffix and summarised via
    :func:`_summarise_list` so the resulting columns are
    architecture-agnostic (mean / std / slope / q25 / q50 / q75 over
    the layer axis). Without this, models with more layers fill a
    ``layer_31`` column that shorter models leave NaN — inducing a
    spurious depth-bias in any downstream PCA / Lasso / correlation
    analysis.
    """
    out: Dict[str, float] = {}

    # Special case: if *every* key in ``d`` is a bare integer, treat it
    # as a layer-indexed scalar/dict bundle (used by
    # ``interpretability_waa.layer_waa_alignments``).
    if d and all(isinstance(k, str) and _BARE_INT_RE.match(k) for k in d.keys()):
        d = {f"layer_{k}": v for k, v in d.items()}

    # Split keys into layer-indexed groups vs passthrough.
    layer_groups: Dict[str, Dict[int, Any]] = {}
    passthrough: Dict[str, Any] = {}
    for k, v in d.items():
        m = _LAYER_RE.match(k)
        if m is None:
            passthrough[k] = v
            continue
        idx = int(m.group(1))
        suffix = m.group(2) or ""
        layer_groups.setdefault(suffix, {})[idx] = v

    # Emit per-layer summaries for any group with ≥3 layers; push the
    # rest back into passthrough so the ≤2-layer case still appears.
    for suffix, by_idx in layer_groups.items():
        if len(by_idx) < 3:
            for idx, v in by_idx.items():
                restore = f"layer_{idx}_{suffix}" if suffix else f"layer_{idx}"
                passthrough[restore] = v
            continue
        ordered = sorted(by_idx.items(), key=lambda t: t[0])
        vals = [v for _, v in ordered]
        group_prefix = f"{prefix}.{suffix}" if suffix and prefix else (suffix or prefix)
        if all(isinstance(v, (int, float, bool)) for v in vals):
            out.update(_summarise_list(vals, group_prefix))
        elif all(isinstance(v, dict) for v in vals):
            sub_keys: set = set()
            for dv in vals:
                sub_keys.update(dv.keys())
            for sk in sub_keys:
                sub_vals = [dv[sk] for dv in vals
                            if sk in dv and isinstance(dv[sk], (int, float, bool))]
                if len(sub_vals) >= 3:
                    sk_prefix = f"{group_prefix}.{sk}" if group_prefix else sk
                    out.update(_summarise_list(sub_vals, sk_prefix))

    for k, v in passthrough.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, (int, float)):
            out[key] = float(v) if np.isfinite(v) else np.nan
        elif isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        elif isinstance(v, list) and v and all(isinstance(x, (int, float, bool)) for x in v):
            out.update(_summarise_list(v, key))
    return out


def _apply_normalizations(features: Dict[str, float], d_model: int) -> Dict[str, float]:
    """Apply per-feature normalizations (e.g. divide effective rank by d_model)."""
    normed = dict(features)
    log_dmodel = float(np.log(max(2, d_model)))
    for (task, field), norm_type in NORMALIZATIONS.items():
        key = f"{task}.{field}"
        if key in normed and np.isfinite(normed[key]):
            if norm_type == "div_dmodel":
                normed[f"{key}_norm"] = normed[key] / d_model
            elif norm_type == "div_log_dmodel":
                normed[f"{key}_norm"] = normed[key] / log_dmodel
    return normed


def _compute_edg(collapse_result: Dict[str, Any], d_model: int) -> Dict[str, float]:
    """Compute the Effective Dimensionality Gradient and extended features."""
    from scipy.stats import spearmanr

    erank_per_layer = collapse_result.get("erank_per_layer") or []
    if not isinstance(erank_per_layer, list) or len(erank_per_layer) < 3:
        return {}

    ratios = np.asarray([e / d_model for e in erank_per_layer], dtype=np.float64)
    L = len(ratios)
    layers = np.arange(L)

    edg, _ = spearmanr(layers, ratios)
    edg = float(edg) if edg is not None and np.isfinite(edg) else np.nan

    # Early/late phase EDG
    third = max(1, L // 3)
    edg_early, _ = spearmanr(layers[:third], ratios[:third]) if third >= 3 else (np.nan, None)
    edg_late, _ = spearmanr(layers[-third:], ratios[-third:]) if third >= 3 else (np.nan, None)

    # Smoothness
    delta = np.diff(ratios)
    if len(delta) > 0 and np.mean(np.abs(delta)) > 0:
        smoothness = 1.0 - float(np.std(delta) / np.mean(np.abs(delta)))
    else:
        smoothness = np.nan

    return {
        "edg": edg,
        "edg_early": float(edg_early) if edg_early is not None and np.isfinite(edg_early) else np.nan,
        "edg_late": float(edg_late) if edg_late is not None and np.isfinite(edg_late) else np.nan,
        "erank_utilization_first": float(ratios[0]),
        "erank_utilization_last": float(ratios[-1]),
        "compression_smoothness": smoothness,
    }


def _extract_blme_features(blme_dir: Path, model_name: str, d_model: int) -> Optional[Dict[str, float]]:
    """Load a model's BLME results.json and return a flat feature dict."""
    results_path = blme_dir / model_name / "results.json"
    if not results_path.exists():
        return None

    try:
        with open(results_path) as f:
            envelope = json.load(f)
    except Exception as e:
        print(f"  ERROR reading {results_path}: {e}")
        return None

    raw_results = envelope.get("results", {})
    features: Dict[str, float] = {}

    for task_name, task_result in raw_results.items():
        if not isinstance(task_result, dict) or "error" in task_result:
            continue
        flat = _flatten_dict(task_result, prefix=task_name)
        features.update(flat)

    # Compute EDG from geometry_collapse output
    collapse = raw_results.get("geometry_collapse", {})
    if isinstance(collapse, dict) and "error" not in collapse:
        edg_feats = _compute_edg(collapse, d_model)
        features.update(edg_feats)

    # Apply dimension normalizations
    features = _apply_normalizations(features, d_model)

    features["model"] = model_name
    return features


def _extract_lm_eval_scores(lm_eval_dir: Path, model_name: str,
                             extended_dir: Optional[Path] = None) -> Dict[str, float]:
    """Extract benchmark accuracies from lm_eval output JSONs.

    Scans both the original lm_eval/ subdirs (with _mmlu suffix for 5-shot
    MMLU) and the extended comprehensive-benchmark suite at lm_eval_extended/
    (with _<benchname> suffix per benchmark group).
    """
    out: Dict[str, float] = {}

    # Primary lm_eval dir: {model_name}/ (main 0-shot) + {model_name}_mmlu/
    dirs_to_scan = [lm_eval_dir / f"{model_name}",
                    lm_eval_dir / f"{model_name}_mmlu"]
    # Extended benchmarks: {model_name}_{benchmark_tag}/ under lm_eval_extended/
    if extended_dir is not None and extended_dir.exists():
        for sub in extended_dir.iterdir():
            if sub.is_dir() and sub.name.startswith(f"{model_name}_"):
                dirs_to_scan.append(sub)

    for model_subdir in dirs_to_scan:
        if not model_subdir.exists():
            continue
        json_files = list(model_subdir.rglob("results*.json"))
        for jf in json_files:
            try:
                with open(jf) as f:
                    data = json.load(f)
                tasks = data.get("results", {})
                for task_name, task_metrics in tasks.items():
                    if not isinstance(task_metrics, dict):
                        continue
                    # Find the main accuracy / acc_norm metric
                    # Also handle perplexity-style and exact-match metrics for
                    # benchmarks like lambada_openai (acc), drop (em,f1),
                    # triviaqa (exact_match), nq_open (exact_match).
                    for metric_key in ["acc,none", "acc_norm,none",
                                       "em,none", "exact_match,flexible-extract",
                                       "exact_match,strict-match",
                                       "exact_match,get-answer",
                                       "exact_match,remove_whitespace",
                                       "perplexity,none",
                                       "acc", "acc_norm"]:
                        if metric_key in task_metrics:
                            metric_name = metric_key.split(',')[0]
                            col = f"benchmark_{task_name}_{metric_name}"
                            out[col] = float(task_metrics[metric_key])
                            break
            except Exception as e:
                print(f"  ERROR reading {jf}: {e}")
    return out


def _load_model_metadata() -> pd.DataFrame:
    """Build a DataFrame of model metadata from model_zoo + HF config."""
    rows = []
    for m in MODELS:
        rows.append({
            "model": m["name"],
            "family": m["family"],
            "hf_id": m["id"],
            "dtype": m["dtype"],
            "n_gpus": m["n_gpus"],
            "purpose": ",".join(m["purpose"]),
        })
    return pd.DataFrame(rows)


def _fetch_hf_config_sizes(metadata: pd.DataFrame) -> pd.DataFrame:
    """Fetch hidden_size, num_layers, n_params from HF config (best-effort)."""
    try:
        from transformers import AutoConfig
    except ImportError:
        return metadata

    sizes = []
    for _, row in metadata.iterrows():
        row_out = {"model": row["model"]}
        try:
            cfg = AutoConfig.from_pretrained(row["hf_id"], trust_remote_code=True)
            # Handle nested text_config (Gemma 4, Qwen 3.5)
            tc = getattr(cfg, "text_config", cfg)
            row_out["d_model"] = int(getattr(tc, "hidden_size", 0) or 0)
            row_out["n_layers"] = int(getattr(tc, "num_hidden_layers", 0) or 0)
            row_out["n_heads"] = int(getattr(tc, "num_attention_heads", 0) or 0)
            row_out["vocab_size"] = int(getattr(tc, "vocab_size", 0) or 0)
        except Exception as e:
            print(f"  WARN: could not load config for {row['hf_id']}: {e}")
            row_out["d_model"] = 0
            row_out["n_layers"] = 0
            row_out["n_heads"] = 0
            row_out["vocab_size"] = 0
        sizes.append(row_out)
    sizes_df = pd.DataFrame(sizes)

    # Rough parameter count heuristic from size series (can be overridden by hand)
    # n_params ≈ 12 * L * d^2 (transformer rule of thumb)
    sizes_df["n_params_est"] = 12 * sizes_df["n_layers"] * sizes_df["d_model"] ** 2

    return metadata.merge(sizes_df, on="model", how="left")


# Manual parameter counts (authoritative, in millions)
MANUAL_PARAM_COUNTS_M = {
    "gpt2-small": 124, "gpt2-medium": 355, "gpt2-large": 774, "gpt2-xl": 1500,
    "pythia-70m": 70, "pythia-160m": 162, "pythia-410m": 405, "pythia-1b": 1011,
    "pythia-1.4b": 1415, "pythia-2.8b": 2775, "pythia-6.9b": 6857, "pythia-12b": 11847,
    "llama3-1b": 1236, "llama3-1b-it": 1236, "llama3-3b": 3213, "llama3-8b": 8030,
    "qwen3.5-0.8b": 800, "qwen3.5-0.8b-it": 800,
    "qwen3.5-2b": 2000, "qwen3.5-2b-it": 2000,
    "qwen3.5-4b": 4000, "qwen3.5-4b-it": 4000,
    "qwen3.5-9b": 9000, "qwen3.5-9b-it": 9000,
    "qwen3.5-27b-it": 27000,
    "gemma4-e2b": 2300, "gemma4-e4b": 4500, "gemma4-e4b-it": 4500,
    "gemma4-31b": 31000,
    "olmo-1b": 1180, "tinyllama-1.1b": 1100, "phi-2": 2700,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="results/study_v1",
                    help="Study output directory (contains blme/ and lm_eval/)")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write aggregated.csv (default: input-dir)")
    ap.add_argument("--skip-hf-config", action="store_true",
                    help="Skip online HF config fetch (use manual param counts only)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    blme_dir = input_dir / "blme"
    lm_eval_dir = input_dir / "lm_eval"
    # Extended comprehensive benchmark suite (gsm8k, bbh, drop, triviaqa, ...)
    lm_eval_extended_dir = input_dir / "lm_eval_extended"

    # Build model metadata
    print("Loading model metadata...")
    meta = _load_model_metadata()
    if not args.skip_hf_config:
        meta = _fetch_hf_config_sizes(meta)
    meta["n_params_M"] = meta["model"].map(MANUAL_PARAM_COUNTS_M)
    meta["log_n_params"] = np.log(meta["n_params_M"].fillna(1) * 1e6)

    # Extract BLME features
    print("Extracting BLME features...")
    feature_rows: List[Dict[str, float]] = []
    for _, row in meta.iterrows():
        d_model = int(row.get("d_model", 0) or 0)
        if d_model == 0:
            d_model = 1024  # sensible fallback
        feats = _extract_blme_features(blme_dir, row["model"], d_model)
        if feats is None:
            print(f"  SKIP {row['model']} (no results.json)")
            continue
        feature_rows.append(feats)
    print(f"  Extracted {len(feature_rows)} model feature rows")

    features_df = pd.DataFrame(feature_rows)

    # Extract benchmark scores (Y-variables) — scans both lm_eval/ and
    # the extended comprehensive suite in lm_eval_extended/
    print("Extracting lm_eval benchmark scores...")
    benchmark_rows: List[Dict[str, float]] = []
    for _, row in meta.iterrows():
        scores = _extract_lm_eval_scores(lm_eval_dir, row["model"],
                                          extended_dir=lm_eval_extended_dir)
        scores["model"] = row["model"]
        benchmark_rows.append(scores)
    benchmarks_df = pd.DataFrame(benchmark_rows)

    # Merge everything
    aggregated = meta.merge(features_df, on="model", how="left").merge(benchmarks_df, on="model", how="left")

    # ── Post-merge cleaning (from the BLME audit findings) ─────────────
    # Audit timeline:
    #   2026-04-13  (a) hyperparameter constants leaked as features
    #   2026-04-13  (b) dynamics_sharpness.n_params is the parameter count
    #   2026-04-13  (c) gemma4-e4b-it calibration is a chat-template bug
    #   2026-04-17  (d) per-layer absolute-index columns cause depth bias
    #                   (fixed in _flatten_dict above: layer-indexed keys
    #                   now become mean/std/slope/q25/q50/q75 summaries)
    #   2026-04-17  (e) cache shift-by-1 bug fixed ⇒ geometry_perplexity
    #                   is no longer inverted; the __deprecated_inverted
    #                   rename was retired.
    #   2026-04-17  (f) gemma4-e4b-it perplexity ppl_overall=7311 (770× base
    #                   model) — same chat-template / tokenisation bug as
    #                   the calibration issue; null out perplexity for this
    #                   one model so it doesn't dominate Pearson means.

    # (a) Drop features with zero variance across all models (these are
    # hyperparameter constants: n_samples, num_points, sam_rho, n_facts, ...).
    feature_cols = [c for c in aggregated.columns
                    if c not in {"model", "family", "hf_id", "dtype", "n_gpus",
                                 "purpose", "d_model", "n_layers", "n_heads",
                                 "vocab_size", "n_params_est", "n_params_M",
                                 "log_n_params", "composite_benchmark"}
                    and not c.startswith("benchmark_")]
    dropped_constants: List[str] = []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(aggregated[col]):
            continue
        vals = aggregated[col].dropna()
        if len(vals) < 3:
            continue
        # Zero or near-zero coefficient of variation → a constant
        mean = vals.mean()
        std = vals.std(ddof=0)
        cv = std / (abs(mean) + 1e-12)
        if cv < 1e-4 and mean != 0:
            dropped_constants.append(col)
        elif std == 0:
            dropped_constants.append(col)
    if dropped_constants:
        print(f"  Dropping {len(dropped_constants)} constant/hyperparameter features "
              f"(zero variance — these are config values not features)")
        aggregated = aggregated.drop(columns=dropped_constants)

    # (b) Drop dynamics_sharpness.n_params — it leaks model size into the
    # feature matrix (ρ=0.97 with log N_params per audit).
    if "dynamics_sharpness.n_params" in aggregated.columns:
        aggregated = aggregated.drop(columns=["dynamics_sharpness.n_params"])
        print("  Dropped dynamics_sharpness.n_params (confounded with model size)")

    # (c) Null out gemma4-e4b-it's calibration AND perplexity values — both
    # exhibit the same chat-template / tokenisation bug (ECE 20× base;
    # ppl_overall 770× base at 7311 vs 9.56). All other gemma4 variants
    # are fine, so the root cause is the -it tokenizer / chat template,
    # not the metric.
    corrupt_prefixes = ("consistency_calibration.", "geometry_perplexity.")
    gemma_it_bad_cols = [c for c in aggregated.columns
                          if c.startswith(corrupt_prefixes)]
    if gemma_it_bad_cols:
        bad_idx = aggregated["model"] == "gemma4-e4b-it"
        if bad_idx.any():
            n_nulled = 0
            for c in gemma_it_bad_cols:
                before = aggregated.loc[bad_idx, c].notna().sum()
                aggregated.loc[bad_idx, c] = np.nan
                n_nulled += int(before)
            if n_nulled > 0:
                print(f"  Nulled {n_nulled} calibration/perplexity entries for "
                      f"gemma4-e4b-it (chat-template tokenization bug)")

    # (d) Add effective_rank_ratio = geometry_svd.effective_rank / d_model.
    # This is a more discriminative isotropy proxy than IsoScore (which
    # saturates near 1.0 for most models — see audit findings).
    if ("geometry_svd.effective_rank" in aggregated.columns
            and "d_model" in aggregated.columns):
        aggregated["geometry_svd.effective_rank_ratio"] = (
            aggregated["geometry_svd.effective_rank"]
            / aggregated["d_model"].replace(0, np.nan)
        )
        print("  Added geometry_svd.effective_rank_ratio (= effective_rank / d_model)")

    # Compute composite benchmark score
    bench_cols = [c for c in aggregated.columns if c.startswith("benchmark_")]
    if bench_cols:
        # Min-max normalize each benchmark, then mean
        bench_matrix = aggregated[bench_cols].copy()
        for c in bench_cols:
            col = bench_matrix[c]
            valid = col.notna()
            if valid.sum() >= 2:
                mn, mx = col[valid].min(), col[valid].max()
                if mx > mn:
                    bench_matrix[c] = (col - mn) / (mx - mn)
        aggregated["composite_benchmark"] = bench_matrix.mean(axis=1, skipna=True)
    else:
        aggregated["composite_benchmark"] = np.nan

    # Write outputs
    agg_path = output_dir / "aggregated.csv"
    meta_path = output_dir / "metadata.csv"
    aggregated.to_csv(agg_path, index=False)
    meta.to_csv(meta_path, index=False)
    print(f"\nWrote aggregated features: {agg_path} ({len(aggregated)} models x {len(aggregated.columns)} columns)")
    print(f"Wrote metadata: {meta_path}")

    # Feature metadata CSV (for filtering by category/tier)
    feature_meta_rows = []
    for col in features_df.columns:
        if col == "model":
            continue
        # Parse task name from column
        task_name = col.split(".")[0] if "." in col else col
        category, tier = TASK_CATEGORY.get(task_name, ("other", 0))
        feature_meta_rows.append({
            "feature": col,
            "task": task_name,
            "category": category,
            "tier": tier,
        })
    feature_meta = pd.DataFrame(feature_meta_rows)
    feature_meta_path = output_dir / "feature_metadata.csv"
    feature_meta.to_csv(feature_meta_path, index=False)
    print(f"Wrote feature metadata: {feature_meta_path}")

    # Summary
    n_models = len(aggregated)
    n_features = len(features_df.columns) - 1  # minus "model"
    n_benchmarks = len(bench_cols)
    print(f"\n=== Summary ===")
    print(f"  Models: {n_models}")
    print(f"  Features: {n_features}")
    print(f"  Benchmarks: {n_benchmarks}")
    print(f"  Composite benchmark coverage: {aggregated['composite_benchmark'].notna().sum()}/{n_models}")


if __name__ == "__main__":
    main()
