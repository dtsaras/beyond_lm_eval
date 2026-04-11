"""
Correlation analysis for the BLME benchmark study.

Computes:
  1. Univariate Spearman correlations between each feature and each benchmark
  2. Partial correlations controlling for log(n_params)
  3. Benjamini-Hochberg FDR correction
  4. LASSO feature selection
  5. PCA on feature matrix for clustering

Outputs:
  results/study_v1/analysis/univariate.csv    (feature x benchmark x rho, p, q)
  results/study_v1/analysis/partial.csv       (same, controlling for log_params)
  results/study_v1/analysis/lasso_features.csv (top LASSO-selected features)
  results/study_v1/analysis/pca_coords.csv    (models x PC1, PC2, PC3)

Usage:
    python scripts/analyze_correlations.py --input-dir results/study_v1
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _load_data(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load aggregated features and feature metadata."""
    agg_path = input_dir / "aggregated.csv"
    meta_path = input_dir / "feature_metadata.csv"
    if not agg_path.exists():
        raise FileNotFoundError(f"{agg_path} not found. Run aggregate_results.py first.")
    agg = pd.read_csv(agg_path)
    feat_meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
    return agg, feat_meta


def _bh_correction(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(pvalues, dtype=np.float64)
    valid = np.isfinite(p)
    n_valid = int(valid.sum())
    q = np.full_like(p, np.nan)
    if n_valid == 0:
        return q
    p_valid = p[valid]
    order = np.argsort(p_valid)
    ranks = np.arange(1, n_valid + 1)
    q_sorted = p_valid[order] * n_valid / ranks
    # Enforce monotonicity
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_valid = np.empty(n_valid)
    q_valid[order] = np.minimum(q_sorted, 1.0)
    q[valid] = q_valid
    return q


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """Partial Spearman correlation of x, y controlling for z.
    Done by ranking all three, then computing partial Pearson correlation on ranks."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if mask.sum() < 5:
        return np.nan, np.nan
    x, y, z = x[mask], y[mask], z[mask]

    # Convert to ranks
    def _rank(a):
        return pd.Series(a).rank().values

    rx, ry, rz = _rank(x), _rank(y), _rank(z)

    # Partial correlation formula: r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))
    r_xy = np.corrcoef(rx, ry)[0, 1]
    r_xz = np.corrcoef(rx, rz)[0, 1]
    r_yz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt(max(1e-12, (1 - r_xz**2) * (1 - r_yz**2)))
    partial_r = (r_xy - r_xz * r_yz) / denom

    # p-value via t-distribution approximation
    n = mask.sum()
    df = n - 3
    if df < 1:
        return float(partial_r), np.nan
    t_stat = partial_r * np.sqrt(df / max(1e-12, 1 - partial_r**2))
    from scipy.stats import t
    p = 2 * (1 - t.cdf(abs(t_stat), df))
    return float(partial_r), float(p)


def _find_feature_columns(agg: pd.DataFrame) -> List[str]:
    """Return the feature (X) column names, excluding metadata and Y variables."""
    exclude_exact = {
        "model", "family", "hf_id", "dtype", "n_gpus", "purpose",
        "d_model", "n_layers", "n_heads", "vocab_size", "n_params_est",
        "n_params_M", "log_n_params", "composite_benchmark",
    }
    exclude_prefixes = ["benchmark_"]

    cols = []
    for c in agg.columns:
        if c in exclude_exact:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        # Must be numeric
        if pd.api.types.is_numeric_dtype(agg[c]):
            cols.append(c)
    return cols


def _find_benchmark_columns(agg: pd.DataFrame) -> List[str]:
    """Return the Y-variable column names."""
    ys = [c for c in agg.columns if c.startswith("benchmark_")]
    if "composite_benchmark" in agg.columns:
        ys.append("composite_benchmark")
    return ys


def run_univariate(agg: pd.DataFrame, feature_cols: List[str],
                    benchmark_cols: List[str]) -> pd.DataFrame:
    """Compute Spearman correlation for every (feature, benchmark) pair."""
    rows = []
    for feat in feature_cols:
        for bench in benchmark_cols:
            x = agg[feat].values.astype(np.float64)
            y = agg[bench].values.astype(np.float64)
            mask = np.isfinite(x) & np.isfinite(y)
            n = int(mask.sum())
            if n < 5:
                rows.append({
                    "feature": feat, "benchmark": bench, "rho": np.nan,
                    "p": np.nan, "n": n,
                })
                continue
            rho, p = spearmanr(x[mask], y[mask])
            rows.append({
                "feature": feat, "benchmark": bench,
                "rho": float(rho) if np.isfinite(rho) else np.nan,
                "p": float(p) if np.isfinite(p) else np.nan,
                "n": n,
            })
    df = pd.DataFrame(rows)
    # BH correction within each benchmark
    for bench in benchmark_cols:
        mask = df["benchmark"] == bench
        df.loc[mask, "q"] = _bh_correction(df.loc[mask, "p"].values)
    return df


def run_partial(agg: pd.DataFrame, feature_cols: List[str],
                benchmark_cols: List[str]) -> pd.DataFrame:
    """Compute partial Spearman correlations controlling for log(n_params)."""
    if "log_n_params" not in agg.columns:
        return pd.DataFrame()

    z = agg["log_n_params"].values.astype(np.float64)
    rows = []
    for feat in feature_cols:
        if feat == "log_n_params":
            continue
        for bench in benchmark_cols:
            x = agg[feat].values.astype(np.float64)
            y = agg[bench].values.astype(np.float64)
            rho, p = _partial_spearman(x, y, z)
            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            rows.append({
                "feature": feat, "benchmark": bench,
                "partial_rho": rho, "p": p, "n": int(mask.sum()),
            })
    df = pd.DataFrame(rows)
    for bench in benchmark_cols:
        mask = df["benchmark"] == bench
        df.loc[mask, "q"] = _bh_correction(df.loc[mask, "p"].values)
    return df


def run_lasso(agg: pd.DataFrame, feature_cols: List[str],
              target: str = "composite_benchmark") -> pd.DataFrame:
    """LASSO regression with cross-validation to find predictive features."""
    try:
        from sklearn.linear_model import LassoCV, RidgeCV
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("sklearn not installed; skipping LASSO")
        return pd.DataFrame()

    if target not in agg.columns:
        return pd.DataFrame()

    # Prep data
    y = agg[target].values.astype(np.float64)
    mask_y = np.isfinite(y)
    if mask_y.sum() < 10:
        print(f"Too few models with {target}; skipping LASSO")
        return pd.DataFrame()

    X = agg[feature_cols].copy()
    # Fill NaN with median per column
    X = X.fillna(X.median(numeric_only=True))
    # Drop columns that are still NaN or constant
    X = X.loc[:, X.nunique() > 1]
    X = X.loc[:, X.notna().all()]

    if X.shape[1] == 0:
        return pd.DataFrame()

    X_arr = X.values.astype(np.float64)
    y_arr = y[mask_y]
    X_arr = X_arr[mask_y]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    # LASSO with CV
    n = X_scaled.shape[0]
    cv_folds = min(5, max(2, n // 4))
    lasso = LassoCV(cv=cv_folds, max_iter=10000, random_state=42)
    lasso.fit(X_scaled, y_arr)

    # Build result
    coefs = lasso.coef_
    nonzero_mask = np.abs(coefs) > 1e-8
    rows = []
    for col, coef, nz in zip(X.columns, coefs, nonzero_mask):
        if nz:
            rows.append({
                "feature": col,
                "coefficient": float(coef),
                "abs_coefficient": float(abs(coef)),
            })
    df = pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False)

    # Also compute R² against baseline (log_n_params alone)
    from sklearn.linear_model import LinearRegression
    baseline_r2 = np.nan
    if "log_n_params" in agg.columns:
        z = agg["log_n_params"].values[mask_y].reshape(-1, 1)
        mask_baseline = np.isfinite(z).ravel()
        if mask_baseline.sum() >= 5:
            baseline = LinearRegression().fit(z[mask_baseline], y_arr[mask_baseline])
            baseline_r2 = baseline.score(z[mask_baseline], y_arr[mask_baseline])

    full_r2 = lasso.score(X_scaled, y_arr)
    print(f"LASSO on {target}: R² = {full_r2:.3f} (baseline log_n_params R² = {baseline_r2:.3f})")
    print(f"  Selected {nonzero_mask.sum()} features out of {X.shape[1]}")
    return df


def _find_base_instruct_pairs(agg: pd.DataFrame) -> List[Tuple[pd.Series, pd.Series, str]]:
    """Locate (base_row, instruct_row, family) tuples by the ``-it`` suffix.

    A model named ``foo-it`` is the instruction-tuned counterpart of ``foo``
    (the base) when both rows exist in the aggregated table. Models with no
    matching base (e.g. ``qwen3.5-27b-it`` for which there is no public
    Qwen3.5-27B base) are silently skipped.
    """
    pairs = []
    name_to_row = {row["model"]: row for _, row in agg.iterrows()}
    for name, it_row in name_to_row.items():
        if not isinstance(name, str) or not name.endswith("-it"):
            continue
        base_name = name[:-3]
        base_row = name_to_row.get(base_name)
        if base_row is None:
            continue
        family = it_row.get("family", "unknown")
        if pd.isna(family):
            family = "unknown"
        pairs.append((base_row, it_row, str(family)))
    return pairs


def run_base_vs_instruct(agg: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Compute paired base-vs-instruct statistics for every feature column.

    For each metric we report:
      * ``n_pairs``                       — pairs with finite values on both sides
      * ``mean_delta`` / ``median_delta`` — instruct − base in raw units
      * ``cohens_d``                      — mean_delta / std_delta (paired effect size)
      * ``std_delta_xmodel``              — mean_delta / (cross-model std of the metric);
                                            this normalises across metrics with very
                                            different scales so a forest plot is meaningful
      * ``sign_agreement``                — fraction of pairs that moved in the same
                                            direction as the mean (1.0 = unanimous);
                                            primary evidence at small n
      * ``wilcoxon_p`` / ``wilcoxon_q``   — paired Wilcoxon signed-rank test +
                                            Benjamini–Hochberg FDR. With n≈6 the test is
                                            low-power; treat sign_agreement as primary.
      * ``n_qwen``/``n_llama``/``n_gemma``— per-family pair counts (transparency for the
                                            n=6 sample dominated by Qwen)

    Output is sorted by |std_delta_xmodel| descending so the largest effects are
    on top.
    """
    pairs = _find_base_instruct_pairs(agg)
    if not pairs:
        return pd.DataFrame()

    # Cross-model std (over the full set of evaluated models) is the natural
    # unit for cross-metric comparison: a delta of "0.5 cross-model stds"
    # means the instruction-tuned model has shifted by half the variation
    # we see across the entire zoo on that metric.
    cross_model_std = agg[feature_cols].std(ddof=1)

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None  # type: ignore

    rows = []
    for feat in feature_cols:
        deltas = []
        per_family: Dict[str, List[float]] = {}
        for base_row, it_row, family in pairs:
            b = base_row.get(feat)
            i = it_row.get(feat)
            if pd.isna(b) or pd.isna(i):
                continue
            d = float(i) - float(b)
            deltas.append(d)
            per_family.setdefault(family, []).append(d)

        n = len(deltas)
        if n < 2:
            continue

        deltas_arr = np.asarray(deltas, dtype=np.float64)
        mean_delta = float(deltas_arr.mean())
        median_delta = float(np.median(deltas_arr))
        std_delta = float(deltas_arr.std(ddof=1)) if n >= 2 else np.nan
        cohens_d = float(mean_delta / std_delta) if std_delta and std_delta > 0 else np.nan

        denom = cross_model_std.get(feat, np.nan)
        if pd.notna(denom) and denom > 0:
            std_delta_xmodel = float(mean_delta / float(denom))
        else:
            std_delta_xmodel = np.nan

        # Sign-agreement: fraction of pairs that moved in the dominant
        # direction. Primary evidence at small n, where Wilcoxon has no power.
        if mean_delta > 0:
            sign_agreement = float((deltas_arr > 0).sum()) / n
        elif mean_delta < 0:
            sign_agreement = float((deltas_arr < 0).sum()) / n
        else:
            sign_agreement = 0.5

        # Wilcoxon signed-rank (small n caveat)
        wilcox_p: float = np.nan
        if wilcoxon is not None and n >= 3 and np.any(deltas_arr != 0):
            try:
                _, wp = wilcoxon(deltas_arr, zero_method="wilcox", alternative="two-sided")
                wilcox_p = float(wp)
            except Exception:
                wilcox_p = np.nan

        rows.append({
            "feature": feat,
            "n_pairs": n,
            "mean_delta": mean_delta,
            "median_delta": median_delta,
            "std_delta": std_delta if not np.isnan(std_delta) else np.nan,
            "std_delta_xmodel": std_delta_xmodel,
            "cohens_d": cohens_d if not np.isnan(cohens_d) else np.nan,
            "sign_agreement": sign_agreement,
            "wilcoxon_p": wilcox_p,
            "n_qwen": len(per_family.get("qwen3.5", [])),
            "n_llama": len(per_family.get("llama3", [])),
            "n_gemma": len(per_family.get("gemma4", [])),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["abs_std_delta_xmodel"] = df["std_delta_xmodel"].abs()
    df = df.sort_values("abs_std_delta_xmodel", ascending=False).reset_index(drop=True)
    df["wilcoxon_q"] = _bh_correction(df["wilcoxon_p"].values)
    return df


def run_pca(agg: pd.DataFrame, feature_cols: List[str], n_components: int = 3) -> pd.DataFrame:
    """PCA on the feature matrix, standardized."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return pd.DataFrame()

    X = agg[feature_cols].fillna(agg[feature_cols].median(numeric_only=True))
    X = X.loc[:, X.nunique() > 1]
    X = X.loc[:, X.notna().all()]
    if X.shape[0] < 3 or X.shape[1] == 0:
        return pd.DataFrame()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    pca = PCA(n_components=min(n_components, X_scaled.shape[0] - 1, X_scaled.shape[1]))
    coords = pca.fit_transform(X_scaled)

    df = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])])
    df["model"] = agg["model"].values
    df["family"] = agg["family"].values if "family" in agg.columns else None
    df["log_n_params"] = agg["log_n_params"].values if "log_n_params" in agg.columns else None
    if "composite_benchmark" in agg.columns:
        df["composite_benchmark"] = agg["composite_benchmark"].values

    # Also store explained variance
    ev = pd.DataFrame({"component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
                       "explained_variance_ratio": pca.explained_variance_ratio_})
    return df, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="results/study_v1")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    agg, feat_meta = _load_data(input_dir)
    feature_cols = _find_feature_columns(agg)
    benchmark_cols = _find_benchmark_columns(agg)

    print(f"Models: {len(agg)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Benchmarks: {len(benchmark_cols)}")

    # 1. Univariate
    print("\n=== Univariate Spearman correlations ===")
    univariate = run_univariate(agg, feature_cols, benchmark_cols)
    univariate.to_csv(output_dir / "univariate.csv", index=False)
    sig = univariate[univariate["q"] < 0.05]
    print(f"  {len(sig)}/{len(univariate)} significant after FDR (q<0.05)")

    # 2. Partial correlations
    print("\n=== Partial correlations (|log_n_params) ===")
    partial = run_partial(agg, feature_cols, benchmark_cols)
    partial.to_csv(output_dir / "partial.csv", index=False)
    sig_partial = partial[partial["q"] < 0.05]
    print(f"  {len(sig_partial)}/{len(partial)} significant after FDR")

    # 3. LASSO
    print("\n=== LASSO feature selection ===")
    lasso = run_lasso(agg, feature_cols)
    if not lasso.empty:
        lasso.to_csv(output_dir / "lasso_features.csv", index=False)
        print(f"  Top 10 LASSO features:")
        for _, row in lasso.head(10).iterrows():
            print(f"    {row['feature']:<50s} {row['coefficient']:+.4f}")

    # 4. Base vs. Instruct paired analysis
    print("\n=== Base vs. Instruct paired analysis ===")
    bvi = run_base_vs_instruct(agg, feature_cols)
    if not bvi.empty:
        bvi.to_csv(output_dir / "base_vs_instruct.csv", index=False)
        n_pairs_max = int(bvi["n_pairs"].max())
        n_features_evaluated = len(bvi)
        unanimous = bvi[bvi["sign_agreement"] >= 0.999]
        print(f"  {n_features_evaluated} features evaluated across up to {n_pairs_max} pairs")
        print(f"  {len(unanimous)} features moved unanimously across all pairs")
        print(f"  Top 10 by |std_delta_xmodel|:")
        for _, r in bvi.head(10).iterrows():
            arrow = "↑" if r["mean_delta"] > 0 else "↓"
            print(f"    {arrow} {r['feature']:<60s} "
                  f"n={int(r['n_pairs'])} "
                  f"std_Δ={r['std_delta_xmodel']:+.2f} "
                  f"agree={r['sign_agreement']:.2f}")
    else:
        print("  No base/instruct pairs found in aggregated.csv")

    # 5. PCA
    print("\n=== PCA ===")
    pca_result = run_pca(agg, feature_cols)
    if isinstance(pca_result, tuple) and not pca_result[0].empty:
        coords, ev = pca_result
        coords.to_csv(output_dir / "pca_coords.csv", index=False)
        ev.to_csv(output_dir / "pca_explained_variance.csv", index=False)
        print(f"  Explained variance: {', '.join(f'{v:.2%}' for v in ev['explained_variance_ratio'])}")

    print(f"\nAnalysis complete. Output: {output_dir}/")


if __name__ == "__main__":
    main()
