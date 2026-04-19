"""Bootstrap confidence intervals for LASSO LOO R² and LOFO R².

For n=32 models with p=730 features we have high CV variance. To report
honest uncertainty on the headline LOO R² = 0.731 / LOFO R² = 0.262,
we bootstrap the model set with replacement (B = 200 resamples by
default) and recompute both R²s inside each bootstrap sample.

Usage:
    python scripts/bootstrap_lasso_r2.py \\
           --input-dir results/study_v2 \\
           --target composite_benchmark \\
           --n-bootstrap 200 \\
           --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut


def _feat_matrix(agg: pd.DataFrame, target: str, bench_names: set):
    """Build (X, feat_names) using the same column-selection logic as
    analyze_correlations.py: drop metadata + all analysed-benchmark
    columns, then keep columns with >1 unique value and zero NaNs after
    median imputation. ``bench_names`` comes from the analysis CSVs so we
    are guaranteed to drop exactly the 68 benchmark targets and nothing
    else.
    """
    meta_cols = {"model", "family", "num_params", "log_n_params",
                 "is_instruct", "n_params_M", "model_path", "size_family",
                 "hf_id", "dtype", "purpose"}
    feats = [c for c in agg.columns
             if c not in meta_cols and c != target and c not in bench_names]
    X = agg[feats].copy()
    # Only keep numeric columns (drop any leftover string cols)
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median(numeric_only=True))
    X = X.loc[:, X.nunique() > 1]
    X = X.loc[:, X.notna().all()]
    return X, list(X.columns)


def _lasso_loo(X_scaled: np.ndarray, y: np.ndarray) -> float:
    preds = np.zeros_like(y)
    loo = LeaveOneOut()
    for tr, te in loo.split(X_scaled):
        m = LassoCV(cv=min(5, max(2, len(tr) // 4)),
                    max_iter=10000, random_state=42)
        m.fit(X_scaled[tr], y[tr])
        preds[te] = m.predict(X_scaled[te])
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _lasso_lofo(X_scaled: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    if len(set(groups)) < 3:
        return float("nan")
    preds = np.zeros_like(y)
    mask = np.zeros_like(y, dtype=bool)
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X_scaled, y, groups=groups):
        if len(tr) < 5:
            continue
        m = LassoCV(cv=min(5, max(2, len(tr) // 4)),
                    max_iter=10000, random_state=42)
        m.fit(X_scaled[tr], y[tr])
        preds[te] = m.predict(X_scaled[te])
        mask[te] = True
    if mask.sum() < 5:
        return float("nan")
    y_e, p_e = y[mask], preds[mask]
    ss_res = float(np.sum((y_e - p_e) ** 2))
    ss_tot = float(np.sum((y_e - y_e.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _baseline_loo(z: np.ndarray, y: np.ndarray) -> float:
    preds = np.zeros_like(y)
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        m = LinearRegression().fit(z[tr].reshape(-1, 1), y[tr])
        preds[i] = float(m.predict(z[i:i + 1].reshape(-1, 1))[0])
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _oob_lasso_r2(X_scaled: np.ndarray, y: np.ndarray, in_bag: np.ndarray) -> float:
    """Out-of-bag R²: fit LASSO on in-bag rows, predict out-of-bag.

    In-bag rows may contain duplicates; OOB rows appear exactly once. This
    gives a non-optimistic test-set estimate because OOB models never
    appear in the training set.
    """
    oob_mask = np.ones(len(y), dtype=bool)
    oob_mask[np.unique(in_bag)] = False
    if oob_mask.sum() < 3:
        return float("nan")
    X_tr = X_scaled[in_bag]
    y_tr = y[in_bag]
    m = LassoCV(cv=min(5, max(2, len(in_bag) // 4)),
                max_iter=10000, random_state=42)
    m.fit(X_tr, y_tr)
    preds = m.predict(X_scaled[oob_mask])
    y_te = y[oob_mask]
    ss_res = float(np.sum((y_te - preds) ** 2))
    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _oob_baseline_r2(z: np.ndarray, y: np.ndarray, in_bag: np.ndarray) -> float:
    """OOB R² for the log(N_params) baseline."""
    oob_mask = np.ones(len(y), dtype=bool)
    oob_mask[np.unique(in_bag)] = False
    if oob_mask.sum() < 3:
        return float("nan")
    m = LinearRegression().fit(z[in_bag].reshape(-1, 1), y[in_bag])
    preds = m.predict(z[oob_mask].reshape(-1, 1))
    y_te = y[oob_mask]
    ss_res = float(np.sum((y_te - preds) ** 2))
    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--target", default="composite_benchmark")
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    agg = pd.read_csv(args.input_dir / "aggregated.csv")
    y = agg[args.target].values.astype(np.float64)
    families = agg.get("family", pd.Series([""] * len(agg))).fillna("").values
    z = agg["log_n_params"].values.astype(np.float64)

    mask = np.isfinite(y) & np.isfinite(z)
    agg = agg[mask].reset_index(drop=True)
    y = y[mask]
    z = z[mask]
    families = families[mask]

    # Load the 68 analysed benchmark names from univariate.csv so we drop
    # the exact same benchmark targets analyze_correlations.py did.
    uni_path = args.input_dir / "analysis" / "univariate.csv"
    if uni_path.exists():
        bench_names = set(pd.read_csv(uni_path)["benchmark"].unique())
    else:
        bench_names = {args.target}
    X_df, feat_names = _feat_matrix(agg, args.target, bench_names)
    X = X_df.values.astype(np.float64)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    n = len(y)
    print(f"n={n} models, p={X_scaled.shape[1]} features, "
          f"{len(set(families))} families")

    # Point estimates
    print("\n── Point estimates ──")
    point_loo = _lasso_loo(X_scaled, y)
    point_lofo = _lasso_lofo(X_scaled, y, families)
    point_base = _baseline_loo(z, y)
    print(f"  LASSO LOO R² = {point_loo:.3f}")
    print(f"  LASSO LOFO R² = {point_lofo:.3f}")
    print(f"  Baseline log(N) LOO R² = {point_base:.3f}")
    print(f"  Gain over baseline = {point_loo - point_base:+.3f}")

    # Out-of-bag bootstrap: for each of B resamples draw n indices with
    # replacement (in-bag), fit LASSO on in-bag, predict on the
    # non-sampled rows (out-of-bag ≈ 37 % of the data per bootstrap).
    # This avoids the optimistic bias of running LOO *inside* a
    # bootstrap sample where duplicates make prediction trivial.
    rng = np.random.default_rng(args.seed)
    oob_lasso, oob_base, oob_gain = [], [], []
    print(f"\n── OOB bootstrap (B={args.n_bootstrap}) ──")
    import time
    t0 = time.time()
    for b in range(args.n_bootstrap):
        in_bag = rng.choice(n, size=n, replace=True)
        r_lasso = _oob_lasso_r2(X_scaled, y, in_bag)
        r_base = _oob_baseline_r2(z, y, in_bag)
        oob_lasso.append(r_lasso)
        oob_base.append(r_base)
        if np.isfinite(r_lasso) and np.isfinite(r_base):
            oob_gain.append(r_lasso - r_base)
        else:
            oob_gain.append(float("nan"))
        if (b + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (b + 1) / elapsed
            eta = (args.n_bootstrap - b - 1) / rate if rate > 0 else 0
            print(f"  {b+1:3d}/{args.n_bootstrap}: "
                  f"OOB_LASSO median={np.nanmedian(oob_lasso):.3f} "
                  f"OOB_base median={np.nanmedian(oob_base):.3f} "
                  f"gain median={np.nanmedian(oob_gain):+.3f}  "
                  f"(ETA {eta:.0f}s)")

    oob_lasso = np.array(oob_lasso)
    oob_base = np.array(oob_base)
    oob_gain = np.array(oob_gain)

    def _ci(xs, lo=2.5, hi=97.5):
        v = xs[np.isfinite(xs)]
        if len(v) == 0:
            return float("nan"), float("nan")
        return float(np.percentile(v, lo)), float(np.percentile(v, hi))

    out = {
        "n": int(n),
        "p": int(X_scaled.shape[1]),
        "n_families": int(len(set(families))),
        "n_bootstrap": args.n_bootstrap,
        "point_estimates": {
            "lasso_loo": float(point_loo),
            "lasso_lofo": float(point_lofo),
            "baseline_loo": float(point_base),
            "gain": float(point_loo - point_base),
        },
        "oob_bootstrap_ci_95": {
            "lasso_r2": {"lo": _ci(oob_lasso)[0], "median": float(np.nanmedian(oob_lasso)), "hi": _ci(oob_lasso)[1]},
            "baseline_r2": {"lo": _ci(oob_base)[0], "median": float(np.nanmedian(oob_base)), "hi": _ci(oob_base)[1]},
            "gain": {"lo": _ci(oob_gain)[0], "median": float(np.nanmedian(oob_gain)), "hi": _ci(oob_gain)[1]},
        },
    }
    out_path = args.input_dir / "analysis" / "bootstrap_ci.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("\n── Point estimates ──")
    for k, v in out["point_estimates"].items():
        print(f"  {k:20s}  {v:+.3f}")
    print("\n── 95 % OOB-bootstrap CIs ──")
    for k, v in out["oob_bootstrap_ci_95"].items():
        print(f"  {k:20s}  CI=[{v['lo']:+.3f}, {v['hi']:+.3f}]  median={v['median']:+.3f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
