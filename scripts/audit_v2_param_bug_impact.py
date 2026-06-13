"""Quantify the impact of the v3 parameter-count bug on the paper's headline claims.

Audit-V2 finding (CRITICAL). scripts/aggregate_results.py:434-435 maps each model to
MANUAL_PARAM_COUNTS_M (which only lists the 32 v2 models) and then does
`np.log(meta["n_params_M"].fillna(1) * 1e6)`. Every v3 model absent from that dict —
26 of 58, including ALL the 70B/72B anchors — is silently assigned 1e6 params, i.e.
log_n_params == log(1e6) == 13.8155. Of those, 14 carry benchmark scores and therefore
enter the regression, and they are disproportionately the highest-capability models.

The log(N_params) baseline is thereby crippled, which manufactures the paper's v3
headline ("baseline collapses 0.43 -> 0.06", "gain +0.72", "LASSO beats baseline on
11/13 families"). This script reproduces the published numbers and recomputes them with
corrected parameter counts.

Run:  OPENBLAS_NUM_THREADS=8 python scripts/audit_v2_param_bug_impact.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.preprocessing import StandardScaler

CSV = Path("results/study_v3/aggregated.csv")
OUT = Path("results/audit_v2/param_bug_impact.json")

# Approximate true parameter counts (millions) for the v3 models missing from
# MANUAL_PARAM_COUNTS_M. Precision is immaterial on a log scale; what matters is
# that a 70B model is ~70000, not 1.
TRUE_M = {
    "llama2-7b": 6738, "llama2-70b": 68976, "llama3-70b": 70554,
    "llama3.1-8b": 8030, "llama3.1-70b": 70554, "llama3.3-70b-it": 70554,
    "qwen2-1.5b": 1544, "qwen2-7b": 7616, "qwen2-72b": 72706,
    "qwen2.5-1.5b": 1544, "qwen2.5-7b": 7616, "qwen2.5-32b": 32764, "qwen2.5-72b": 72706,
    "qwen3-1.7b": 1720, "qwen3-8b": 8190, "qwen3-14b": 14768, "qwen3-32b": 32762,
    "gemma1-2b": 2510, "gemma1-7b": 8538,
    "gemma2-2b": 2614, "gemma2-9b": 9242, "gemma2-27b": 27227,
    "gemma3-1b": 1000, "gemma3-4b": 4300, "gemma3-12b": 12000, "gemma3-27b": 27000,
}

SIZE_PAT = re.compile(
    r"(^|\.)(n_params_est|d_model|n_layers|num_layers|n_layers_analyzed|"
    r"num_layers_analyzed|n_heads|num_heads|vocab_size|sample_size|n_samples|"
    r"traced_layers|hidden_size)(\.|$)"
)
META = {"model", "family", "num_params", "log_n_params", "is_instruct", "n_params_M",
        "model_path", "size_family", "hf_id", "dtype", "purpose"}


def true_z(row):
    if row["model"] in TRUE_M:
        return np.log(TRUE_M[row["model"]] * 1e6)
    if pd.notna(row.get("n_params_M")):
        return np.log(float(row["n_params_M"]) * 1e6)
    return float(row["log_n_params"])


def build_X(agg, bench, drop_size):
    feats = [c for c in agg.columns
             if c not in META and c != "composite_benchmark" and c not in bench]
    X = agg[feats].select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.loc[:, X.nunique() > 1]
    X = X.loc[:, X.notna().all()]
    if drop_size:
        X = X[[c for c in X.columns if not SIZE_PAT.search(c)]]
    return X.values, X.shape[1]


def lasso_loo(Xv, y):
    Xs = StandardScaler().fit_transform(Xv)
    preds = np.zeros_like(y)
    for tr, te in LeaveOneOut().split(Xs):
        m = LassoCV(cv=min(5, max(2, len(tr) // 4)), max_iter=10000, random_state=42)
        m.fit(Xs[tr], y[tr])
        preds[te] = m.predict(Xs[te])
    return _r2(y, preds)


def lasso_lofo(Xv, y, g):
    Xs = StandardScaler().fit_transform(Xv)
    preds = np.zeros_like(y)
    mask = np.zeros_like(y, bool)
    for tr, te in LeaveOneGroupOut().split(Xs, y, g):
        if len(tr) < 5:
            continue
        m = LassoCV(cv=min(5, max(2, len(tr) // 4)), max_iter=10000, random_state=42)
        m.fit(Xs[tr], y[tr])
        preds[te] = m.predict(Xs[te])
        mask[te] = True
    return _r2(y[mask], preds[mask])


def base_loo(z, y):
    preds = np.zeros_like(y)
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        m = LinearRegression().fit(z[tr].reshape(-1, 1), y[tr])
        preds[i] = m.predict(z[i:i + 1].reshape(-1, 1))[0]
    return _r2(y, preds)


def base_lofo(z, y, g):
    preds = np.zeros_like(y)
    mask = np.zeros_like(y, bool)
    for tr, te in LeaveOneGroupOut().split(z.reshape(-1, 1), y, g):
        if len(tr) < 5:
            continue
        m = LinearRegression().fit(z[tr].reshape(-1, 1), y[tr])
        preds[te] = m.predict(z[te].reshape(-1, 1))
        mask[te] = True
    return _r2(y[mask], preds[mask])


def _r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def main():
    agg = pd.read_csv(CSV)
    agg = agg[agg["composite_benchmark"].notna()].reset_index(drop=True)
    y = agg["composite_benchmark"].values.astype(float)
    g = agg["family"].fillna("").values
    bench = [c for c in agg.columns if c.startswith("benchmark_")]

    z_buggy = agg["log_n_params"].values.astype(float)
    z_fixed = agg.apply(true_z, axis=1).values.astype(float)
    sentinel = float(np.log(1e6))
    n_corrupt = int(np.sum(np.abs(z_buggy - sentinel) < 1e-3))

    Xfull, nf = build_X(agg, bench, False)
    Xclean, nc = build_X(agg, bench, True)

    res = {
        "n_models": int(len(y)), "n_families": int(agg["family"].nunique()),
        "n_models_with_corrupt_param_count": n_corrupt,
        "feature_count_full": nf, "feature_count_size_removed": nc,
        "LOO": {
            "lasso_full": lasso_loo(Xfull, y),
            "lasso_size_removed": lasso_loo(Xclean, y),
            "baseline_buggy": base_loo(z_buggy, y),
            "baseline_corrected": base_loo(z_fixed, y),
        },
        "LOFO": {
            "lasso_full": lasso_lofo(Xfull, y, g),
            "lasso_size_removed": lasso_lofo(Xclean, y, g),
            "baseline_buggy": base_lofo(z_buggy, y, g),
            "baseline_corrected": base_lofo(z_fixed, y, g),
        },
    }
    res["LOO"]["gain_published"] = res["LOO"]["lasso_full"] - res["LOO"]["baseline_buggy"]
    res["LOO"]["gain_honest"] = res["LOO"]["lasso_size_removed"] - res["LOO"]["baseline_corrected"]
    res["LOFO"]["gain_published"] = res["LOFO"]["lasso_full"] - res["LOFO"]["baseline_buggy"]
    res["LOFO"]["gain_honest"] = res["LOFO"]["lasso_size_removed"] - res["LOFO"]["baseline_corrected"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nCorrupt param counts in regression: {n_corrupt}/{res['n_models']} models")
    print("LOO  : baseline 0.06(buggy) -> {:.2f}(fixed); honest gain {:+.2f} (paper +0.72)".format(
        res["LOO"]["baseline_corrected"], res["LOO"]["gain_honest"]))
    print("LOFO : baseline {:.2f}(buggy) -> {:.2f}(fixed); LASSO {:.2f}  => baseline {} cross-family".format(
        res["LOFO"]["baseline_buggy"], res["LOFO"]["baseline_corrected"], res["LOFO"]["lasso_size_removed"],
        "BEATS LASSO" if res["LOFO"]["baseline_corrected"] > res["LOFO"]["lasso_size_removed"] else "trails LASSO"))


if __name__ == "__main__":
    main()
