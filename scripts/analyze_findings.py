"""
Synthesize findings from the completed BLME study across the six research
questions in the paper:

  Q1. Distribution of benchmark Y (composite + individual benchmarks)
  Q2. How much of Y does model size alone explain?
  Q3. Which intrinsic metrics add predictive value beyond scale?
  Q4. Which BLME categories are most/least informative?
  Q5. EDG (our novel metric): does it validate?
  Q6. Within-family scaling (Pythia 70M–12B, n=8)
  Q7. Base-vs-instruct paired shifts (n=6)
  Q8. PCA / cross-family structure

Inputs:
  results/study_v1/aggregated.csv
  results/study_v1/feature_metadata.csv
  results/study_v1/analysis/{univariate, partial, lasso_features, base_vs_instruct,
                             pca_coords, pca_explained_variance}.csv

Output:
  results/study_v1/analysis/findings_report.md  (human-readable summary)

Usage:
    python scripts/analyze_findings.py
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ── Helpers ─────────────────────────────────────────────────────────────

def _fmt(x, nd=3):
    if pd.isna(x):
        return "--"
    f = float(x)
    if abs(f) < 1e-3 and f != 0:
        return f"{f:.1e}"
    if abs(f) >= 1e4:
        return f"{f:.2e}"
    return f"{f:.{nd}f}"


def _fmt_rho(rho, q=None):
    if pd.isna(rho):
        return "--"
    s = f"{rho:+.2f}"
    if q is not None and pd.notna(q):
        if q < 0.001:
            s += "***"
        elif q < 0.01:
            s += "**"
        elif q < 0.05:
            s += "*"
    return s


def _category(feat: str) -> str:
    for prefix, cat in [
        ("geometry_tokenizer_efficiency", "geom-token"),
        ("geometry_spectral", "geom-spectral"),
        ("geometry_unembedding", "geom-unembed"),
        ("geometry_hubness", "geom-hubness"),
        ("geometry_weight_norms", "geom-weight"),
        ("geometry_svd", "geom-svd"),
        ("geometry_isoscore", "geom-isoscore"),
        ("geometry_lid", "geom-lid"),
        ("geometry_collapse", "geom-collapse"),
        ("geometry_lipschitz", "geom-lipschitz"),
        ("geometry_intrinsic_dim", "geom-idim"),
        ("geometry_matrix_entropy", "geom-matH"),
        ("geometry_hsic", "geom-hsic"),
        ("geometry_rsa", "geom-rsa"),
        ("geometry_cka", "geom-cka"),
        ("geometry_correlation_dimension", "geom-corrdim"),
        ("geometry_positional_decay", "geom-posdec"),
        ("geometry_prediction_alignment", "geom-align"),
        ("geometry_contextualization", "geom-context"),
        ("geometry_neural_collapse", "geom-nc"),
        ("geometry_perplexity", "geom-ppl"),
        ("interpretability_logit_lens", "interp-ll"),
        ("interpretability_attention_entropy", "interp-attH"),
        ("interpretability_attention_rank", "interp-attR"),
        ("interpretability_induction_heads", "interp-ind"),
        ("interpretability_head_roles", "interp-hr"),
        ("interpretability_prediction_entropy", "interp-predH"),
        ("interpretability_sparsity", "interp-sparse"),
        ("interpretability_superposition", "interp-super"),
        ("interpretability_waa", "interp-waa"),
        ("interpretability_attention_graph", "interp-attG"),
        ("causality_tracing", "caus-trace"),
        ("causality_attention_knockout", "caus-ko"),
        ("causality_circuit_quality", "caus-circ"),
        ("causality_knowledge_neurons", "caus-kneu"),
        ("causality_edge_attribution", "caus-edge"),
        ("dynamics_gradient_flow", "dyn-grad"),
        ("dynamics_sharpness", "dyn-sharp"),
        ("consistency_position_sensitivity", "cons-pos"),
        ("consistency_format_robustness", "cons-fmt"),
        ("consistency_icl_slope", "cons-icl"),
        ("consistency_calibration", "cons-cal"),
        ("repe_task_vectors", "repe-tv"),
        ("repe_concept_separability", "repe-cs"),
        ("repe_refusal_direction", "repe-rd"),
        ("edg", "edg"),
    ]:
        if feat.startswith(prefix) or feat == prefix:
            return cat
    return "other"


def _major_category(feat: str) -> str:
    if feat.startswith("geometry_") or feat == "edg":
        return "geometry"
    if feat.startswith("interpretability_"):
        return "interpretability"
    if feat.startswith("causality_"):
        return "causality"
    if feat.startswith("dynamics_"):
        return "dynamics"
    if feat.startswith("consistency_"):
        return "consistency"
    if feat.startswith("repe_"):
        return "repe"
    return "other"


def _section(title: str, body: str) -> str:
    return f"\n## {title}\n\n{body}\n"


# ── Load ────────────────────────────────────────────────────────────────

def load(input_dir: Path):
    agg = pd.read_csv(input_dir / "aggregated.csv")
    uni = pd.read_csv(input_dir / "analysis" / "univariate.csv")
    par = pd.read_csv(input_dir / "analysis" / "partial.csv")
    lasso = pd.read_csv(input_dir / "analysis" / "lasso_features.csv")
    bvi = pd.read_csv(input_dir / "analysis" / "base_vs_instruct.csv")
    pca = pd.read_csv(input_dir / "analysis" / "pca_coords.csv")
    pca_ev = pd.read_csv(input_dir / "analysis" / "pca_explained_variance.csv")
    return agg, uni, par, lasso, bvi, pca, pca_ev


# ── Q1. Benchmark Y distribution ────────────────────────────────────────

def analyze_benchmark_y(agg: pd.DataFrame) -> str:
    out = []
    out.append("### Model-size range")
    n_params = agg["n_params_M"].dropna()
    out.append(f"- {len(n_params)} models, "
               f"parameter range **{n_params.min():.0f}M – {n_params.max()/1000:.1f}B** "
               f"(median {n_params.median():.0f}M).")
    out.append(f"- Log-range spans {np.log10(n_params.max() / n_params.min()):.1f} decades.")

    out.append("\n### Composite benchmark distribution")
    comp = agg[["model", "family", "n_params_M", "composite_benchmark"]].dropna()
    comp = comp.sort_values("composite_benchmark", ascending=False)
    out.append(f"- n={len(comp)} models with finite composite Y.")
    out.append(f"- Range: **{comp['composite_benchmark'].min():.3f} – {comp['composite_benchmark'].max():.3f}**, "
               f"median {comp['composite_benchmark'].median():.3f}.")
    out.append("\n**Top 5 by composite score:**")
    for _, r in comp.head(5).iterrows():
        out.append(f"- {r['model']:<22s} ({r['family']}, {r['n_params_M']:.0f}M) → {r['composite_benchmark']:.3f}")
    out.append("\n**Bottom 5 by composite score:**")
    for _, r in comp.tail(5).iterrows():
        out.append(f"- {r['model']:<22s} ({r['family']}, {r['n_params_M']:.0f}M) → {r['composite_benchmark']:.3f}")

    # Individual benchmark correlations with size
    out.append("\n### Individual benchmark Spearman vs. log(N_params)")
    log_n = np.log(agg["n_params_M"].fillna(1) * 1e6)
    bench_cols = [c for c in agg.columns if c.startswith("benchmark_") and "mmlu" in c]
    # Keep only the top-level mmlu acc (not subtasks)
    target_cols = [
        "benchmark_hellaswag_acc_norm", "benchmark_hellaswag_acc",
        "benchmark_piqa_acc", "benchmark_arc_easy_acc", "benchmark_arc_challenge_acc",
        "benchmark_winogrande_acc", "benchmark_mmlu_acc",
    ]
    target_cols = [c for c in target_cols if c in agg.columns]
    # Also include composite
    target_cols.append("composite_benchmark")
    for bc in target_cols:
        y = agg[bc].values
        mask = np.isfinite(y) & np.isfinite(log_n.values)
        if mask.sum() < 5:
            continue
        rho, _ = spearmanr(log_n.values[mask], y[mask])
        out.append(f"- {bc.replace('benchmark_', ''):<32s} ρ(log N, Y) = {rho:+.3f}  (n={int(mask.sum())})")

    return "\n".join(out)


# ── Q2/Q3. Size baseline + top predictors beyond scale ──────────────────

def analyze_predictors(agg: pd.DataFrame, uni: pd.DataFrame, par: pd.DataFrame,
                       lasso: pd.DataFrame, min_n: int = 20) -> str:
    """min_n filters out low-power correlations (default 20).

    At n<20, Spearman ρ approaches ±1.0 with meaningful probability under
    the null, so "top correlates" tables get contaminated by spurious
    perfect correlations from per-layer features that only exist for a
    handful of models. min_n=20 retains ~62% of our 32 models per feature
    as a reasonable power threshold."""
    out = []

    # Size baseline R² on composite (simple linear)
    target = "composite_benchmark"
    from sklearn.linear_model import LinearRegression
    if target in agg.columns and "n_params_M" in agg.columns:
        df = agg[[target, "n_params_M"]].dropna()
        df["log_n"] = np.log(df["n_params_M"] * 1e6)
        if len(df) >= 5:
            lr = LinearRegression().fit(df[["log_n"]], df[target])
            r2_size = lr.score(df[["log_n"]], df[target])
            rho_size, p_size = spearmanr(df["log_n"], df[target])
            out.append(f"### Size-only baseline on composite (n={len(df)})")
            out.append(f"- Linear R² = **{r2_size:.3f}**")
            out.append(f"- Spearman ρ(log N, composite) = **{rho_size:+.3f}** (p = {p_size:.1e})")

    # Univariate top correlates with composite
    uni_comp = uni[uni["benchmark"] == "composite_benchmark"].copy()
    uni_comp["abs_rho"] = uni_comp["rho"].abs()
    uni_filt = uni_comp[uni_comp["n"] >= min_n].sort_values("abs_rho", ascending=False)
    n_sig = int((uni_filt["q"] < 0.05).sum())
    out.append(f"\n### Univariate Spearman with composite (n≥{min_n})")
    out.append(f"- {len(uni_filt)} features survive the n≥{min_n} power filter "
               f"(out of {len(uni_comp)} tested).")
    out.append(f"- {n_sig} / {len(uni_filt)} of those are FDR-significant at q<0.05.")
    out.append("\n**Top 20 by |ρ| (univariate, n≥{}):**".format(min_n))
    for _, r in uni_filt.head(20).iterrows():
        out.append(f"- `{r['feature']:<55s}` {_fmt_rho(r['rho'], r.get('q'))} (n={int(r['n'])})")

    # Partial correlates with composite — the paper's main result
    par_comp = par[par["benchmark"] == "composite_benchmark"].copy()
    par_comp["abs_rho"] = par_comp["partial_rho"].abs()
    par_filt = par_comp[par_comp["n"] >= min_n].sort_values("abs_rho", ascending=False)
    n_sig_par = int((par_filt["q"] < 0.05).sum())
    out.append(f"\n### Partial correlates with composite, controlling for log(N_params) (n≥{min_n})")
    out.append(f"- {len(par_filt)} features survive the n≥{min_n} power filter.")
    out.append(f"- **{n_sig_par} / {len(par_filt)}** remain FDR-significant at q<0.05 after "
               f"partialling out log N_params — features carrying signal BEYOND model scale.")
    out.append("\n**Top 25 features ranked by |partial ρ| (n≥{}):**".format(min_n))
    for _, r in par_filt.head(25).iterrows():
        cat = _category(r["feature"])
        out.append(f"- `{r['feature']:<55s}` partial ρ={_fmt_rho(r['partial_rho'], r.get('q'))} "
                   f"(n={int(r['n'])}) [{cat}]")

    # Also report what got filtered out — for transparency
    par_lowpower = par_comp[par_comp["n"] < min_n].sort_values("abs_rho", ascending=False)
    n_inflated = int((par_lowpower["abs_rho"].fillna(0) > 0.9).sum())
    out.append(f"\n**Low-power filter removed** {len(par_lowpower)} features at n<{min_n}, "
               f"of which {n_inflated} had spurious |ρ|>0.9 (likely chance inflation).")

    # LASSO selected features
    out.append("\n### LASSO multivariate selection")
    out.append(f"- Selected **{len(lasso)}** features out of ~920 candidates.")
    out.append("- Training R² is overfit (n=32 × p≈900); see console output for held-out LOO/LOFO R².")
    out.append("\n**Top 20 LASSO coefficients (signed):**")
    for _, r in lasso.head(20).iterrows():
        cat = _category(r["feature"])
        sign = "↑" if r["coefficient"] > 0 else "↓"
        out.append(f"- {sign} `{r['feature']:<55s}` β={r['coefficient']:+.4f}  [{cat}]")

    return "\n".join(out)


# ── Q4. Which categories carry the signal? ──────────────────────────────

def analyze_category_signal(par: pd.DataFrame, min_n: int = 20) -> str:
    """Category-level signal table, filtered to features with n>=min_n to
    avoid reporting inflated spurious partial ρ from low-n features."""
    par_comp = par[par["benchmark"] == "composite_benchmark"].copy()
    par_comp = par_comp[par_comp["n"] >= min_n].copy()
    par_comp["major_cat"] = par_comp["feature"].apply(_major_category)
    par_comp["abs_partial"] = par_comp["partial_rho"].abs()

    if par_comp.empty:
        return "*(no features survive the power filter)*"

    agg = par_comp.groupby("major_cat").agg(
        n_features=("feature", "count"),
        n_fdr_sig=("q", lambda x: int((x < 0.05).sum())),
        max_abs_partial=("abs_partial", "max"),
        mean_abs_partial=("abs_partial", "mean"),
        best_feature=("feature", lambda x: par_comp.loc[
            par_comp.loc[x.index, "abs_partial"].idxmax(), "feature"
        ]),
    ).reset_index()
    agg["sig_rate"] = agg["n_fdr_sig"] / agg["n_features"].clip(lower=1)
    agg = agg.sort_values("sig_rate", ascending=False)

    out = ["### FDR-significant features per BLME major category"]
    out.append(f"(partial Spearman with composite, controlling for log N_params, n≥{min_n})\n")
    out.append("| Category | n features | FDR-sig | sig rate | max \\|partial ρ\\| | best feature |")
    out.append("|---|---:|---:|---:|---:|---|")
    for _, r in agg.iterrows():
        out.append(
            f"| {r['major_cat']} | {int(r['n_features'])} | {int(r['n_fdr_sig'])} | "
            f"{r['sig_rate']:.1%} | {r['max_abs_partial']:.2f} | `{r['best_feature']}` |"
        )
    return "\n".join(out)


# ── Q5. EDG validation ──────────────────────────────────────────────────

def analyze_edg(agg: pd.DataFrame, par: pd.DataFrame) -> str:
    out = ["### Effective Dimensionality Gradient (EDG) — novel metric"]
    edg_cols = [c for c in agg.columns if "edg" in c.lower() and "erank" not in c]
    if not edg_cols:
        out.append("EDG columns not found.")
        return "\n".join(out)

    df = agg[["model", "family", "n_params_M", "composite_benchmark"] + edg_cols].copy()
    df["log_n"] = np.log(df["n_params_M"].fillna(1) * 1e6)

    for col in edg_cols:
        d = df[[col, "composite_benchmark", "log_n"]].dropna()
        if len(d) < 5:
            continue
        rho_y, p_y = spearmanr(d[col], d["composite_benchmark"])
        # partial ρ
        row = par[(par["benchmark"] == "composite_benchmark") & (par["feature"] == col)]
        if len(row) >= 1:
            partial_rho = row.iloc[0]["partial_rho"]
            partial_q = row.iloc[0].get("q", np.nan)
        else:
            partial_rho, partial_q = np.nan, np.nan
        rho_n, _ = spearmanr(d[col], d["log_n"])
        out.append(
            f"- **{col}** (n={len(d)}): "
            f"ρ(.,Y)={rho_y:+.2f}, ρ(.,log N)={rho_n:+.2f}, "
            f"partial ρ={_fmt_rho(partial_rho, partial_q)}"
        )

    # Pythia-specific EDG scaling
    pythia = agg[agg["family"] == "pythia"].dropna(subset=["n_params_M", "edg", "composite_benchmark"])
    if len(pythia) >= 4:
        out.append(f"\n**Within Pythia (n={len(pythia)}):**")
        log_n = np.log(pythia["n_params_M"] * 1e6)
        for col in ["edg", "edg_early", "edg_late"]:
            if col not in pythia.columns:
                continue
            rho_n, p_n = spearmanr(log_n, pythia[col])
            rho_y, p_y = spearmanr(pythia[col], pythia["composite_benchmark"])
            out.append(f"- {col}: ρ(log N, {col})={rho_n:+.2f}  ρ({col}, composite)={rho_y:+.2f}")
    return "\n".join(out)


# ── Q6. Within-family (Pythia) ──────────────────────────────────────────

def analyze_within_family(agg: pd.DataFrame) -> str:
    out = ["### Within-family analysis"]
    for family, min_n in [("pythia", 4), ("gpt2", 3), ("qwen3.5", 5), ("llama3", 3)]:
        df = agg[agg["family"] == family].dropna(subset=["n_params_M", "composite_benchmark"])
        if len(df) < min_n:
            continue
        log_n = np.log(df["n_params_M"] * 1e6)
        rho_n, p_n = spearmanr(log_n, df["composite_benchmark"])
        out.append(f"\n**{family}** (n={len(df)}): ρ(log N, composite) = {rho_n:+.3f} (p = {p_n:.1e})")
        out.append("| model | N_params | composite |")
        out.append("|---|---:|---:|")
        for _, r in df.sort_values("n_params_M").iterrows():
            out.append(f"| {r['model']} | {r['n_params_M']:.0f}M | {r['composite_benchmark']:.3f} |")
    return "\n".join(out)


# ── Q7. Base vs Instruct ────────────────────────────────────────────────

def analyze_base_vs_instruct(bvi: pd.DataFrame, agg: pd.DataFrame) -> str:
    out = ["### Base → Instruct paired shifts (n=6 pairs)"]
    n_pairs_max = int(bvi["n_pairs"].max())
    unanimous_up = bvi[(bvi["sign_agreement"] == 1.0) & (bvi["mean_delta"] > 0)]
    unanimous_dn = bvi[(bvi["sign_agreement"] == 1.0) & (bvi["mean_delta"] < 0)]
    out.append(f"- n_pairs max: {n_pairs_max} (llama3-1b, qwen3.5-{{0.8, 2, 4, 9}}b, gemma4-e4b).")
    out.append(f"- **{len(unanimous_up) + len(unanimous_dn)}** of {len(bvi)} evaluated features "
               f"moved unanimously across all available pairs — {len(unanimous_up)} up, {len(unanimous_dn)} down.")

    out.append("\n**Top 15 unanimous shifts by |cross-model-standardised Δ|:**")
    unanimous = bvi[bvi["sign_agreement"] == 1.0].copy()
    unanimous = unanimous.reindex(unanimous["std_delta_xmodel"].abs().sort_values(ascending=False).index)
    for _, r in unanimous.head(15).iterrows():
        sign = "↑" if r["mean_delta"] > 0 else "↓"
        cat = _category(r["feature"])
        out.append(
            f"- {sign} `{r['feature']:<55s}` "
            f"std_Δ={r['std_delta_xmodel']:+.2f}  d={_fmt(r['cohens_d'], 2)}  [{cat}]"
        )

    # Group the unanimous shifts by category
    unanimous["major_cat"] = unanimous["feature"].apply(_major_category)
    cat_counts = unanimous["major_cat"].value_counts()
    out.append("\n**Unanimous shifts by category:**")
    for cat, n in cat_counts.items():
        out.append(f"- {cat}: {int(n)}")

    # Directional themes
    out.append("\n**Directional themes (from unanimous set):**")
    themes = {
        "Calibration degradation": ["consistency_calibration"],
        "Sharper minima (SAM, gradient norms, Hessian trace)": ["dynamics_sharpness", "dynamics_gradient_flow"],
        "Higher surface-form NLL": ["consistency_format_robustness.mean_nll", "geometry_perplexity"],
        "Lower attention entropy": ["interpretability_attention_entropy"],
        "Refusal direction emergence": ["repe_refusal_direction"],
        "Lower activation sparsity / higher kurtosis": ["interpretability_sparsity"],
    }
    for theme, prefixes in themes.items():
        subset = unanimous[unanimous["feature"].apply(lambda f: any(p in f for p in prefixes))]
        if len(subset) == 0:
            out.append(f"- {theme}: none significant")
            continue
        top = subset.reindex(subset["std_delta_xmodel"].abs().sort_values(ascending=False).index).head(3)
        out.append(f"- **{theme}** — {len(subset)} features unanimous, top:")
        for _, r in top.iterrows():
            sign = "↑" if r["mean_delta"] > 0 else "↓"
            out.append(f"  - {sign} `{r['feature']}` std_Δ={r['std_delta_xmodel']:+.2f}")
    return "\n".join(out)


# ── Q8. PCA / clustering ────────────────────────────────────────────────

def analyze_pca(pca: pd.DataFrame, pca_ev: pd.DataFrame) -> str:
    out = ["### PCA of the feature matrix"]
    if pca_ev is not None and not pca_ev.empty:
        out.append("Explained variance:")
        for _, r in pca_ev.iterrows():
            out.append(f"- {r['component']}: {r['explained_variance_ratio']:.2%}")

    # Check whether PC1 correlates with log N_params (i.e., PC1 is "size")
    if "log_n_params" in pca.columns and "PC1" in pca.columns:
        valid = pca[["PC1", "PC2", "log_n_params", "composite_benchmark"]].dropna()
        if len(valid) >= 5:
            rho_pc1_n, _ = spearmanr(valid["PC1"], valid["log_n_params"])
            rho_pc2_n, _ = spearmanr(valid["PC2"], valid["log_n_params"])
            rho_pc1_y, _ = spearmanr(valid["PC1"], valid["composite_benchmark"])
            rho_pc2_y, _ = spearmanr(valid["PC2"], valid["composite_benchmark"])
            out.append(f"\n- ρ(PC1, log N_params) = {rho_pc1_n:+.2f}  ρ(PC1, composite) = {rho_pc1_y:+.2f}")
            out.append(f"- ρ(PC2, log N_params) = {rho_pc2_n:+.2f}  ρ(PC2, composite) = {rho_pc2_y:+.2f}")

    if "family" in pca.columns:
        out.append("\n**Family centroids in PCA space:**")
        out.append("| Family | n | PC1 mean | PC2 mean | PC3 mean |")
        out.append("|---|---:|---:|---:|---:|")
        for fam, sub in pca.groupby("family"):
            pc3 = sub["PC3"].mean() if "PC3" in sub.columns else np.nan
            out.append(
                f"| {fam} | {len(sub)} | {sub['PC1'].mean():+.2f} | "
                f"{sub['PC2'].mean():+.2f} | {pc3:+.2f} |"
            )
    return "\n".join(out)


# ── Misc observations ───────────────────────────────────────────────────

def analyze_misc(agg: pd.DataFrame) -> str:
    out = ["### Notable observations"]
    # 27B beats 31B
    qwen27 = agg[agg["model"] == "qwen3.5-27b-it"]
    gemma31 = agg[agg["model"] == "gemma4-31b"]
    if len(qwen27) == 1 and len(gemma31) == 1:
        q = qwen27.iloc[0]
        g = gemma31.iloc[0]
        out.append(f"- **qwen3.5-27b-it** (27B) composite = {q['composite_benchmark']:.3f}  vs  "
                   f"**gemma4-31b** (31B) composite = {g['composite_benchmark']:.3f}")
        out.append(f"  Smaller-better-than-larger: {q['composite_benchmark'] > g['composite_benchmark']}")
        for bc in ["benchmark_mmlu_acc", "benchmark_hellaswag_acc_norm",
                   "benchmark_arc_challenge_acc"]:
            if bc in q.index and bc in g.index and pd.notna(q[bc]) and pd.notna(g[bc]):
                diff = q[bc] - g[bc]
                out.append(f"  - {bc.replace('benchmark_','')}: qwen {q[bc]:.3f}  gemma {g[bc]:.3f}  "
                           f"(qwen − gemma = {diff:+.3f})")

    # Calibration gap: which models have worst ECE?
    if "consistency_calibration.ece" in agg.columns:
        ece = agg[["model", "family", "n_params_M", "consistency_calibration.ece"]].dropna()
        ece = ece.sort_values("consistency_calibration.ece", ascending=False)
        out.append("\n**Worst ECE (top 5) — poorly calibrated models:**")
        for _, r in ece.head(5).iterrows():
            out.append(f"- {r['model']} ({r['family']}, {r['n_params_M']:.0f}M): ECE = {r['consistency_calibration.ece']:.3f}")
        out.append("\n**Best ECE (top 5):**")
        for _, r in ece.tail(5).iterrows():
            out.append(f"- {r['model']} ({r['family']}, {r['n_params_M']:.0f}M): ECE = {r['consistency_calibration.ece']:.3f}")

    # Refusal direction: which models have clearest refusal direction?
    rd_cols = [c for c in agg.columns if "refusal_direction" in c and "best_layer_separability_auc" in c]
    if rd_cols:
        col = rd_cols[0]
        rd = agg[["model", "family", col]].dropna()
        rd = rd.sort_values(col, ascending=False)
        out.append(f"\n**Highest refusal-direction separability ({col}):**")
        for _, r in rd.head(5).iterrows():
            out.append(f"- {r['model']} ({r['family']}): {r[col]:.3f}")

    return "\n".join(out)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="results/study_v1")
    ap.add_argument("--output", default=None,
                    help="Output markdown file (default: <input>/analysis/findings_report.md)")
    ap.add_argument("--min-n", type=int, default=20,
                    help="Minimum sample size for a feature to appear in top-correlate "
                         "tables (default 20). Filters out low-power correlations that "
                         "spuriously hit ±1.0 at small n.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_path = Path(args.output) if args.output else input_dir / "analysis" / "findings_report.md"

    agg, uni, par, lasso, bvi, pca, pca_ev = load(input_dir)

    parts = ["# BLME Study — Findings Report",
             f"\n*Generated from {input_dir}/aggregated.csv ({len(agg)} models, "
             f"{len(agg.columns)} columns) and analysis/*.csv outputs.*",
             f"\n*Low-power filter: features with n < {args.min_n} excluded from top "
             f"correlates to prevent spurious ±1.0 ρ values.*"]
    parts.append(_section("Q1. Benchmark Y distribution", analyze_benchmark_y(agg)))
    parts.append(_section("Q2/Q3. Size baseline & top predictors beyond scale",
                          analyze_predictors(agg, uni, par, lasso, min_n=args.min_n)))
    parts.append(_section("Q4. Category-level signal",
                          analyze_category_signal(par, min_n=args.min_n)))
    parts.append(_section("Q5. EDG validation", analyze_edg(agg, par)))
    parts.append(_section("Q6. Within-family analysis", analyze_within_family(agg)))
    parts.append(_section("Q7. Base vs Instruct paired shifts", analyze_base_vs_instruct(bvi, agg)))
    parts.append(_section("Q8. PCA / cross-family structure", analyze_pca(pca, pca_ev)))
    parts.append(_section("Misc. notable observations", analyze_misc(agg)))

    report = "\n".join(parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Wrote findings report: {out_path}")
    print(f"  ({len(report)} chars, {report.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
