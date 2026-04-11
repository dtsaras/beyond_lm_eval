"""
Generate publication-quality figures for the NeurIPS paper.

Figures produced:
  fig_correlation_heatmap.pdf   — univariate + partial correlation heatmap
  fig_edg_validation.pdf         — EDG vs composite benchmark + Pythia scaling
  fig_feature_importance.pdf     — LASSO top features (horizontal bar)
  fig_pca_clustering.pdf         — PCA scatter coloured by family
  fig_base_vs_instruct.pdf       — paired differences for base/instruct models
  fig_within_family_scaling.pdf  — metric trajectories within Pythia
  fig_compression_profile.pdf    — per-layer erank profiles for ~6 key models

Usage:
    python scripts/make_figures.py --input-dir results/study_v1
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Set matplotlib style for NeurIPS-quality figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # TrueType fonts for camera-ready
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Colour scheme: one colour per model family
FAMILY_COLORS = {
    "gpt2": "#4C72B0",       # muted blue
    "pythia": "#DD8452",     # orange
    "llama3": "#55A868",     # green
    "qwen3.5": "#C44E52",    # red
    "gemma4": "#8172B2",     # purple
    "olmo": "#CCB974",       # yellow
    "tinyllama": "#64B5CD",  # teal
    "phi": "#937860",        # brown
}


def _load(input_dir: Path):
    agg = pd.read_csv(input_dir / "aggregated.csv")
    feat_meta_path = input_dir / "feature_metadata.csv"
    feat_meta = pd.read_csv(feat_meta_path) if feat_meta_path.exists() else pd.DataFrame()
    analysis_dir = input_dir / "analysis"
    univariate = pd.read_csv(analysis_dir / "univariate.csv") if (analysis_dir / "univariate.csv").exists() else pd.DataFrame()
    partial = pd.read_csv(analysis_dir / "partial.csv") if (analysis_dir / "partial.csv").exists() else pd.DataFrame()
    lasso = pd.read_csv(analysis_dir / "lasso_features.csv") if (analysis_dir / "lasso_features.csv").exists() else pd.DataFrame()
    pca_coords = pd.read_csv(analysis_dir / "pca_coords.csv") if (analysis_dir / "pca_coords.csv").exists() else pd.DataFrame()
    return agg, feat_meta, univariate, partial, lasso, pca_coords


def _sig_matrix(df: pd.DataFrame, value_col: str, top_features: list, benchmarks: list):
    """Build a (features x benchmarks) matrix of correlation values."""
    mat = pd.DataFrame(index=top_features, columns=benchmarks, dtype=np.float64)
    for _, row in df.iterrows():
        f = row["feature"]
        b = row["benchmark"]
        if f in mat.index and b in mat.columns:
            mat.loc[f, b] = row[value_col]
    return mat.astype(float)


def fig_correlation_heatmap(input_dir: Path, univariate: pd.DataFrame,
                             partial: pd.DataFrame, out_path: Path,
                             max_features: int = 30):
    """Two-panel heatmap: univariate (left) and partial (right) Spearman correlations."""
    if univariate.empty or partial.empty:
        print("  skipping correlation heatmap (no data)")
        return

    # Select features by max |rho| across benchmarks (partial rho priority)
    partial_agg = partial.groupby("feature")["partial_rho"].apply(
        lambda x: np.nanmax(np.abs(x.values))
    ).sort_values(ascending=False)
    top_features = partial_agg.head(max_features).index.tolist()
    benchmarks = sorted(partial["benchmark"].unique().tolist())
    # Put composite at the end
    if "composite_benchmark" in benchmarks:
        benchmarks.remove("composite_benchmark")
        benchmarks.append("composite_benchmark")

    uni_mat = _sig_matrix(univariate, "rho", top_features, benchmarks)
    par_mat = _sig_matrix(partial, "partial_rho", top_features, benchmarks)

    fig, axes = plt.subplots(1, 2, figsize=(10, 0.28 * len(top_features) + 2),
                              gridspec_kw={"wspace": 0.08})
    for ax, mat, title in zip(axes, [uni_mat, par_mat],
                                ["Univariate Spearman", r"Partial Spearman $|\log N$"]):
        im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(benchmarks)))
        ax.set_xticklabels([b.replace("benchmark_", "").replace(",acc", "") for b in benchmarks],
                           rotation=45, ha="right")
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features if ax is axes[0] else [], fontsize=7)
        ax.set_title(title)
        ax.grid(False)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Spearman $\\rho$")
    plt.savefig(out_path)
    plt.close()
    print(f"  wrote {out_path}")


def fig_edg_validation(input_dir: Path, agg: pd.DataFrame, out_path: Path):
    """Two-panel: (a) EDG vs composite benchmark across all models;
    (b) EDG within Pythia scaling series."""
    if "edg" not in agg.columns or "composite_benchmark" not in agg.columns:
        print("  skipping EDG figure (no data)")
        return

    valid = agg.dropna(subset=["edg", "composite_benchmark"])
    if len(valid) < 3:
        print("  skipping EDG figure (too few models)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))

    # Panel A: EDG vs composite benchmark
    ax = axes[0]
    for family, group in valid.groupby("family"):
        ax.scatter(group["edg"], group["composite_benchmark"],
                   s=40, c=FAMILY_COLORS.get(family, "#999"),
                   label=family, edgecolors="black", linewidths=0.5, alpha=0.85)
    # Fit line
    from scipy.stats import spearmanr
    rho, p = spearmanr(valid["edg"], valid["composite_benchmark"])
    ax.set_xlabel("Effective Dimensionality Gradient (EDG)")
    ax.set_ylabel("Composite benchmark score")
    ax.set_title(f"(a) EDG predicts capability ($\\rho={rho:.2f}$, $p={p:.1e}$)")
    ax.legend(fontsize=6, loc="best", framealpha=0.85)

    # Panel B: EDG within Pythia scaling
    pythia = agg[agg["family"] == "pythia"].sort_values("log_n_params")
    if len(pythia) >= 3 and "edg" in pythia.columns:
        ax = axes[1]
        ax.plot(pythia["log_n_params"], pythia["edg"], "-o",
                color=FAMILY_COLORS["pythia"], linewidth=1.5, markersize=6)
        ax.set_xlabel(r"$\log$ parameter count")
        ax.set_ylabel("EDG")
        ax.set_title("(b) EDG scales with model size (Pythia)")
        for _, row in pythia.iterrows():
            ax.annotate(row["model"].replace("pythia-", ""),
                        (row["log_n_params"], row["edg"]),
                        xytext=(5, 0), textcoords="offset points",
                        fontsize=6, alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  wrote {out_path}")


def fig_feature_importance(input_dir: Path, lasso: pd.DataFrame, out_path: Path,
                            top_n: int = 15):
    """Horizontal bar chart of top LASSO-selected features."""
    if lasso.empty:
        print("  skipping feature importance (no LASSO data)")
        return

    top = lasso.head(top_n).iloc[::-1]  # reverse so biggest is on top
    fig, ax = plt.subplots(figsize=(5.5, max(2.5, 0.28 * len(top))))
    colors = ["#C44E52" if c < 0 else "#4C72B0" for c in top["coefficient"]]
    ax.barh(range(len(top)), top["coefficient"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized LASSO coefficient")
    ax.set_title(f"Top {len(top)} features predicting composite benchmark score")
    plt.savefig(out_path)
    plt.close()
    print(f"  wrote {out_path}")


def fig_pca_clustering(input_dir: Path, pca_coords: pd.DataFrame, out_path: Path):
    """PCA scatter coloured by family, sized by log(params)."""
    if pca_coords.empty or "PC1" not in pca_coords.columns:
        print("  skipping PCA (no data)")
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    size_scale = 30
    if "log_n_params" in pca_coords.columns:
        lp = pca_coords["log_n_params"]
        sizes = size_scale + 80 * (lp - lp.min()) / max(1e-6, (lp.max() - lp.min()))
    else:
        sizes = 60

    for family, group in pca_coords.groupby("family"):
        ax.scatter(group["PC1"], group["PC2"],
                   s=(sizes[group.index] if hasattr(sizes, "__getitem__") else sizes),
                   c=FAMILY_COLORS.get(family, "#999"),
                   label=family, edgecolors="black", linewidths=0.5, alpha=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Model landscape by intrinsic properties (PCA)")
    ax.legend(fontsize=7, loc="best", framealpha=0.85)
    plt.savefig(out_path)
    plt.close()
    print(f"  wrote {out_path}")


def fig_base_vs_instruct(input_dir: Path, agg: pd.DataFrame, out_path: Path):
    """Paired differences in key metrics for base-vs-instruct pairs."""
    # Find base/instruct pairs
    pairs = []
    for _, row in agg.iterrows():
        name = row["model"]
        if name.endswith("-it"):
            base_name = name[:-3]
            base_row = agg[agg["model"] == base_name]
            if len(base_row) == 1:
                pairs.append((base_row.iloc[0], row))
    if not pairs:
        print("  skipping base vs instruct (no pairs)")
        return

    # Pick a handful of interesting metrics
    metrics = [
        ("edg", "EDG"),
        ("interpretability_attention_entropy.avg_entropy_total", "Attention entropy"),
        ("interpretability_sparsity.global_mean_l0", "MLP sparsity $L_0$"),
        ("repe_refusal_direction.best_layer_separability_auc", "Refusal direction AUROC"),
        ("geometry_isoscore.isoscore", "IsoScore"),
        ("dynamics_sharpness.hutchinson_trace_per_param", "Hessian trace per param"),
    ]
    metrics = [(k, lbl) for k, lbl in metrics if k in agg.columns]

    if not metrics:
        print("  skipping base vs instruct (no metrics available)")
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(2.2 * len(metrics), 3.0), sharey=False)
    if len(metrics) == 1:
        axes = [axes]
    for ax, (key, label) in zip(axes, metrics):
        base_vals = [p[0][key] for p in pairs if pd.notna(p[0][key]) and pd.notna(p[1][key])]
        it_vals = [p[1][key] for p in pairs if pd.notna(p[0][key]) and pd.notna(p[1][key])]
        names = [p[0]["model"] for p in pairs if pd.notna(p[0][key]) and pd.notna(p[1][key])]
        xs_base = np.zeros(len(base_vals))
        xs_it = np.ones(len(base_vals))
        for b, i, n in zip(base_vals, it_vals, names):
            ax.plot([0, 1], [b, i], "-o", alpha=0.7, markersize=5,
                    color=FAMILY_COLORS.get(n.split("-")[0].replace("3.5", "3.5"), "#555"))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Base", "Instruct"], fontsize=8)
        ax.set_title(label, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  wrote {out_path}")


def fig_compression_profile(input_dir: Path, agg: pd.DataFrame, out_path: Path):
    """Per-layer effective rank profile for selected representative models."""
    # We need the raw erank_per_layer, which lives in individual results.json files
    import json
    blme_dir = input_dir / "blme"

    # Select representative models (one per family at similar sizes)
    picks = ["gpt2-small", "pythia-1.4b", "llama3-1b", "qwen3.5-2b", "gemma4-e2b", "olmo-1b"]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for model in picks:
        path = blme_dir / model / "results.json"
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        collapse = data.get("results", {}).get("geometry_collapse", {})
        erank = collapse.get("erank_per_layer", [])
        if not erank:
            continue
        family = model.split("-")[0].split(".")[0]
        if family == "qwen3":
            family = "qwen3.5"
        xs = np.linspace(0, 1, len(erank))
        # Normalise by model width (approximate — use d_model from metadata)
        d_model = agg.loc[agg["model"] == model, "d_model"].values
        d = d_model[0] if len(d_model) > 0 and d_model[0] > 0 else 1
        ys = np.asarray(erank) / d
        ax.plot(xs, ys, "-o", markersize=4,
                label=model, color=FAMILY_COLORS.get(family, "#555"),
                linewidth=1.5, alpha=0.85)
    ax.set_xlabel("Normalised depth")
    ax.set_ylabel("Effective rank / $d_{\\mathrm{model}}$")
    ax.set_title("Compression profile across layers")
    ax.legend(fontsize=7, loc="best", framealpha=0.85)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  wrote {out_path}")


def fig_within_family_scaling(input_dir: Path, agg: pd.DataFrame, out_path: Path):
    """Key metric trajectories across the Pythia scaling series."""
    pythia = agg[agg["family"] == "pythia"].sort_values("log_n_params")
    if len(pythia) < 4:
        print("  skipping scaling plot (need ≥4 Pythia models)")
        return

    metrics = [
        ("edg", "EDG", False),
        ("geometry_spectral.avg_alpha", "Power-law $\\alpha$", False),
        ("dynamics_sharpness.hutchinson_trace_per_param", "Hessian trace / param", True),  # log y
        ("composite_benchmark", "Composite benchmark", False),
    ]
    metrics = [(k, lbl, lg) for k, lbl, lg in metrics if k in pythia.columns]

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 3.0))
    if n == 1:
        axes = [axes]
    for ax, (key, label, logy) in zip(axes, metrics):
        ax.plot(pythia["log_n_params"], pythia[key], "-o",
                color=FAMILY_COLORS["pythia"], linewidth=1.5, markersize=6)
        ax.set_xlabel(r"$\log N_{\mathrm{params}}$")
        ax.set_ylabel(label)
        if logy:
            ax.set_yscale("log")
    fig.suptitle("Pythia scaling series", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="results/study_v1")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    agg, feat_meta, univariate, partial, lasso, pca_coords = _load(input_dir)

    print(f"Generating figures to {output_dir}/")
    fig_correlation_heatmap(input_dir, univariate, partial, output_dir / "fig_correlation_heatmap.pdf")
    fig_edg_validation(input_dir, agg, output_dir / "fig_edg_validation.pdf")
    fig_feature_importance(input_dir, lasso, output_dir / "fig_feature_importance.pdf")
    fig_pca_clustering(input_dir, pca_coords, output_dir / "fig_pca_clustering.pdf")
    fig_base_vs_instruct(input_dir, agg, output_dir / "fig_base_vs_instruct.pdf")
    fig_compression_profile(input_dir, agg, output_dir / "fig_compression_profile.pdf")
    fig_within_family_scaling(input_dir, agg, output_dir / "fig_within_family_scaling.pdf")

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
