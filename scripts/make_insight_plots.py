"""
Exploratory insight plots for the BLME study (works with partial data).

Unlike make_figures.py (which is for the final paper), this script produces
plots for in-progress exploration that don't require benchmark scores:

  insight_edg_per_family.pdf          — EDG distribution by model family
  insight_scaling_grid.pdf            — 6 key metrics vs log(params) across all models
  insight_family_comparison.pdf       — heat-grid of mean metric values by family
  insight_task_failure_rates.pdf      — which BLME tasks fail most often, by model
  insight_metric_correlations.pdf     — intra-metric Spearman correlation heatmap
  insight_compression_profiles.pdf    — per-layer erank for one model per family
  insight_instruct_vs_base.pdf        — differences for base/instruct pairs we have

Usage:
    python scripts/make_insight_plots.py --input-dir results/study_v1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
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
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

FAMILY_COLORS = {
    "gpt2": "#4C72B0", "pythia": "#DD8452", "llama3": "#55A868",
    "qwen3.5": "#C44E52", "gemma4": "#8172B2",
    "olmo": "#CCB974", "tinyllama": "#64B5CD", "phi": "#937860",
}
FAMILY_ORDER = ["gpt2", "pythia", "llama3", "qwen3.5", "gemma4", "olmo", "tinyllama", "phi"]


def insight_edg_per_family(agg: pd.DataFrame, out_path: Path):
    """Box-plot of EDG by model family + scatter by individual model."""
    if "edg" not in agg.columns:
        print("  skip EDG per family (no edg column)")
        return
    df = agg.dropna(subset=["edg"])
    if df.empty:
        print("  skip EDG per family (no data)")
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    families = [f for f in FAMILY_ORDER if f in df["family"].values]
    for i, fam in enumerate(families):
        vals = df[df["family"] == fam]["edg"].values
        if len(vals) == 0:
            continue
        ax.scatter([i] * len(vals), vals, s=60, alpha=0.8,
                   color=FAMILY_COLORS.get(fam, "#555"),
                   edgecolors="black", linewidths=0.5, zorder=3)
        # Mean marker
        if len(vals) > 1:
            ax.plot([i - 0.15, i + 0.15], [vals.mean()] * 2, "k-", lw=2, zorder=4)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=20)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("EDG (Spearman ρ(layer, erank ratio))")
    ax.set_title("Effective Dimensionality Gradient by family")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  wrote {out_path}")


def insight_scaling_grid(agg: pd.DataFrame, out_path: Path):
    """6-panel grid of key intrinsic metrics vs log(params), colored by family."""
    if "log_n_params" not in agg.columns:
        print("  skip scaling grid (no log_n_params)")
        return

    metrics = [
        ("edg", "EDG", False, None),
        ("geometry_spectral.avg_alpha", "Power-law α", False, None),
        ("geometry_isoscore.isoscore", "IsoScore", False, (0, 1)),
        ("dynamics_sharpness.hutchinson_trace_per_param", "Hessian trace / param", True, None),
        ("interpretability_attention_entropy.avg_normalized_entropy_total", "Normalized attention entropy", False, None),
        ("geometry_weight_norms.norm_uniformity", "Weight norm uniformity", False, None),
    ]
    metrics = [m for m in metrics if m[0] in agg.columns and agg[m[0]].notna().sum() >= 3]
    if not metrics:
        print("  skip scaling grid (no metrics)")
        return

    n = len(metrics)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for idx, (key, label, logy, ylim) in enumerate(metrics):
        ax = axes[idx // cols][idx % cols]
        for fam in FAMILY_ORDER:
            sub = agg[(agg["family"] == fam) & agg[key].notna() & agg["log_n_params"].notna()]
            if sub.empty:
                continue
            sub = sub.sort_values("log_n_params")
            ax.plot(sub["log_n_params"], sub[key], "-o",
                    color=FAMILY_COLORS.get(fam, "#555"),
                    markersize=6, linewidth=1.2, label=fam, alpha=0.85,
                    markeredgecolor="black", markeredgewidth=0.4)
        ax.set_xlabel(r"$\log N_{\text{params}}$")
        ax.set_ylabel(label)
        if logy:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(ylim)
    # Hide unused
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    # Shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.02),
                   fontsize=8, ncol=2, framealpha=0.9)
    fig.suptitle("Intrinsic metrics vs model size, coloured by family", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  wrote {out_path}")


def insight_family_comparison(agg: pd.DataFrame, out_path: Path):
    """Heatmap of mean metric values by family (z-scored)."""
    metrics = [
        ("edg", "EDG"),
        ("geometry_isoscore.isoscore", "IsoScore"),
        ("geometry_spectral.avg_alpha", "Power α"),
        ("geometry_spectral.avg_stable_rank", "Stable rank"),
        ("geometry_matrix_entropy.mean_matrix_entropy", "Matrix entropy"),
        ("geometry_tokenizer_efficiency.fertility", "Fertility"),
        ("interpretability_sparsity.global_mean_l0", "MLP sparsity"),
        ("interpretability_attention_entropy.avg_normalized_entropy_total", "Attn entropy"),
        ("dynamics_sharpness.hutchinson_trace_per_param", "Sharpness"),
        ("dynamics_gradient_flow.gradient_flow_entropy", "Grad flow entropy"),
        ("repe_refusal_direction.best_layer_separability_auc", "Refusal AUC"),
        ("consistency_icl_slope.icl_slope", "ICL slope"),
    ]
    metrics = [(k, lbl) for k, lbl in metrics if k in agg.columns]
    if not metrics:
        print("  skip family comparison (no metrics)")
        return

    # Per-family means
    families = [f for f in FAMILY_ORDER if f in agg["family"].values]
    mat = np.full((len(metrics), len(families)), np.nan)
    for i, (k, _) in enumerate(metrics):
        for j, fam in enumerate(families):
            vals = agg[agg["family"] == fam][k].dropna()
            if len(vals) > 0:
                mat[i, j] = vals.mean()

    # Z-score each row (metric) for visual comparison
    mat_z = mat.copy()
    for i in range(mat.shape[0]):
        row = mat[i]
        valid = ~np.isnan(row)
        if valid.sum() >= 2 and np.nanstd(row) > 0:
            mat_z[i] = (row - np.nanmean(row)) / np.nanstd(row)

    fig, ax = plt.subplots(figsize=(1.2 * len(families) + 2, 0.4 * len(metrics) + 2))
    im = ax.imshow(mat_z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([lbl for _, lbl in metrics])
    # Annotate with raw values
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(mat_z[i, j]) > 1 else "black")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("z-score")
    ax.set_title("Mean metric value by family (z-scored across families)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  wrote {out_path}")


def insight_task_failure_rates(input_dir: Path, out_path: Path):
    """Per-task failure heatmap across models."""
    blme_dir = input_dir / "blme"
    tasks_all = set()
    per_model = {}
    for d in sorted(blme_dir.iterdir()):
        if not (d / "results.json").exists():
            continue
        with open(d / "results.json") as f:
            data = json.load(f)
        requested = set(data.get("config", {}).get("tasks_requested", []))
        actual = set(k for k, v in data.get("results", {}).items()
                     if isinstance(v, dict) and "error" not in v)
        tasks_all |= requested
        per_model[d.name] = (requested, actual)

    if not per_model:
        print("  skip task failure rates (no data)")
        return

    tasks_sorted = sorted(tasks_all)
    models_sorted = sorted(per_model.keys())
    mat = np.zeros((len(tasks_sorted), len(models_sorted)), dtype=np.int8)
    for j, m in enumerate(models_sorted):
        req, act = per_model[m]
        for i, t in enumerate(tasks_sorted):
            if t not in req:
                mat[i, j] = -1  # not requested
            elif t in act:
                mat[i, j] = 1  # success
            else:
                mat[i, j] = 0  # failed

    fig, ax = plt.subplots(figsize=(0.30 * len(models_sorted) + 2,
                                     0.22 * len(tasks_sorted) + 2))
    cmap = mcolors.ListedColormap(["#ccc", "#E74C3C", "#2ECC71"])
    ax.imshow(mat, cmap=cmap, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(models_sorted)))
    ax.set_xticklabels(models_sorted, rotation=90, fontsize=6)
    ax.set_yticks(range(len(tasks_sorted)))
    ax.set_yticklabels(tasks_sorted, fontsize=6)
    ax.set_title("Task success/failure per model (green = success, red = fail, gray = n/a)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  wrote {out_path}")

    # Also print task failure summary
    fail_rate = {}
    for i, t in enumerate(tasks_sorted):
        total = (mat[i] != -1).sum()
        failed = (mat[i] == 0).sum()
        if total > 0:
            fail_rate[t] = failed / total
    print(f"\n  Task failure rates:")
    for t, r in sorted(fail_rate.items(), key=lambda x: -x[1]):
        if r > 0:
            print(f"    {t}: {r:.0%}")


def insight_metric_correlations(agg: pd.DataFrame, out_path: Path):
    """Heatmap of pairwise Spearman correlations between top-N metrics with ≥20 non-NaN."""
    from scipy.stats import spearmanr
    # Find numeric columns with enough data
    exclude_prefixes = ("benchmark_",)
    exclude_exact = {"model", "family", "hf_id", "dtype", "n_gpus", "purpose",
                     "d_model", "n_layers", "n_heads", "vocab_size", "n_params_est",
                     "n_params_M", "log_n_params", "composite_benchmark"}
    cols = []
    for c in agg.columns:
        if c in exclude_exact or any(c.startswith(p) for p in exclude_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(agg[c]) and agg[c].notna().sum() >= max(15, len(agg) // 2):
            cols.append(c)

    # Pick top 25 columns by variance (most informative)
    if len(cols) == 0:
        print("  skip metric correlations (no eligible columns)")
        return
    variances = agg[cols].var(numeric_only=True).dropna()
    top = variances.nlargest(min(25, len(variances))).index.tolist()

    # Compute correlation matrix
    mat = np.full((len(top), len(top)), np.nan)
    for i, a in enumerate(top):
        for j, b in enumerate(top):
            x = agg[a].values.astype(float)
            y = agg[b].values.astype(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 5:
                continue
            r, _ = spearmanr(x[m], y[m])
            mat[i, j] = r if np.isfinite(r) else np.nan

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([t.replace("geometry_", "g_").replace("interpretability_", "i_")
                        .replace("dynamics_", "d_").replace("causality_", "c_")
                        .replace("consistency_", "cs_").replace("repe_", "r_")
                        for t in top], rotation=90, fontsize=6)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([t.replace("geometry_", "g_").replace("interpretability_", "i_")
                        .replace("dynamics_", "d_").replace("causality_", "c_")
                        .replace("consistency_", "cs_").replace("repe_", "r_")
                        for t in top], fontsize=6)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03)
    cbar.set_label("Spearman ρ")
    ax.set_title(f"Top-{len(top)} metric pairwise correlations (n ≥ {len(agg)//2} models)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  wrote {out_path}")


def insight_compression_profiles(input_dir: Path, agg: pd.DataFrame, out_path: Path):
    """Per-layer effective rank for representative models from each family."""
    blme_dir = input_dir / "blme"
    picks = {
        "gpt2": "gpt2-medium",
        "pythia": "pythia-1.4b",
        "llama3": "llama3-1b",
        "qwen3.5": "qwen3.5-4b-it",
        "gemma4": "gemma4-e2b",
        "olmo": "olmo-1b",
        "phi": "phi-2",
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = 0
    for fam, model in picks.items():
        path = blme_dir / model / "results.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        collapse = data.get("results", {}).get("geometry_collapse", {})
        erank = collapse.get("erank_per_layer", [])
        if not erank:
            continue
        xs = np.linspace(0, 1, len(erank))
        # Normalise per-model by max erank
        d = max(erank) if max(erank) > 0 else 1.0
        ys = np.asarray(erank) / d
        ax.plot(xs, ys, "-o", markersize=4, label=model,
                color=FAMILY_COLORS.get(fam, "#555"), lw=1.5, alpha=0.85)
        plotted += 1
    if plotted == 0:
        print("  skip compression profiles (no data)")
        plt.close()
        return
    ax.set_xlabel("Normalised depth")
    ax.set_ylabel("Effective rank / max per model")
    ax.set_title("Representation compression across layers (representative models)")
    ax.legend(fontsize=7, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  wrote {out_path}")


def insight_instruct_vs_base(agg: pd.DataFrame, out_path: Path):
    """Diff plot for base-vs-instruct pairs on key metrics."""
    pairs = []
    for _, row in agg.iterrows():
        name = row["model"]
        if name.endswith("-it"):
            base = agg[agg["model"] == name[:-3]]
            if len(base) == 1:
                pairs.append((base.iloc[0], row))
    if not pairs:
        print("  skip instruct vs base (no pairs)")
        return

    metrics = [
        ("edg", "EDG"),
        ("geometry_isoscore.isoscore", "IsoScore"),
        ("interpretability_attention_entropy.avg_normalized_entropy_total", "Attn entropy"),
        ("interpretability_sparsity.global_mean_l0", "MLP $L_0$"),
        ("repe_refusal_direction.best_layer_separability_auc", "Refusal AUC"),
        ("dynamics_sharpness.hutchinson_trace_per_param", "Sharpness"),
    ]
    metrics = [(k, lbl) for k, lbl in metrics if k in agg.columns]
    if not metrics:
        print("  skip instruct vs base (no metrics)")
        return

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 3.2), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (key, label) in zip(axes, metrics):
        for base, it in pairs:
            b, i = base.get(key), it.get(key)
            if pd.notna(b) and pd.notna(i):
                family = base["family"]
                ax.plot([0, 1], [b, i], "-o", markersize=5, lw=1.2,
                        color=FAMILY_COLORS.get(family, "#555"), alpha=0.85)
                ax.text(1.05, i, base["model"].replace("-", "\n"),
                        fontsize=5, va="center", ha="left", alpha=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Base", "Inst."])
        ax.set_title(label, fontsize=9)
        ax.set_xlim(-0.3, 1.6)
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
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "insights"
    output_dir.mkdir(parents=True, exist_ok=True)

    agg_path = input_dir / "aggregated.csv"
    if not agg_path.exists():
        print(f"ERROR: {agg_path} not found. Run aggregate_results.py first.")
        return
    agg = pd.read_csv(agg_path)
    print(f"Loaded {len(agg)} models x {len(agg.columns)} columns")

    insight_edg_per_family(agg, output_dir / "insight_edg_per_family.pdf")
    insight_scaling_grid(agg, output_dir / "insight_scaling_grid.pdf")
    insight_family_comparison(agg, output_dir / "insight_family_comparison.pdf")
    insight_task_failure_rates(input_dir, output_dir / "insight_task_failure_rates.pdf")
    insight_metric_correlations(agg, output_dir / "insight_metric_correlations.pdf")
    insight_compression_profiles(input_dir, agg, output_dir / "insight_compression_profiles.pdf")
    insight_instruct_vs_base(agg, output_dir / "insight_instruct_vs_base.pdf")

    print(f"\nInsight plots written to {output_dir}/")


if __name__ == "__main__":
    main()
